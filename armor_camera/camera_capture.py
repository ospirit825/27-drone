#!/usr/bin/env python3
"""GC4653 camera lifecycle and latest-frame capture for this LicheeRV Nano.

Standalone: python3 camera_capture.py --check
            python3 camera_capture.py --snapshot camera_test.jpg
API: with GC4653Camera() as camera: seq, bgr, timestamp = camera.read()
"""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time
import cv2


class LatestFrames:
    """Read continuously, retaining one decoded BGR frame instead of a queue."""
    def __init__(self, source):
        os.environ.setdefault('OPENCV_FFMPEG_CAPTURE_OPTIONS','rtsp_transport;tcp')
        self.capture=cv2.VideoCapture()
        try:
            if str(source).startswith('rtsp://'):
                self.capture.open(source,cv2.CAP_FFMPEG,[cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,3000,cv2.CAP_PROP_READ_TIMEOUT_MSEC,3000])
            else:self.capture.open(int(source) if str(source).isdigit() else source)
            if not self.capture.isOpened():raise RuntimeError(f'Cannot open camera/stream: {source}')
        except BaseException:
            self.capture.release();raise
        self.lock=threading.Lock();self.frame=None;self.seq=0;self.stamp=0.;self.error=None
        self.stop=threading.Event();self.worker=threading.Thread(target=self._run,daemon=True);self.worker.start()

    def _run(self):
        try:
            while not self.stop.is_set():
                ok,frame=self.capture.read()
                if not ok:
                    self.error='Camera read failed or stream ended';break
                with self.lock:self.frame=frame;self.seq+=1;self.stamp=time.monotonic()
        except Exception as exc:self.error=str(exc)
        finally:self.capture.release()

    def next(self, previous=0, timeout=5):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            with self.lock:
                if self.seq>previous:return self.seq,self.frame,self.stamp
            if self.error:raise RuntimeError(self.error)
            if self.stop.is_set():raise RuntimeError('Camera has been closed')
            time.sleep(.005)
        raise RuntimeError('No fresh camera frame within timeout')

    def close(self):
        self.stop.set();self.worker.join(timeout=4)


def camera_status(helper='/mnt/system/usr/bin/test_mmf', config='/mnt/data/sensor_cfg.ini'):
    text=Path(config).read_text(errors='replace') if Path(config).exists() else ''
    return {'capture_tool_exists':Path(helper).is_file(),'sensor_config_exists':bool(text),
            'gc4653_configured':'GCORE_GC4653' in text,
            'gc4653_driver_exists':Path('/mnt/system/usr/lib/libsns_gc4653.so').is_file(),
            'note':'Configuration check only; this does not prove the camera is connected.'}


class GC4653Camera:
    """Start the board's MMF RTSP helper, wait for real frames, and own cleanup."""
    def __init__(self, log_path='camera.log', startup_timeout=25):
        self.log_path=Path(log_path);self.startup_timeout=startup_timeout
        self.process=None;self.log=None;self.frames=None;self.lockfile=None;self.sequence=0

    def open(self):
        if self.frames is not None:return self
        try:
            status=camera_status()
            if not all(status[k] for k in ['capture_tool_exists','sensor_config_exists','gc4653_configured','gc4653_driver_exists']):
                raise RuntimeError(f'GC4653 environment is incomplete: {status}')
            import fcntl
            self.lockfile=open('/tmp/armor_gc4653.lock','a')
            try:fcntl.flock(self.lockfile,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:raise RuntimeError('Another armor camera process is already running')
            probe=socket.socket();probe.settimeout(.3)
            try:
                if probe.connect_ex(('127.0.0.1',8554))==0:
                    raise RuntimeError('Port 8554 is occupied. Stop the other server, or use armor_detector.py --source rtsp://127.0.0.1:8554/live')
            finally:probe.close()
            self.log=open(self.log_path,'ab',buffering=0)
            self.log.write(b'\n--- armor GC4653 camera start ---\n')
            env=dict(os.environ);env['LD_LIBRARY_PATH']='/mnt/system/usr/lib:/usr/bin/lib:'+env.get('LD_LIBRARY_PATH','')
            self.process=subprocess.Popen(['/mnt/system/usr/bin/test_mmf','10'],stdout=self.log,stderr=subprocess.STDOUT,env=env)
            deadline=time.monotonic()+self.startup_timeout
            while time.monotonic()<deadline:
                if self.process.poll() is not None:raise RuntimeError('MMF camera process exited during startup')
                probe=socket.socket();probe.settimeout(.2)
                try:ready=probe.connect_ex(('127.0.0.1',8554))==0
                finally:probe.close()
                if not ready:time.sleep(.2);continue
                candidate=None
                try:
                    candidate=LatestFrames('rtsp://127.0.0.1:8554/live')
                    # Require a decoded frame, not just an open RTSP TCP port.
                    candidate.next(0,timeout=max(.1,min(3,deadline-time.monotonic())))
                    self.frames=candidate;return self
                except Exception:
                    if candidate:candidate.close()
                    time.sleep(.3)
            raise RuntimeError('Timed out waiting for a real GC4653 frame')
        except BaseException as exc:
            self.close()
            if isinstance(exc,(KeyboardInterrupt,SystemExit)):raise
            raise RuntimeError(f'{exc}. Check sensor/cable with power disconnected and inspect {self.log_path}') from exc

    def read(self):
        if self.frames is None:raise RuntimeError('Call camera.open() first')
        self.sequence,frame,stamp=self.frames.next(self.sequence)
        return self.sequence,frame,stamp

    def close(self):
        # Only stop the process created by this object; never kill other camera users.
        if self.process and self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)
            try:self.process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=3)
        self.process=None
        if self.frames:self.frames.close();self.frames=None
        if self.log:self.log.close();self.log=None
        if self.lockfile:self.lockfile.close();self.lockfile=None

    def __enter__(self):return self.open()
    def __exit__(self,*unused):self.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true',help='Inspect configuration without starting the sensor')
    parser.add_argument('--snapshot',default='camera_test.jpg',help='Write a captured image here')
    parser.add_argument('--frames',type=int,default=1,help='Number of camera frames to read; final frame saved')
    parser.add_argument('--log',default='camera.log')
    args=parser.parse_args()
    if args.frames<1:parser.error('--frames must be positive')
    if args.check:
        print(json.dumps(camera_status(),ensure_ascii=False));return 0
    def interrupt(*unused):raise KeyboardInterrupt
    signal.signal(signal.SIGINT,interrupt);signal.signal(signal.SIGTERM,interrupt)
    try:
        with GC4653Camera(args.log) as camera:
            for _ in range(args.frames):seq,frame,stamp=camera.read()
            if not cv2.imwrite(args.snapshot,frame):raise RuntimeError('Could not save snapshot')
            print(json.dumps({'camera_opened':True,'width':frame.shape[1],'height':frame.shape[0],'snapshot':args.snapshot,'frame_sequence':seq}))
        return 0
    except KeyboardInterrupt:return 130
    except Exception as exc:
        print(f'ERROR: {exc}',file=sys.stderr);return 1


if __name__=='__main__':sys.exit(main())
