#!/usr/bin/env python3
"""SG2002 armor detection. stdout: JSONL; diagnostics: stderr.

GC4653 uses the board MMF RTSP source, NOT /dev/video0.
Runtime API: https://github.com/sophgo/cviruntime/blob/master/include/cviruntime.h
"""
import argparse
import ctypes as C
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

import cv2
import numpy as np
from camera_capture import GC4653Camera, LatestFrames

CLASSES = ['G_Blue','G_Red','G_N','1_Blue','1_Red','1_N','2_Blue','2_Red','2_N',
           '3_Blue','3_Red','3_N','4_Blue','4_Red','4_N','5_Blue','5_Red','5_N',
           'O_Blue','O_Red','Bs_Blue','Bs_Red','Bb_Blue','Bb_Red']


class Shape(C.Structure):
    _fields_ = [('dim', C.c_int32 * 6), ('dim_size', C.c_size_t)]


class Runtime:
    """Keep one model loaded; copy input/output via official opaque tensor APIs."""
    def __init__(self, model, library='/usr/bin/lib/libcviruntime.so'):
        self.lib = C.CDLL(library, mode=C.RTLD_GLOBAL)
        l = self.lib
        specs = {
            'CVI_NN_RegisterModel': ([C.c_char_p, C.POINTER(C.c_void_p)], C.c_int),
            'CVI_NN_GetInputOutputTensors': ([C.c_void_p,C.POINTER(C.c_void_p),C.POINTER(C.c_int32),C.POINTER(C.c_void_p),C.POINTER(C.c_int32)], C.c_int),
            'CVI_NN_TensorPtr': ([C.c_void_p], C.c_void_p),
            'CVI_NN_TensorSize': ([C.c_void_p], C.c_size_t),
            'CVI_NN_TensorShape': ([C.c_void_p], Shape),
            'CVI_NN_Forward': ([C.c_void_p,C.c_void_p,C.c_int32,C.c_void_p,C.c_int32], C.c_int),
            'CVI_NN_CleanupModel': ([C.c_void_p], C.c_int),
        }
        for name,(args,ret) in specs.items():
            f=getattr(l,name);f.argtypes=args;f.restype=ret
        self.handle=C.c_void_p();self.inputs=C.c_void_p();self.outputs=C.c_void_p()
        ni=C.c_int32();no=C.c_int32()
        self.check(l.CVI_NN_RegisterModel(os.fsencode(model),C.byref(self.handle)),'load model')
        try:
            self.check(l.CVI_NN_GetInputOutputTensors(self.handle,C.byref(self.inputs),C.byref(ni),C.byref(self.outputs),C.byref(no)),'get tensors')
            if (ni.value,no.value)!=(1,1):raise RuntimeError('Expected one input and one output')
            self.input_shape=self.shape(self.inputs);self.output_shape=self.shape(self.outputs)
            if self.input_shape!=(1,3,640,640) or self.output_shape!=(1,36,8400,1):
                raise RuntimeError(f'Unexpected model shapes: {self.input_shape}, {self.output_shape}')
            if l.CVI_NN_TensorSize(self.inputs)!=1*3*640*640*4 or l.CVI_NN_TensorSize(self.outputs)!=1*36*8400*4:
                raise RuntimeError('Expected FP32 input/output byte sizes')
            self.input_ptr=l.CVI_NN_TensorPtr(self.inputs)
            self.output_ptr=l.CVI_NN_TensorPtr(self.outputs)
            if not self.input_ptr or not self.output_ptr:raise RuntimeError('Null tensor buffer')
        except Exception:
            self.close();raise

    @staticmethod
    def check(ret, action):
        if ret!=0:raise RuntimeError(f'{action} failed: {ret}')

    def shape(self, tensor):
        s=self.lib.CVI_NN_TensorShape(tensor)
        if not 1<=s.dim_size<=6:raise RuntimeError('Unsupported runtime shape ABI')
        return tuple(s.dim[:s.dim_size])

    def infer(self, tensor):
        if tensor.dtype!=np.float32 or tensor.shape!=self.input_shape or not tensor.flags.c_contiguous:
            raise ValueError('Input must be contiguous FP32 [1,3,640,640]')
        C.memmove(self.input_ptr,tensor.ctypes.data,tensor.nbytes)
        start=time.perf_counter()
        self.check(self.lib.CVI_NN_Forward(self.handle,self.inputs,1,self.outputs,1),'inference')
        elapsed=(time.perf_counter()-start)*1000
        result=np.ctypeslib.as_array(C.cast(self.output_ptr,C.POINTER(C.c_float)),shape=(36*8400,)).copy().reshape(self.output_shape)
        if not np.isfinite(result).all():raise RuntimeError('Non-finite model output')
        return result,elapsed

    def close(self):
        if self.handle.value:
            self.lib.CVI_NN_CleanupModel(self.handle);self.handle=C.c_void_p()


def preprocess(bgr):
    h,w=bgr.shape[:2];r=min(640/w,640/h)
    nw,nh=max(1,int(w*r)),max(1,int(h*r));left,top=(640-nw)//2,(640-nh)//2
    canvas=np.full((640,640,3),114,np.uint8)
    canvas[top:top+nh,left:left+nw]=cv2.resize(bgr,(nw,nh),interpolation=cv2.INTER_LINEAR)
    tensor=np.ascontiguousarray(canvas[:,:,::-1].transpose(2,0,1)[None].astype(np.float32)/255.0)
    return tensor,(w,h,r,left,top)


def postprocess(raw, meta, threshold=.25, iou_threshold=.45, color='all', center_method='bbox'):
    a=np.asarray(raw,dtype=np.float32).reshape(36,8400).T
    scores=a[:,4:28].max(axis=1);labels=a[:,4:28].argmax(axis=1)
    valid=np.isfinite(a).all(axis=1)&(scores>=threshold)&(a[:,2]>0)&(a[:,3]>0)
    if color!='all':valid &= np.array([CLASSES[i].endswith('_'+color) for i in labels])
    ids=np.flatnonzero(valid)
    boxes=np.concatenate((a[:,:2]-a[:,2:4]/2,a[:,:2]+a[:,2:4]/2),axis=1)
    areas=np.maximum(0,boxes[:,2:]-boxes[:,:2]).prod(axis=1)
    keep=[]
    for label in np.unique(labels[ids]):
        group=ids[labels[ids]==label];order=group[np.argsort(-scores[group],kind='stable')]
        while order.size:
            k=int(order[0]);keep.append(k);remaining=order[1:]
            if not remaining.size:break
            overlap=np.maximum(0,np.minimum(boxes[k,2:],boxes[remaining,2:])-np.maximum(boxes[k,:2],boxes[remaining,:2])).prod(axis=1)
            iou=overlap/(areas[k]+areas[remaining]-overlap+1e-9)
            order=remaining[iou<=iou_threshold]
    w,h,r,left,top=meta;detections=[]
    for k in sorted(keep,key=lambda i:-float(scores[i])):
        box=(boxes[k]-np.array([left,top,left,top]))/r
        if box[2]<0 or box[3]<0 or box[0]>=w or box[1]>=h:continue
        points=(a[k,28:].reshape(4,2)-[left,top])/r
        center=(box[:2]+box[2:])/2
        if center_method=='keypoints':center=points.mean(axis=0)
        if not (0<=center[0]<w and 0<=center[1]<h):continue
        clipped=box.copy();clipped[[0,2]]=np.clip(clipped[[0,2]],0,w-1);clipped[[1,3]]=np.clip(clipped[[1,3]],0,h-1)
        detections.append({'class_id':int(labels[k]),'class_name':CLASSES[labels[k]],'confidence':round(float(scores[k]),6),
            'center':[round(float(v),2) for v in center],
            'center_normalized':[round(float(center[0]/w),6),round(float(center[1]/h),6)],
            'offset_from_image_center':[round(float(center[0]-(w-1)/2),2),round(float(center[1]-(h-1)/2),2)],
            'bbox_xyxy':[round(float(v),2) for v in clipped],
            'keypoints':[[round(float(v),2) for v in p] for p in points]})
    return detections


def annotate(image, detections):
    out=image.copy()
    for d in detections:
        x1,y1,x2,y2=map(round,d['bbox_xyxy']);cx,cy=map(round,d['center'])
        cv2.rectangle(out,(x1,y1),(x2,y2),(0,255,0),1)
        cv2.drawMarker(out,(cx,cy),(0,0,255),cv2.MARKER_CROSS,12,1)
        cv2.putText(out,f"{d['class_name']} {d['confidence']:.2f} ({cx},{cy})",(max(0,x1),max(12,y1-4)),cv2.FONT_HERSHEY_SIMPLEX,.35,(0,255,0),1)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default='/root/first_model_sg2002_int8_mix.cvimodel')
    p.add_argument('--source',default='rtsp://127.0.0.1:8554/live')
    p.add_argument('--image',help='Test one still image instead of a camera')
    p.add_argument('--start-csi',action='store_true',help='Launch the board GC4653 MMF RTSP sample (camera must be attached)')
    p.add_argument('--camera-log',default='camera.log')
    p.add_argument('--conf',type=float,default=.25);p.add_argument('--iou',type=float,default=.45)
    p.add_argument('--color',choices=['all','Blue','Red','N'],default='all')
    p.add_argument('--center-method',choices=['bbox','keypoints'],default='bbox')
    p.add_argument('--max-frames',type=int,default=0,help='0 means unlimited')
    p.add_argument('--save-json',help='Optional JSONL file')
    p.add_argument('--save-image',help='Optional latest annotated JPEG/PNG (overwritten)')
    p.add_argument('--save-raw',help='Save final raw output NPZ for verification')
    args=p.parse_args()
    if not 0<=args.conf<=1 or not 0<=args.iou<=1:p.error('conf/iou must be within [0,1]')
    if args.image and args.start_csi:p.error('--image and --start-csi are mutually exclusive')
    # Keep native C runtime diagnostic output away from the JSON stdout stream.
    json_stdout=os.fdopen(os.dup(1),'w',buffering=1);os.dup2(2,1)
    cv2.setNumThreads(1)
    os.environ.setdefault('OPENCV_FFMPEG_CAPTURE_OPTIONS','rtsp_transport;tcp')
    running=threading.Event();running.set()
    def stop(*unused):
        running.clear();raise KeyboardInterrupt
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    runtime=None;frames=None;camera=None;json_file=None
    try:
        runtime=Runtime(args.model)
        if args.start_csi:
            camera=GC4653Camera(args.camera_log);camera.open()
        if args.save_json:json_file=open(args.save_json,'w',buffering=1)
        still=cv2.imread(args.image) if args.image else None
        if args.image and still is None:raise RuntimeError(f'Cannot read image: {args.image}')
        if not args.image and camera is None:frames=LatestFrames(args.source)
        count=0;seq=0
        while running.is_set():
            start=time.monotonic()
            if args.image:frame=still;captured=start
            elif camera is not None:seq,frame,captured=camera.read()
            else:seq,frame,captured=frames.next(seq)
            tensor,meta=preprocess(frame);raw,inference_ms=runtime.infer(tensor)
            detections=postprocess(raw,meta,args.conf,args.iou,args.color,args.center_method)
            result={'frame_id':count,'timestamp':time.time(),'image_size':[meta[0],meta[1]],'coordinate_system':'original_image_pixels_top_left_xy',
                'center_method':args.center_method,'found':bool(detections),'target':detections[0] if detections else None,
                'detections':detections,'inference_ms':round(inference_ms,2),'frame_age_ms':round((time.monotonic()-captured)*1000,2)}
            line=json.dumps(result,ensure_ascii=False,allow_nan=False);print(line,file=json_stdout,flush=True)
            if json_file:print(line,file=json_file,flush=True)
            if args.save_image and not cv2.imwrite(args.save_image,annotate(frame,detections)):raise RuntimeError('Failed to save annotated image')
            if args.save_raw:np.savez(args.save_raw,output0_Concat_f32=raw)
            count+=1
            if args.image or (args.max_frames and count>=args.max_frames):break
    except KeyboardInterrupt:pass
    except Exception as exc:
        print(f'ERROR: {exc}',file=sys.stderr);return 1
    finally:
        if frames:frames.close()
        if camera:camera.close()
        if runtime:runtime.close()
        if json_file:json_file.close()
        json_stdout.close()
    return 0


if __name__=='__main__':sys.exit(main())
