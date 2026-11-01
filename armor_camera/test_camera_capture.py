import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock,patch
import camera_capture as camera

class CameraTests(unittest.TestCase):
    def test_failed_capture_releases_device(self):
        capture=Mock();capture.isOpened.return_value=False
        with patch.object(camera.cv2,'VideoCapture',return_value=capture):
            with self.assertRaises(RuntimeError):camera.LatestFrames('0')
        capture.release.assert_called_once()

    def test_close_only_owned_resources_and_is_idempotent(self):
        c=camera.GC4653Camera();process=Mock();process.poll.return_value=None
        frames=Mock();log=Mock();lock=Mock()
        c.process=process;c.frames=frames;c.log=log;c.lockfile=lock
        c.close();c.close()
        process.send_signal.assert_called_once_with(camera.signal.SIGINT)
        process.wait.assert_called_once();frames.close.assert_called_once()
        log.close.assert_called_once();lock.close.assert_called_once()

    def test_incomplete_configuration_does_not_launch_sensor(self):
        with patch.object(camera,'camera_status',return_value={'capture_tool_exists':False,'sensor_config_exists':False,'gc4653_configured':False,'gc4653_driver_exists':False}),patch.object(camera.subprocess,'Popen') as launch:
            c=camera.GC4653Camera()
            with self.assertRaises(RuntimeError):c.open()
            launch.assert_not_called();self.assertIsNone(c.process)

if __name__=='__main__':unittest.main()
