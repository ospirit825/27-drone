import unittest
import numpy as np
from armor_detector import preprocess,postprocess

class GeometryTests(unittest.TestCase):
    def test_letterbox(self):
        a,meta=preprocess(np.zeros((100,200,3),np.uint8))
        self.assertEqual(meta,(200,100,3.2,0,160))
        self.assertEqual(a.shape,(1,3,640,640))
        self.assertAlmostEqual(float(a[0,0,0,0]),114/255,places=6)

    def test_nms_coordinate_restore_and_empty(self):
        a=np.zeros((1,36,8400,1),np.float32)
        self.assertEqual(postprocess(a,(200,100,3.2,0,160)),[])
        for k,score in [(0,.9),(1,.8)]:
            a[0,:4,k,0]=[320,320,64,32];a[0,7,k,0]=score
        out=postprocess(a,(200,100,3.2,0,160))
        self.assertEqual(len(out),1)
        self.assertEqual(out[0]['center'],[100,50])
        self.assertEqual(out[0]['bbox_xyxy'],[90,45,110,55])
        self.assertEqual(postprocess(a,(200,100,3.2,0,160),color='Red'),[])

if __name__=='__main__':unittest.main()
