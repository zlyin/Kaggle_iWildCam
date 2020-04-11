## import packages
import cv2


"""
- CLAHE = Contrast Limited Adaptive Histogram Equalization
"""
class CLAHE:
    def __init__(self, clipLimit=2.0, tileGridSize=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)

    def preprocess(self, image):
        img = image.copy()
        # convert to LAB domain
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab_planes = cv2.split(lab)
        lab_planes[0] = self.clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return bgr


"""
- SimpleWhiteBalance
"""
class SimpleWhiteBalance:
    def __init__(self, p=0.4):
        self.p = p
    
    def preprocess(self, image):
        img = image.copy()
        wb = cv2.xphoto.createSimpleWB()
        wb.setP(self.p)
        img_wb = wb.balanceWhite(img)
        return img_wb

