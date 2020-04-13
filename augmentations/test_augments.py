#!/usr/bin/python3.6

## import packages
import cv2
import numpy as np
import glob
import os
from os import listdir
from os.path import isfile, join
from shutil import move
from imutils import paths
from imagepreprocessing import CLAHE, SimpleWhiteBalance


OUTPUT_FOLDER = "./CLAHE_WB_COMBO"

imagePaths = list(paths.list_images("./test_images"))[:1]
N = len(imagePaths)
print("[INFO] test on %d images" % N)



if __name__ == "__main__":

    for i, path in enumerate(imagePaths):
        img = cv2.imread(path)
        h, w, ch = img.shape

#        if h < w:
#            img=cv2.resize(img,(299, int(299*h/w)))
#        elif h>w:
#            img=cv2.resize(img,(int(299*w/h),299 ))
#        else:
#            img=cv2.resize(img,(299,299 ))
        
        img_clahe = CLAHE(img)
        img_swb = SimpleWhiteBalance(img_clahe) 
        final = img_swb

        # save
        name = path.split(os.path.sep)[-1].replace(".jpg", "_(%d, %d).jpg" % (h, w))
        savepath = os.path.sep.join([OUTPUT_FOLDER, name])
        cv2.imwrite(savepath, final)

print("[INFO] done!")


