#!/usr/local/bin/python3

## import packages
import cv2
import numpy as np
import glob
import os
import sys
sys.path.append("../utils")
from preprocessings import *
from imutils import paths
from tqdm import tqdm


#OUTPUT_FOLDER = "./RandomSatuation"
#OUTPUT_FOLDER = "./RandomHue"
#OUTPUT_FOLDER = "./RandomLightingNoise"
#OUTPUT_FOLDER = "./Sharpen"
OUTPUT_FOLDER = "./GreenGray"
#OUTPUT_FOLDER = "./RedGray"
#OUTPUT_FOLDER = "./BlueGray"
#OUTPUT_FOLDER = "./Reverse"
#OUTPUT_FOLDER = "./Log"
#OUTPUT_FOLDER = "./Gamma"
OUTPUT_FOLDER = "./Deblur"


if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

# fetch all image paths
imagePaths = list(paths.list_images("./test_images/blurred"))
N = len(imagePaths)
print("[INFO] test on %d images" % N)

if __name__ == "__main__":
    for i, path in enumerate(tqdm(imagePaths)):
        image = cv2.imread(path)
        h, w, ch = image.shape
        
        # random satuation
        #img_sat = RandomSaturation().process(image)
        #final = img_sat

        # Random hue
        #img_hue = RandomHue().process(image)
        #final = img_hue

        # Random lighting noise
        #img_light_noise = RandomLightingNoise().process(image)
        #final = img_light_noise

        # Sharpen images
        #img_sharp = Sharpen().process(image)
        #final = img_sharp

        # extract green to gray
        #img_single_gray = SingleChannelToGray(chanDim=1).process(image)
        #final = img_single_gray

        # reverse image
        #img_reverse = Reverse().process(image)
        #final = img_reverse

        # Log transform
        #img_log = LogTransform(c=35).process(image)
        #final = img_log
        
        # gamma        
        #img_gamma = Gamma().process(image)
        #final = img_gamma

        # Deblur
        img_deblur = Deblur().process(image)
        final = img_deblur
    
        # save
        name = path.split(os.path.sep)[-1].replace(".jpg", "_(%d, %d).jpg" % (h, w))
        savepath = os.path.sep.join([OUTPUT_FOLDER, name])
        cv2.imwrite(savepath, final)

print("[INFO] done!")



