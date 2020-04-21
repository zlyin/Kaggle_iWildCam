#!/usr/local/bin/python3.7

## import package
import os
import sys
import numpy as np
import argparse
from imutils import paths
from tqdm import tqdm

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


## config paths
DIRECTORY = "./data/debug"
AUG_THRES = 20

## initiate data agumentation generator
aug = ImageDataGenerator(
        rotation_range=10, 
        #width_shift_range=0.1, 
        #height_shift_range=0.1, 
        shear_range=0.1, 
        zoom_range=0.1,
        channel_shift_range=10.0,
        horizontal_flip=True, 
        fill_mode="nearest")


## loop over class folders and augment on the ones < AUG_THRES
for folder in tqdm(os.listdir(DIRECTORY)):
    imagePaths = list(paths.list_images(os.path.sep.join([DIRECTORY, folder])))
    num = len(imagePaths)
    
    # augment if num < AUG_THRES
    if num < AUG_THRES:
        print("[INFO] augmenting " + folder + "...")

        for path in tqdm(imagePaths):
            image = load_img(path)
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)   # has dim=4!

            # create a data generator with ImageDataGenerator.flow() 
            i = 0
            ImageGen = aug.flow(
                    image,
                    batch_size=4, 
                    save_to_dir=os.path.sep.join([DIRECTORY, folder]),
                    save_prefix="aug", 
                    save_format="jpg")

            for i in range(int(AUG_THRES / num)):
                ImageGen.next()
        pass
    pass

print("[INFO] Done!")



           
        







