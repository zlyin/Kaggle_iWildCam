#!/usr/bin/python3.6

## import packages
import os
import sys
import cv2
import pandas as pd
import glob
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from tqdm import tqdm
import numpy as np
import shutil

OUTPUT = "./data/debug"
DATASET = "./data/iwildcam-2020-fgvc7/train"
assert DATASET and OUTPUT
MODE = "train"

# load in megadetector result
detector_results_images = pd.read_csv("./data/iwildcam2020_megadetector_results_images.csv")
print("[INFO] head of detector results...\n", detector_results_images.head())
print("cols = ", detector_results_images.columns)

# load in train_annotations &
if MODE == "train":
    trainAnno_annotations = pd.read_csv("./data/iwildcam2020_train_annotations_annotations.csv")
    trainAnno_categories = pd.read_csv("./data/iwildcam2020_train_annotations_categories.csv")
    trainAnno_images = pd.read_csv("./data/iwildcam2020_train_annotations_images.csv")

    print("[INFO] head of train annotations...\n", trainAnno_annotations.head())
    print("[INFO] head of train images...\n", trainAnno_images.head())
    print("[INFO] head of train categories...\n", trainAnno_categories.head())
    print()

    # sort values by train categories
    cat_df = trainAnno_categories[trainAnno_categories["count"] == 25]
    print("count == 20 categories are\n", cat_df.head())

    #sys.exit()
    
    # pick this guy
    #target = "philander opossum"
    #target_id = 67
    target = "hemigalus derbyanus"
    target_id = 142
    target_annotations = trainAnno_annotations[trainAnno_annotations["category_id"]==target_id]
    target_images = target_annotations["image_id"].values

    # try sort
    sorted(target_images)
    
    # copy all target images to a folder
    for img_id in tqdm(target_images):
        # create a folder
        folder = os.path.sep.join([OUTPUT, target])
        if not os.path.exists(folder):
            os.makedirs(folder)

        imgsrc = os.path.sep.join([DATASET, img_id + ".jpg"])
        image = cv2.imread(imgsrc)
        cv2.imshow("img", image)
        cv2.waitKey(0)

        imgdst = os.path.sep.join([folder, img_id + ".jpg"])
        shutil.copy(imgsrc, imgdst)
        pass
        

