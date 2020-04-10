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



## output folder
OUTPUT = "./data/animal_crops_test"
DATASET = "./data/iwildcam-2020-fgvc7/test"
assert DATASET and OUTPUT
MODE = "test"  # "test"


"""
extract objects from train images
"""
def extract_objects_train(img_path, show=False):
    objects = []
    confidences = []
    categories = []
    
    # split out image_id
    img_id = img_path.split('/')[-1].split('.')[0] 
    img = cv2.imread(img_path)

    if not (detector_results_images[detector_results_images.id == img_id].detections.values):
        return None

    # convert from str dict
    detections = detector_results_images[detector_results_images.id == img_id].detections.values[0]
    detections = eval(detections)

    annotation = trainAnno_annotations[trainAnno_annotations.image_id == img_id]
    print(annotation.head())

    cat_id = annotation.category_id
    cat_name = trainAnno_categories[trainAnno_categories.id == int(cat_id)].name.values[0]
    
    # loop over bboxes dict
    for idx, detection in enumerate(detections):
        # save confidence
        confidences.append(detection["conf"])

        # distinguish human or animal
        if detection['category'] == "1":
            categories.append(cat_name)
        else:
            categories.append('human')

        x_rel, y_rel, w_rel, h_rel = detection['bbox']    
        img_height, img_width, _ = img.shape
        x = float(x_rel * img_width)
        y = float(y_rel * img_height)
        w = float(w_rel * img_width)
        h = float(h_rel * img_height)

        obj = img[int(y):int(y+h), int(x):int(x+w), :]
        objects.append(obj)
        
        # display or not
        if show:
            cv2.imshow("crop", obj)
            cv2.waitKey(0)
    
    return (objects, categories, confidences)


"""
extract objects from test images
"""
def extract_objects_test(img_path, show=False):
    objects = []
    confidences = []
    categories = []
    
    # split out image_id
    img_id = img_path.split('/')[-1].split('.')[0] 
    img = cv2.imread(img_path)

    if not (detector_results_images[detector_results_images.id == img_id].detections.values):
        return None

    # convert from str dict
    detections = detector_results_images[detector_results_images.id == img_id].detections.values[0]
    detections = eval(detections)
    if not detections:
        return None

    # loop over bboxes dict
    for idx, detection in enumerate(detections):
        # save confidence
        confidences.append(detection["conf"])

        # distinguish human or animal
        if detection['category'] == "1":
            categories.append("animal")
        else:
            categories.append('human')

        x_rel, y_rel, w_rel, h_rel = detection['bbox']    
        img_height, img_width, _ = img.shape
        x = float(x_rel * img_width)
        y = float(y_rel * img_height)
        w = float(w_rel * img_width)
        h = float(h_rel * img_height)

        obj = img[int(y):int(y+h), int(x):int(x+w), :]
        objects.append(obj)
        
        # display or not
        if show:
            cv2.imshow("crop", obj)
            cv2.waitKey(0)
    
    return (objects, categories, confidences)




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

if MODE == "test":
    test_categories = pd.read_csv("./data/iwildcam2020_train_annotations_categories_test.csv")
    test_images = pd.read_csv("./data/iwildcam2020_train_annotations_images_test.csv")

    print("[INFO] head of test images...\n", test_images.head())
    print("[INFO] head of test categories...\n", test_categories.head())
    print() 


## list all raw images to be croped
imagePaths = list(glob.glob(os.path.sep.join([DATASET, "*.jpg"])))
print("[INFO] crop animals from %d images.." % len(imagePaths))

for path in tqdm(imagePaths):
    image_id = path.split("/")[-1].split(".")[0]

    try:
        if MODE == "train":
            items = extract_objects_train(path, show=False) 
        if MODE == "test":
            items = extract_objects_test(path, show=False) 
        if items is None: 
            continue

        # store images to different folder
        animals, categories, confidences = np.array(items[0]), np.array(items[1]), np.array(items[2]) 
        
        # only save confidence score > threshold value=0.9
        idxs = confidences > 0.9
        animals = animals[idxs]
        categories = categories[idxs]
        confidences = confidences[idxs]

        # double check if there are 2 types of animals in ONE image
        if len(set(categories)) > 1:
            print("got %d types of objects in image!" % len(set(categories)))

        cat_counts = {}
        # loop over animals & save
        for i in range(len(animals)):
            cat_name = categories[i]
            if MODE == "train":
                class_folder = os.path.sep.join([OUTPUT, cat_name])

                if not os.path.exists(class_folder):
                    os.makedirs(class_folder)

                # count current jth of a specific category
                j = cat_counts.get(cat_name, 0) + 1
                image_name = image_id + "_" + str(j) + ".jpg"
                cv2.imwrite(os.path.sep.join([class_folder, image_name]), animals[i])

            elif MODE == "test":
                class_folder = OUTPUT
                image_name = image_id + ".jpg"
                cv2.imwrite(os.path.sep.join([class_folder, image_name]), animals[i])
            pass
    except:
        continue

print("[INFO] Done!")
        
            


    

## debug
#if __name__ == "__main__":
#    info = extract_objects("./data/iwildcam-2020-fgvc7/train/8ed20ee4-21bc-11ea-a13a-137349068a90.jpg",
#            show=True)



