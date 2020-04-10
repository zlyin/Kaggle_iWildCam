#!/usr/bin/python3.6

## import packages
import os
import sys
sys.path.append("../DL2CV/Orca/io")
from hdf5datasetwriter import HDF5DatasetWriter
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.model_selection import train_test_split
from imutils import paths
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from shutil import copy
#from config import tiny_imagenet_config as cfg

"""
Prepare animal crops for iWildCam 2020
Generate train/val/test hdf5 files;
"""
TRAIN_IMAGES = "./data/animal_crops"
TEST_IMAGES = "./data/animal_crops_test"

OUTPUT = "./data/animal_crops_hdf5"
TRAIN_HDF5 = os.path.sep.join([OUTPUT, "animal_crops_train.hdf5"])
VAL_HDF5 = os.path.sep.join([OUTPUT, "animal_crops_val.hdf5"])
TEST_HDF5 = os.path.sep.join([OUTPUT, "animal_crops_test.hdf5"])

DATASET_MEAN = os.path.sep.join([OUTPUT, "animal_crops_train_mean.json"])
DATSET_CLASS_WEIGHT = os.path.sep.join([OUTPUT, "animal_crops_class_weights.json"])
LABEL_MAPPING = os.path.sep.join([OUTPUT, "encodedLabels_to_categoryID_mapping.json"])
STATS_HIST = os.path.sep.join([OUTPUT, "animal_crops_dim_stats.png"])
STATS_INFO = os.path.sep.join([OUTPUT, "animal_crops_dim_stats_description.csv"])

TARGET_H = 64
TARGET_W = 64


"""
- prepare all train images to prepare HDF5 file, label mapping, RGB means, and
  class_weights
"""
# grab trainPaths
trainPaths = list(paths.list_images(TRAIN_IMAGES))
trainLabels = [p.split(os.path.sep)[-2] for p in trainPaths]    # class names

trainLabels = np.array(trainLabels)
lb = LabelBinarizer()
trainLabels = lb.fit_transform(trainLabels)

# stratify sampling val data out of trainPaths
trainPaths, valPaths, trainLabels, valLabels = train_test_split(trainPaths, 
        trainLabels, test_size=0.2, stratify=trainLabels, random_state=42)

# serialize encoded label => category_id in annotation files
print("[INFO] form encoded_class => classNames...")
encoded_class = lb.classes_ # the order of set(classNames) to be encoded
encoded_label_mapping = dict(enumerate(encoded_class))

print("[INFO] fetch classNames => category_id...")
category_df = pd.read_csv("./data/iwildcam2020_train_annotations_categories.csv")
print(category_df.head())
print(category_df.tail())
category_id_mapping = category_df.groupby("name")["id"].apply(list).to_dict() # {'acinonyx jubatus': [122]}

print("[INFO] serizaling encoded_class to category_id mapping...")
mapping_dict = {
        "encodedLabel_to_className" : encoded_label_mapping,
        "className_to_categoryID" : category_id_mapping,
        }
with open(LABEL_MAPPING, "w") as json_file:
    json_file.write(json.dumps(mapping_dict))
json_file.close()

# assign class_weights to counter class imbalance!
print("[INFO] compute & serialize class weights of training data")
classTotals = trainLabels.sum(axis=0)
data_class_weights = classTotals.max() / classTotals
class_weights_dict = dict(enumerate(data_class_weights))
        
with open(DATSET_CLASS_WEIGHT, "w") as json_file:
    json_file.write(json.dumps(class_weights_dict))
json_file.close()
print("[INFO] (min, max) =", (data_class_weights.min(), data_class_weights.max()))


"""
- prepare test images; noting that the `labels` will be filenames;
"""
testPaths = list(paths.list_images(TEST_IMAGES))
testLabels = [path.split(os.path.sep)[-1] for path in testPaths]   # file name!


"""
- construct a list of different dataset
"""
datasets = [
        ("train", trainPaths, trainLabels, TRAIN_HDF5),
        ("val", valPaths, valLabels, VAL_HDF5),
        #("test", testPaths, testLabels, TEST_HDF5),
        ]

# stats dims of training images
fnames = []
H = []
W = []

# initialize RBG mean values
(R, G, B) = ([], [], [])

# loop over dataset & generate hdf5 file
for (dType, paths, labels, outputPath) in tqdm(datasets):
    # create HDF5 writer
    print("[INFO] building %s ..." % outputPath)
    if dType in ["train", "val"]:
        writer = HDF5DatasetWriter(outputPath, (len(paths), TARGET_H, TARGET_W, 3),
                bufSize=1000, labelDtype="int", labelOneHot=len(encoded_class))
    else:
        writer = HDF5DatasetWriter(outputPath, (len(paths), TARGET_H, TARGET_W, 3),
                bufSize=1000, labelDtype="string", labelOneHot=0)

    # progress bar
    for i, (path, label) in enumerate(tqdm(zip(paths, labels))):
        # load in image
        image = cv2.imread(path)
        image = cv2.resize(image, (TARGET_H, TARGET_W))
        
        # stats dim of animal crops
        if dType in ["train", "val"]:
            name = path.split(os.path.sep)[-1]
            h, w = image.shape[:2]
            fnames.append(name)
            H.append(h)
            W.append(w)

            # record RGB mean from training set
            (b, g, r) = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)

        # add image, label to HDF5 dataset
        writer.add([image], [label])
        pass
    
    # close HDF5 db when finish
    writer.close()


"""
- serialize RBG mean values to json file
"""
if R and G and B:
    print("[INFO] serializing training means ...")
    mean_dict = {"R" : np.mean(R), "G" : np.mean(G), "B" : np.mean(B)}
    with open(DATASET_MEAN, "w") as f:
        f.write(json.dumps(mean_dict))
    f.close()


"""
- stats dim of dataset
"""
#print("[INFO] animal crops dim:")
#stats = pd.DataFrame({"name" : fnames, "H" : H, "W" : W, "AR=W/H" : [-1] * len(H)})
#for index, row in stats.iterrows():
#    stats.loc[[index], ["AR=W/H"]] = row["W"] / float(row["H"])
#des_df = stats.describe()
#print(des_df)
#print("stats.shape =", stats.shape)
#des_df.to_csv(STATS_INFO, index=False)
#
#fig, ax = plt.subplots()
#stats.hist(bins=20, grid=True, ax=ax)
#fig.savefig(STATS_HIST)



print("Done!")
