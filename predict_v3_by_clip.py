#!/usr/bin/python3.6

## import packages
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import tensorflow as tf
tfconfig = tf.compat.v1.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=tfconfig)

import sys
sys.path.append("../DL2CV/Orca/nn/conv")
from resnet import ResNet50
sys.path.append("../DL2CV/Orca/preprocessing")
from imagetoarraypreprocessor import ImageToArrayPreprocessor
from simplepreprocessor import SimplePreprocessor
from aspectawarepreprocessor import AspectAwarePreprocessor
from meanpreprocessor import MeanPreprocessor
sys.path.append("../DL2CV/Orca/io")
from hdf5datasetgenerator import HDF5DatasetGenerator
sys.path.append("./utils")
from preprocessings import *
from metrics import f1_score

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import classification_report

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from imutils import paths
import argparse
import json
import cv2
from tqdm import tqdm


## arguments
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mode", required=True, help="evaluate or predict")
parser.add_argument("-c", "--checkpoints", required=True, help="path to a checkpoints folder")
parser.add_argument("-s", "--start_epoch", required=True, help="epoch number")
args = vars(parser.parse_args())
assert os.path.exists(args["checkpoints"])
assert args["mode"] in ["evaluate", "predict"]


## Params
BATCH = 64
MODE = args["mode"]
NUM_CLASSES = 209   # change according to different crop sets

MODEL = os.path.sep.join([args["checkpoints"], "epoch_%s_weights.h5" % args["start_epoch"]])
CSV = os.path.sep.join([args["checkpoints"], "submission_at_%s_by_clip.csv" % args["start_epoch"]])

INPUT = "./data/animal_crops_224x224/animal_crops_224x224_hdf5"
VAL_HDF5 = os.path.sep.join([INPUT, "animal_crops_val.hdf5"])
DATASET_MEAN = os.path.sep.join([INPUT, "animal_crops_train_mean.json"])
LABEL_MAPPING = os.path.sep.join([INPUT, "encodedLabels_to_categoryID_mapping.json"])
DATASET_CLASS_WEIGHT = os.path.sep.join([INPUT, "animal_crops_class_weights_frequency.json"])
# Exp9 & 11
#DATASET_CLASS_WEIGHT = os.path.sep.join([INPUT, "animal_crops_class_weights_effective_num.json"])

# load encoded_class to category_id mapping...
mapping_dict = json.loads(open(LABEL_MAPPING, "r").read())
encodedLabel_to_className = mapping_dict["encodedLabel_to_className"]
className_to_categoryID = mapping_dict["className_to_categoryID"]

# class weights
class_weights_dict = json.loads(open(DATASET_CLASS_WEIGHT, "r").read())
data_class_weights = list(class_weights_dict.values())
print("[INFO] class weights (min, max, mean) =", min(data_class_weights), ",", 
        max(data_class_weights), ",", sum(data_class_weights) / len(data_class_weights))

# augmentation
means = json.loads(open(DATASET_MEAN, "r").read())
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
sp = SimplePreprocessor(224, 224)
iap = ImageToArrayPreprocessor()
clahe = CLAHEPreprocessor()
swb = SimpleWhiteBalancePreprocessor()
sn = SimpleNormalize()
shp = Sharpen()

# Exp8
GenPreprocessors = [sp, mp, iap]

# Exp10
#GenPreprocessors = [sp, clahe, swb, mp, iap]


"""
- predict on images under same seq_i => location => clip
- Args:
    - seqid: seq_id to fitler out images
    - model: model used to predict
- Returns:
    - final predicted categoryIDs
    - corresponding image names
"""
def predict_by_clip(seq_id, loc, clip_index, model, mode="predict"):

    # load in all crops under same clip
    clipPaths = list(paths.list_images(os.path.sep.join(
        [crops_folder, seq_id, str(loc), str(clip_index)])))
    imageIds = [path.split(os.path.sep)[-1].split(".")[0] for path in clipPaths]

    # check if there are images to predict
    if len(clipPaths) == 0:
        return [], []

    # extract images of the clip
    images_of_clip = []
    for path in clipPaths:
        image = cv2.imread(path)             
        for processor in GenPreprocessors:
            image = processor.preprocess(image)
        # store & label 
        images_of_clip.append(image)
        pass

    N = len(images_of_clip)
    images_of_clip = np.array(images_of_clip)
    clip_predictions = model.predict(images_of_clip, batch_size=N)
    clip_pred_labels = clip_predictions.argmax(axis=1)

    if mode == "evaluate":
        return imageIds, clip_pred_labels

    if mode == "predict":
        # map pred_labels from encodedLabel -> categoryID via
        clip_pred_categoryIDs = []
        # encodedLabel_to_className -> className_to_categoryID
        for l in clip_pred_labels:
            clip_pred_className = encodedLabel_to_className[str(l)]  # cast into str!
            # fixing manually merged "background" => "empty"
            clip_pred_className = "empty" if clip_pred_className == "background" else clip_pred_className
            clip_pred_categoryIDs.append(className_to_categoryID[clip_pred_className][0])

        return imageIds, clip_pred_categoryIDs


"""
- load in models & predict
"""
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    # reinitiate model as train_parallel_v3.py
    model = ResNet50(224, 224, 3, NUM_CLASSES, reg=5e-4, bnEps=2e-5, bnMom=0.9)

    print("[INFO] loading weights from %s ..." % MODEL)
    model.load_weights(MODEL, by_name=True)

 
    if MODE == "evaluate":
        print("[INFO] evaluating on validation set...")
        valGen = HDF5DatasetGenerator(VAL_HDF5, BATCH,
                preprocessors=GenPreprocessors, 
                binarize=False, classes=NUM_CLASSES)

        predictions = model.predict(
                valGen.generator(),
                steps=valGen.numImages // BATCH + 1,    # ensure all samples are included
                max_queue_size=BATCH,
                verbose=1,
                )
        pred_labels = predictions.argmax(axis=1)

        print(classification_report(np.array(valGen.db["labels"]).argmax(axis=1), pred_labels,
            #target_names=encodedLabel_to_className.values()),
            ))


        """
        - evaluate by clips
        """
#        val_crops_info = pd.read_csv(
#                "./data/animal_crops_224x224/animal_crops_val_224x224/sorted_val_crops_directory.csv")
#        valAnno_annotations = pd.read_csv(
#                "./data/animal_crops_224x224/animal_crops_val_224x224/valAnno_annotations_categories.csv")
#
#        # reverse
#        className_to_encodedLabel = {n : l for l, n in encodedLabel_to_className.items()}
#
#
#        ## Need to modify!!
#        crops_folder = "./data/animal_crops_224x224/animal_crops_val_224x224"
#        print("[INFO] evaluating on crops from=", crops_folder)
#        
#        print("[INFO] evaluating by each clip...")
#        GT_Labels = []
#        Pred_Labels = []
#
#        unique_seq_id = val_crops_info["seq_id"].unique()
#        for seq_id in tqdm(unique_seq_id):
#            loc_df = val_crops_info[val_crops_info["seq_id"] == seq_id]
#            locations = loc_df["location"].unique()
#
#            for loc in locations:
#                clip_df = loc_df[loc_df["location"] == loc]
#                clips = clip_df["clip_index"].unique()
#            
#                for clip_index in clips:
#                    # noting that pred_labels are not mapping batck to CategoryID!
#                    imageIds, pred_labels = predict_by_clip(seq_id, loc, clip_index, model)
#                    if not imageIds:
#                        continue
#                    
#                    # average based on crops of the clip
#                    unique_labels, label_counts = np.unique(np.array(pred_labels), return_counts=True)
#                    most_popular_label = unique_labels[label_counts.argmax()]
#
#                    # update the clip pred_labels
#                    pred_labels = [most_popular_label] * len(pred_labels)
#                    Pred_Labels.extend(pred_labels)
#
#                    # retrieve gt labels by image file names
#                    for imgid in imageIds:
#                        # valAnno_annotations has "empty" ==> convert to "background" 
#                        className = valAnno_annotations[valAnno_annotations["image_id"] == imgid]["name"].values[0]
#                        gt_catid = valAnno_annotations[valAnno_annotations["image_id"] ==
#                                imgid]["category_id"].values[0]
#
#                        get_catid = 0 if className not in className_to_encodedLabel.keys() else gt_catid
#                        GT_Labels.append(gt_catid)
#
#                        #className = "background" if className not in className_to_encodedLabel.keys() else className
#                        #gt_label = className_to_encodedLabel[className]
#                        #GT_Labels.append(int(gt_label))
#                pass
#            pass
#        assert len(GT_Labels) == len(Pred_Labels)
#        print(classification_report(GT_Labels, Pred_Labels))

    elif MODE == "predict":
        print("[INFO] predicting on test set...")

        # load submission.csv & reset 0
        submission = pd.read_csv("./sample_submission.csv")
        submission["Category"] = [0] * submission.shape[0]
        print("[INFO] sample_sumission expects to predict =", submission.shape)

        # read in dataframes
        testAnno_categories = pd.read_csv("./data/iwildcam2020_train_annotations_categories_test.csv")
        testAnno_images = pd.read_csv("./data/iwildcam2020_train_annotations_images_test.csv")
        print("\n[INFO] head of test images...\n", testAnno_images.head())
        print("cols = ", testAnno_images.columns, "shape = ", testAnno_images.shape)
        print("[INFO] head of test categories...\n", testAnno_categories.head())
        print("cols = ", testAnno_categories.columns)

        test_images_info = pd.read_csv("./data/iwildcam-2020-fgvc7/sorted_test_images_directory.csv")
        test_crops_info = pd.read_csv("./data/animal_crops_224x224/animal_crops_test_224x224/sorted_test_crops_directory.csv")

        # loop over each seq_id
        unique_seq_id = testAnno_images["seq_id"].unique()        
        
        ## Need to modify!!
        crops_folder = "./data/animal_crops_224x224/animal_crops_test_224x224"
        print("[INFO] predicting the crops from=", crops_folder)
        
        ## V2 = average per clip
        print("[INFO] predicting by each clip...")
        for seq_id in tqdm(unique_seq_id):
            loc_df = test_crops_info[test_crops_info["seq_id"] == seq_id]
            locations = loc_df["location"].unique()

            for loc in locations:
                clip_df = loc_df[loc_df["location"] == loc]
                clips = clip_df["clip_index"].unique()
            
                for clip_index in clips:
                    imageIds, pred_categoryIDs = predict_by_clip(seq_id, loc, clip_index, model,
                            mode=MODE)
                    if not imageIds or np.any(np.array(pred_categoryIDs) != 0) == False:
                        continue
                    
                    # average based on crops of the clip
                    unique_ids, id_counts = np.unique(np.array(pred_categoryIDs), return_counts=True)
                    most_popular_id = unique_ids[id_counts.argmax()]
                    
                    # pad to all test images
                    test_images_to_pad = test_images_info[
                            (test_images_info["seq_id"] == seq_id) &
                            (test_images_info["location"] == loc) &
                            (test_images_info["clip_index"] == clip_index)]["file_name"].values
                    test_images_to_pad = [name.split(".")[0] for name in test_images_to_pad]
                    for imgID in test_images_to_pad:
                        submission.loc[submission["Id"] == imgID, "Category"] = most_popular_id
                pass
            pass

        submission.to_csv(CSV, index=False)
        print("[INFO] prediction result")
        print(submission.head())
        pass

print("[INFO] Done!")



