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
sys.path.append("../DL2CV/Orca/preprocessing")
from imagetoarraypreprocessor import ImageToArrayPreprocessor
from simplepreprocessor import SimplePreprocessor
from aspectawarepreprocessor import AspectAwarePreprocessor
from meanpreprocessor import MeanPreprocessor
sys.path.append("../DL2CV/Orca/datasets")
from simpledatasetloader import SimpleDatasetLoader
from utils.metrics import recall, precision, f1_score

from keras.models import load_model
from keras.utils import multi_gpu_model

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from imutils import paths
import random
import argparse
import json
from tqdm import tqdm


# images to be ignored => has been removed from sample_submission.csv
over_id_list = [
	'89362ed4-21bc-11ea-a13a-137349068a90', '86994b3e-21bc-11ea-a13a-137349068a90', '8985bb98-21bc-11ea-a13a-137349068a90', 
	'8e940310-21bc-11ea-a13a-137349068a90', '8d705d8a-21bc-11ea-a13a-137349068a90', '88b99aae-21bc-11ea-a13a-137349068a90', 
	'9044a3b8-21bc-11ea-a13a-137349068a90', '8b91394e-21bc-11ea-a13a-137349068a90', '920ee4c4-21bc-11ea-a13a-137349068a90', 
	'9955d012-21bc-11ea-a13a-137349068a90', '8b8e02a6-21bc-11ea-a13a-137349068a90', '98da656c-21bc-11ea-a13a-137349068a90', 
	'8e930668-21bc-11ea-a13a-137349068a90', '89e09b26-21bc-11ea-a13a-137349068a90', '88a28616-21bc-11ea-a13a-137349068a90', 
	'9522d4fe-21bc-11ea-a13a-137349068a90', '950ed288-21bc-11ea-a13a-137349068a90', '882a533a-21bc-11ea-a13a-137349068a90', 
	'98552f5a-21bc-11ea-a13a-137349068a90', '8fff9dc2-21bc-11ea-a13a-137349068a90', '8a804608-21bc-11ea-a13a-137349068a90', 
	'8cc46b6a-21bc-11ea-a13a-137349068a90', '96bacf06-21bc-11ea-a13a-137349068a90', '8ea6a768-21bc-11ea-a13a-137349068a90'
        ]


## arguments
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", required=True, help="path to the test data")
parser.add_argument("-c", "--checkpoints", required=True, help="path to a checkpoints folder")
parser.add_argument("-s", "--start_epoch", required=True, help="epoch number")
args = vars(parser.parse_args())
assert os.path.exists(args["dataset"])
assert os.path.exists(args["checkpoints"])


## Params
BATCH = 64 * 2
#METRICS = ["accuracy", f1_score]
MODEL = os.path.sep.join([args["checkpoints"], "epoch_%s.hdf5" % args["start_epoch"]])
DATASET_MEAN = os.path.sep.join([args["checkpoints"], "trainX_mean.json"])
ID_MAPPING = os.path.sep.join([args["checkpoints"], "encodedLabels_to_categoryID_mapping.json"])
CSV = os.path.sep.join([args["checkpoints"], "submission_at_%s.csv" % args["start_epoch"]])

# load encoded_class to category_id mapping...
mapping_dict = json.loads(open(ID_MAPPING, "r").read())
encodedLabel_to_className = mapping_dict["encodedLabel_to_className"]
className_to_categoryID = mapping_dict["className_to_categoryID"]

# load submission.csv & reset 0
submission = pd.read_csv("./sample_submission.csv")
submission["Category"] = [0] * submission.shape[0]
print("[INFO] sample_sumission\n")
print(submission.head())
print("[INFO] expect to predict =", submission.shape)

## augmentation
aap = AspectAwarePreprocessor(64, 64)
iap = ImageToArrayPreprocessor()
means = json.loads(open(DATASET_MEAN).read())
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
sdl = SimpleDatasetLoader(preprocessors=[aap, mp, iap], mode="test")

# load in images
print("[INFO] loading test images....")
imagePaths = list(paths.list_images(args["dataset"]))
print("[INFO] fetched %d images to test" % len(imagePaths))

data, names = sdl.load(imagePaths, verbose=1e4)
testX = data.astype("float") / 255.0
imageIds = [name.split(".")[0] for name in names]


## load in models & predict
with tf.device("/cpu:0"):
    model = load_model(MODEL, custom_objects={"f1_score" : f1_score}) 

# create distribute strategy for TF2.0
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    parallel_model = multi_gpu_model(model, gpus=2)
    #parallel_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=METRICS)
 
predictions = parallel_model.predict(testX, batch_size=BATCH)
pred_labels = predictions.argmax(axis=1)

## map pred_labels from encodedLabel -> categoryID via
# encodedLabel_to_className -> className_to_categoryID
print("[INFO] filling in submission file....")
pred_categoryIDs = []
for l in pred_labels:
    pred_className = encodedLabel_to_className[str(l)]  # cast into str!
    pred_categoryIDs.append(className_to_categoryID[pred_className][0])

# feed into submission cvs & save
for imgID, pred_catID in tqdm(zip(imageIds, pred_categoryIDs)):
    if imgID in submission["Id"].values:    # need to check series.values!
        submission.loc[submission["Id"] == imgID, "Category"] = pred_catID
    pass
submission.to_csv(CSV, index=False)
print("[INFO] prediction result")
print(submission.head())

print("[INFO] Done!")


