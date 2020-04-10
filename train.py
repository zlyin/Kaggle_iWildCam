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
sys.path.append("../DL2CV/Orca/callbacks")
from trainingmonitor import TrainingMonitor
from epochcheckpoint import EpochCheckpoint
sys.path.append("../DL2CV/Orca/preprocessing")
from imagetoarraypreprocessor import ImageToArrayPreprocessor
from simplepreprocessor import SimplePreprocessor
from aspectawarepreprocessor import AspectAwarePreprocessor
from meanpreprocessor import MeanPreprocessor
sys.path.append("../DL2CV/Orca/datasets")
from simpledatasetloader import SimpleDatasetLoader
from utils.metrics import recall, precision, f1_score

from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD, Adam
from keras.models import load_model
from keras.utils import multi_gpu_model
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from imutils import paths
import random
import argparse
import json
import pandas as pd


"""
Train on iWildCam2020 with ResNet
"""
def poly_decay(epoch):
    """
    polynomial learning rate decay: alpha = alpha0 * (1 - epoch/num_epochs) ** p
    - alpha0 = initial learning rate
    - p = exp index, can be 1, 2, 3 ... etc
    - epoch = current epoch number of training process
    """
    maxEpochs = EPOCHS
    baseLR = INIT_LR
    power = 1.0

    # compute
    lr = baseLR * (1 - (epoch / float(maxEpochs))) ** power
    return lr


## Build arguments parser
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", required=True, help="path to the training data")
parser.add_argument("-c", "--checkpoints", required=True, help="path to the checkpoints directory")
parser.add_argument("-m", "--model", type=str, help="path to a specific loaded model")
parser.add_argument("-s", "--start_epoch", type=int, default=0, help="epoch to start training from")
args = vars(parser.parse_args())
assert os.path.exists(args["checkpoints"])


## Hyperparams
EPOCHS = 30
INIT_LR = 1e-1
BATCH = 32
METRICS = ["accuracy", f1_score]


## augmentation
aap = AspectAwarePreprocessor(64, 64)
iap = ImageToArrayPreprocessor()
# means = json.loads(open(cfg.DATASET_MEAN).read())
#mp = MeanPreprocessor(means["R"], means["G"], means["B"])
sdl = SimpleDatasetLoader(preprocessors=[aap, iap])

aug = ImageDataGenerator(
        horizontal_flip=True,
        )

## prepare all train images
print("[INFO] fetching category ids ....")
category_df = pd.read_csv("./data/iwildcam2020_train_annotations_categories.csv")
print(category_df.head())
print(category_df.tail())
category_id_mapping = category_df.groupby("name")["id"].apply(list).to_dict() # {'acinonyx jubatus': [122]}

print("[INFO] loading images....")
imagePaths = list(paths.list_images(args["dataset"]))
#imagePaths = imagePaths[:1000]
classNames = [path.split(os.path.sep)[-2] for path in imagePaths]
classes = len(set(classNames))
print("[INFO] fetched %d classes and %d images in total" % (classes, len(imagePaths)))

## prepare images & labels
print("[INFO] loading data...")
data, labels = sdl.load(imagePaths, verbose=1e4)
data = data.astype("float") / 255.0

labels = np.array(labels)
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

# serialize encoded label => category_id in annotation files
print("[INFO] serizaling encoded_class to category_id mapping...")
encoded_class = lb.classes_ # the order of set(classNames) to be encoded
encoded_label_mapping = {}
for i, name in enumerate(encoded_class):
    encoded_label_mapping[str(i)] = name

mapping_dict = {
        "encodedLabel_to_className" : encoded_label_mapping,
        "className_to_categoryID" : category_id_mapping,
        }
with open(os.path.sep.join([args["checkpoints"],
    "encodedLabels_to_categoryID_mapping.json"]), "w") as json_file:
    json_file.write(json.dumps(mapping_dict))
json_file.close()

# assign class_weights to counter class imbalance!
classTotals = labels.sum(axis=0)
data_class_weights = classTotals.max() / classTotals
print("[INFO] class weights (min, max) =", (data_class_weights.min(),
    data_class_weights.max()))

# split train & val set => np.ndarray
trainX, valX, trainY, valY = train_test_split(data, labels, test_size=0.2, \
        stratify=labels, random_state=42)
print("[INFO] training data has %d samples over %d classes" \
        % (trainX.shape[0], len(set(trainY.argmax(axis=1)))))
print("[INFO] validation data has %d samples over %d classes" \
        % (valX.shape[0], len(set(valY.argmax(axis=1)))))

# apply mean substraction of mean_img
mean_img = np.mean(trainX, axis=0)
trainX -= mean_img
valX -= mean_img

# store in BGR order because SimpleDatasetLoader uses cv2
mean_dict = {"B" : np.mean(mean_img[:, :, 0]), "G" : np.mean(mean_img[:, :, 1]), "R" :
        np.mean(mean_img[:, :, 2])}
with open(os.path.sep.join([args["checkpoints"], "trainX_mean.json"]), "w") as jf:
    jf.write(json.dumps(mean_dict))
jf.close()

# del some not useful vars
del data


## build model
# opt = Adam(lr=INIT_LR)
opt = SGD(lr=INIT_LR, momentum=0.9)

if args["model"] is None:
    print("[INFO] compiling model...")
    model = ResNet50(64, 64, 3, classes, reg=5e-4, bnEps=2e-5, bnMom=0.9)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=METRICS)
else:
    print("[INFO] loading %s ..." % args["model"])
    # need to specify extra metrics!
    model = load_model(args["model"], custom_objects={"f1_score" : f1_score}) 
    print("[INFO] old learning rate =", K.get_value(model.optimizer.lr))
    K.set_value(model.optimizer.lr, INIT_LR)
    print("[INFO] new learning rate =", K.get_value(model.optimizer.lr))

# set up callbacks
FIG_PATH = os.path.sep.join([args["checkpoints"], "resnet50_learning_curve.png"])
JSON_PATH = os.path.sep.join([args["checkpoints"], "resnet50.json"])
callbacks = [
        EpochCheckpoint(args["checkpoints"], every=5, startAt=args["start_epoch"]), 
        TrainingMonitor(FIG_PATH, jsonPath=JSON_PATH, startAt=args["start_epoch"]),
        LearningRateScheduler(poly_decay),  # Linear decay
        ]

# training
print("[INFO] training...")
trainGen = aug.flow(trainX, trainY, batch_size=BATCH)
model.fit_generator(
        trainGen,
        steps_per_epoch=len(trainX) // BATCH, 
        validation_data=(valX, valY),
        validation_steps=len(valX) // BATCH,
        epochs=EPOCHS, 
        max_queue_size=BATCH,
        callbacks=callbacks,
        class_weight=data_class_weights,   # counter imbalanced classes
        #use_multiprocessing=True,          # use multi_gpu
        verbose=1)

print("[INFO] evaluating...")
predictions = model.predict(valX, batch_size=BATCH)
print(classification_report(predictions.argmax(axis=1), valY.argmax(axis=1), \
        target_names=list(set(classNames))))

print("[INFO] Done!")

