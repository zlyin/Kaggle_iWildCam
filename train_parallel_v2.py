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
from parallelcheckpoint import ParallelCheckpoint
sys.path.append("../DL2CV/Orca/preprocessing")
from imagetoarraypreprocessor import ImageToArrayPreprocessor
from simplepreprocessor import SimplePreprocessor
from aspectawarepreprocessor import AspectAwarePreprocessor
from meanpreprocessor import MeanPreprocessor
sys.path.append("../DL2CV/Orca/datasets")
from simpledatasetloader import SimpleDatasetLoader
from utils.metrics import recall, precision, f1_score
sys.path.append("../DL2CV/Orca/io")
from hdf5datasetgenerator import HDF5DatasetGenerator
sys.path.append("./utils")
from preprocessings import CLAHE, SimpleWhiteBalance
from label_smoothing import smooth_labels

from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler
from keras.optimizers import SGD, Adam
from keras.models import load_model
from keras.utils import multi_gpu_model
from tensorflow.keras.losses import CategoricalCrossentropy
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
Train on iWildCam2020 with new ResNet50 (with an extra first block) for exp5~
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
    power = 2.5

    # compute
    lr = baseLR * (1 - (epoch / float(maxEpochs))) ** power
    return lr


## Build arguments parser
parser = argparse.ArgumentParser()
parser.add_argument("-c", "--checkpoints", required=True, help="path to the checkpoints directory")
parser.add_argument("-m", "--model", type=str, help="path to a specific loaded model")
parser.add_argument("-s", "--start_epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("-lsohe", "--label_smooth_one_hot", type=bool, default=False, help="if apply label smoothing to one-hot encoded labels")
args = vars(parser.parse_args())
assert os.path.exists(args["checkpoints"])


## Hyperparams
EPOCHS = 40
INIT_LR = 1e-1
BATCH = 64 * 2
NUM_CLASSES = 209   # need to double check with each dataset folder
METRICS = ["accuracy", f1_score]
SMOOTH_ONEHOT = args["label_smooth_one_hot"]

INPUT = "./data/animal_crops_hdf5"
TRAIN_HDF5 = os.path.sep.join([INPUT, "smooth_animal_crops_train.hdf5"])
VAL_HDF5 = os.path.sep.join([INPUT, "smooth_animal_crops_val.hdf5"])
TEST_HDF5 = os.path.sep.join([INPUT, "smooth_animal_crops_test.hdf5"])
DATASET_MEAN = os.path.sep.join([INPUT, "animal_crops_train_mean.json"])
LABEL_MAPPING = os.path.sep.join([INPUT, "encodedLabels_to_categoryID_mapping.json"])
DATSET_CLASS_WEIGHT = os.path.sep.join([INPUT, "animal_crops_class_weights_frequency.json"])

# load encoded_class to category_id mapping...
mapping_dict = json.loads(open(LABEL_MAPPING, "r").read())
encodedLabel_to_className = mapping_dict["encodedLabel_to_className"]
className_to_categoryID = mapping_dict["className_to_categoryID"]

class_weights_dict = json.loads(open(DATSET_CLASS_WEIGHT, "r").read())
data_class_weights = list(class_weights_dict.values())
print("[INFO] class weights (min, max) =", (min(data_class_weights), max(data_class_weights)))


## augmentation
means = json.loads(open(DATASET_MEAN, "r").read())
mp = MeanPreprocessor(means["R"], means["G"], means["B"])
sp = SimplePreprocessor(64, 64)
iap = ImageToArrayPreprocessor()
#aap = AspectAwarePreprocessor(64, 64)
#sdl = SimpleDatasetLoader(preprocessors=[aap, mp, iap])

aug = ImageDataGenerator(
        horizontal_flip=True,
        )

# initiate data genrators
trainGen = HDF5DatasetGenerator(TRAIN_HDF5, 64, aug=aug, 
        preprocessors=[sp, mp, iap], binarize=False, classes=NUM_CLASSES, mode="r+")
valGen = HDF5DatasetGenerator(VAL_HDF5, 64, aug=aug, 
        preprocessors=[sp, mp, iap], binarize=False, classes=NUM_CLASSES, mode="r+")

# apply label smoothing to one-hot encoded labels
if SMOOTH_ONEHOT == True:
    print("[INFO] smoothing train labels...")
    train_labels = np.zeros((trainGen.numImages, NUM_CLASSES))
    trainGen.db["labels"].read_direct(train_labels)
    smoothed_train_labels = smooth_labels(train_labels, factor=0.01)
    trainGen.db["labels"].write_direct(smoothed_train_labels)

    print("[INFO] smoothing val labels...")
    val_labels = np.zeros((valGen.numImages, NUM_CLASSES))
    valGen.db["labels"].read_direct(val_labels)
    smoothed_val_labels = smooth_labels(val_labels, factor=0.01)
    valGen.db["labels"].write_direct(smoothed_val_labels)

    print("[INFO] Label smooth finished, restart training!")
    sys.exit()


"""
- build parallel model on 2 GPUs
"""
# opt = Adam(lr=INIT_LR)
opt = SGD(lr=INIT_LR, momentum=0.9)

if args["model"] is None:
    print("[INFO] compiling distributed model...")
    # create model on CPU
    with tf.device('/cpu:0'):

        #model = ResNet50(64, 64, 3, NUM_CLASSES, reg=5e-4, bnEps=2e-5,
        #        bnMom=0.9)

        # EPX 1-4 & 6
        model = ResNet50(64, 64, 3, NUM_CLASSES, reg=5e-4, bnEps=2e-5,
                bnMom=0.9, dataset="cifar")

    # create distribute strategy for TF2.0
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        parallel_model = multi_gpu_model(model, gpus=2)
        parallel_model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=METRICS)

else:
    print("[INFO] loading %s ..." % args["model"])
    # load single GPU model back! need to specify extra metrics!
    with tf.device('/cpu:0'):
        model = load_model(args["model"], custom_objects={"f1_score" : f1_score}) 

    # create distribute strategy for TF2.0
    new_strategy = tf.distribute.MirroredStrategy()
    with new_strategy.scope():
        parallel_model = multi_gpu_model(model, gpus=2)
        parallel_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=METRICS)

        print("[INFO] old learning rate =", K.get_value(parallel_model.optimizer.lr))
        K.set_value(parallel_model.optimizer.lr, INIT_LR)
        print("[INFO] new learning rate =", K.get_value(parallel_model.optimizer.lr))

# set up callbacks
FIG_PATH = os.path.sep.join([args["checkpoints"], "resnet50_learning_curve.png"])
JSON_PATH = os.path.sep.join([args["checkpoints"], "resnet50.json"])
callbacks = [
        # use new parallel checkpoint callback which saves single template model!
        ParallelCheckpoint(model, args["checkpoints"], every=5, startAt=args["start_epoch"]), 
        TrainingMonitor(FIG_PATH, jsonPath=JSON_PATH, startAt=args["start_epoch"]),
        LearningRateScheduler(poly_decay),
        ]

# distribute BATCH to 2 GPUs, each GPUs computes BATCH / 2
print("[INFO] training...")
parallel_model.fit_generator(
        trainGen.generator(),
        steps_per_epoch=trainGen.numImages // BATCH, 
        validation_data=valGen.generator(),
        validation_steps=valGen.numImages // BATCH,
        epochs=EPOCHS, 
        max_queue_size=BATCH * 2,
        callbacks=callbacks,
        class_weight=data_class_weights,   # counter imbalanced classes
        verbose=1)

trainGen.close()
valGen.close()

#print(classification_report(predictions.argmax(axis=1), .argmax(axis=1), \
#        target_names=list(set(classNames))))

print("[INFO] Done!")

