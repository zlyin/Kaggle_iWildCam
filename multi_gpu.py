#!/usr/bin/python3.6
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import tensorflow as tf
tfconfig = tf.compat.v1.ConfigProto(allow_soft_placement=True)
tfconfig.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=tfconfig)

from keras.applications import Xception
from keras.models import load_model
from keras.utils import multi_gpu_model
from keras.callbacks import LearningRateScheduler
import numpy as np

import sys
sys.path.append("../DL2CV/Orca/callbacks")
from trainingmonitor import TrainingMonitor
from epochcheckpoint import EpochCheckpoint
from parallelcheckpoint import ParallelCheckpoint
from utils.metrics import recall, precision, f1_score
import argparse


## set up data
num_samples = 1000
height = 224
width = 224
num_classes = 1000

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--checkpoints", required=True, help="path to the checkpoints directory")
parser.add_argument("-m", "--model", type=str, help="path to a specific loaded model")
parser.add_argument("-s", "--start_epoch", type=int, default=0, help="epoch to start training from")
args = vars(parser.parse_args())
assert os.path.exists(args["checkpoints"])


# 生成虚拟数据
x = np.random.random((num_samples, height, width, 3))
y = np.random.random((num_samples, num_classes))


# 实例化基础模型（或者「模版」模型）, 我们推荐在 CPU 设备范围内做此操作，
# 这样模型的权重就会存储在 CPU 内存中, 否则它们会存储在 GPU 上，而完全被共享。
with tf.device('/cpu:0'):
    model = Xception(weights=None,
                     input_shape=(height, width, 3),
                     classes=num_classes)

## EXP1 = benchmark with single GPU (work)
#parallel_model = model
#parallel_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')


## EXP2 = use multi GPU without any callbacks (work)
## 复制模型到 8 个 GPU 上
#strategy = tf.distribute.MirroredStrategy()
#with strategy.scope():
#    parallel_model = multi_gpu_model(model, gpus=2)
#    parallel_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
#
## 这个 `fit` 调用将分布在2个GPU 上, 由于 batch size 是32*2, 每个 GPU 将处理 32 个样本。
#parallel_model.fit(x, y, epochs=2, batch_size=32 * 2)
#
## 通过模版模型存储模型（共享相同权重）：
#print("saving model")
#model.save('my_model.hdf5')
#
## load model back
#print("loading model back")
#new_model = load_model("my_model.hdf5")
#
#new_strategy = tf.distribute.MirroredStrategy()
#with new_strategy.scope():
#    new_parallel_model = multi_gpu_model(new_model, gpus=2)
#    new_parallel_model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
#new_parallel_model.fit(x, y, epochs=2, batch_size=32 * 2)


## EPX3 = use multi GPU with callbacks (work)
FIG_PATH = os.path.sep.join([args["checkpoints"], "debug.png"])
JSON_PATH = os.path.sep.join([args["checkpoints"], "debug.json"])
callbacks = [
        ParallelCheckpoint(model, args["checkpoints"], every=1,
            startAt=args["start_epoch"]), 
        TrainingMonitor(FIG_PATH, jsonPath=JSON_PATH, startAt=args["start_epoch"]),
        ]

if args["model"] is None:
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        parallel_model = multi_gpu_model(model, gpus=2)
        parallel_model.compile(loss='categorical_crossentropy',
                optimizer='rmsprop', metrics=["accuracy", f1_score])

    # 这个 `fit` 调用将分布在2个GPU 上, 由于 batch size 是32*2, 每个 GPU 将处理 32 个样本。
    parallel_model.fit(x, y, epochs=2, batch_size=32 * 2, callbacks=callbacks)

    # save template model! ==> the model saved by new ParallelCheckpoint
    print("saving model")
    del parallel_model

else:
    # load model back
    print("loading model back")
    new_model = load_model(args["model"], custom_objects={"f1_score" : f1_score})

    new_strategy = tf.distribute.MirroredStrategy()
    with new_strategy.scope():
        new_parallel_model = multi_gpu_model(new_model, gpus=2)
        new_parallel_model.compile(loss='categorical_crossentropy',
                optimizer='rmsprop', metrics=["accuracy", f1_score])
    new_parallel_model.fit(x, y, epochs=2, batch_size=32 * 2, callbacks=callbacks)
    

