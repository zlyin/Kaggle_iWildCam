# import pacakges
import keras.backend as K
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from sklearn.metrics import f1_score as sk_f1_score
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

def recall(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)
    # tn = K.sum(K.cast((1 - y_true) * (1-y_pred), 'float'), axis=0)
    #fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)

    # calculate recall = TP / (TP + FN)
    recall = tp / (tp + fn + K.epsilon())
    recall = tf.where(tf.math.is_nan(recall), tf.zeros_like(recall), recall)
    return K.mean(recall)

def precision(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    # tn = K.sum(K.cast((1 - y_true) * (1-y_pred), 'float'), axis=0)
    #fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    # calculate precision = TP / (TP + FP)
    precision = tp / (tp + fp + K.epsilon())
    precision = tf.where(tf.math.is_nan(precision), tf.zeros_like(precision), precision)
    return K.mean(precision)


def f1_score(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    # tn = K.sum(K.cast((1 - y_true) * (1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)



if __name__ == "__main__":

    y_true = np.array([[1,1,0,0,1], [1,0,1,1,0], [0,1,1,0,0]])
    y_pred = np.array([[0,1,1,1,1], [1,0,0,1,1], [1,0,1,0,0]])

    with tf.compat.v1.Session() as sess:
        print("precision=", sess.run(precision(y_true, y_pred)))
        print("recall=", sess.run(recall(y_true, y_pred)))
        print("f1 =", sess.run(f1_score(y_true, y_pred)))


    print("precision=", precision_score(y_true, y_pred, average="macro"))
    print("recall=", recall_score(y_true, y_pred, average="macro"))
    print("f1 =", sk_f1_score(y_true, y_pred, average="macro"))
