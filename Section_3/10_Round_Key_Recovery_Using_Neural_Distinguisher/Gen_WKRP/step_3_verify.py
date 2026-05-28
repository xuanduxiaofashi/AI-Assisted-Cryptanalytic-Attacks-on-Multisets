import sys
import os
import math
import numpy as np
import os
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.layers import Dense, Conv1D, Input, Reshape, Permute, Add, Flatten, BatchNormalization, Activation
from keras.regularizers import l2
from keras.models import Model
from tensorflow.keras import backend as K
import random
import tensorflow as tf
from sklearn.metrics import accuracy_score
from keras.utils import get_custom_objects
from neural_model import   cnn_model



model_name = "cnn_model"

train_func = cnn_model




model_name = "train_models/" + "cnn_model/" + "cnn_model.h5"


wdir = './train_u_v_models/' + model_name + "/"

os.makedirs(wdir, exist_ok=True)

class TPRMetric(tf.keras.metrics.Metric):
    def __init__(self, name="tpr", **kwargs):
        super(TPRMetric, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="tp", initializer="zeros")
        self.false_negatives = self.add_weight(name="fn", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):

        y_pred = tf.round(y_pred)
        
        tp = tf.reduce_sum(tf.cast(y_true * y_pred, tf.float32))
        fn = tf.reduce_sum(tf.cast(y_true * (1 - y_pred), tf.float32))

        self.true_positives.assign_add(tp)
        self.false_negatives.assign_add(fn)

    def result(self):
        return self.true_positives / (self.true_positives + self.false_negatives + tf.keras.backend.epsilon())


class TNRMetric(tf.keras.metrics.Metric):
    def __init__(self, name="tnr", **kwargs):
        super(TNRMetric, self).__init__(name=name, **kwargs)
        self.true_negatives = self.add_weight(name="tn", initializer="zeros")
        self.false_positives = self.add_weight(name="fp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
   
        y_pred = tf.round(y_pred)
        
        tn = tf.reduce_sum(tf.cast((1 - y_true) * (1 - y_pred), tf.float32))
        fp = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, tf.float32))
        
        self.true_negatives.assign_add(tn)
        self.false_positives.assign_add(fp)

    def result(self):
        return self.true_negatives / (self.true_negatives + self.false_positives + tf.keras.backend.epsilon())
    


get_custom_objects().update({'TPRMetric': TPRMetric, 'TNRMetric': TNRMetric})


def make_checkpoint(datei):
  res = ModelCheckpoint(datei, monitor='val_loss', save_best_only = True)
  return(res)

def cyclic_lr(num_epochs, high_lr, low_lr):
  res = lambda i: low_lr + ((num_epochs-1) - i % num_epochs)/(num_epochs-1) * (high_lr - low_lr)
  return(res)



def verify( a, b):

    model = tf.keras.models.load_model(model_name)


    file_name_list = f"cipher_txt_npy_8r/cipher_8r_2_16_key_{a}_{b}.npy"


    X_val = np.load(file_name_list)

    predictions = model.predict(X_val)
    print(predictions)
    mean_value = np.mean(predictions)

    std = np.std(predictions)
    with open('output.txt', 'a') as f:
        f.write(f"{a}_{b}, mean: {mean_value}, std : {std}\n")







if __name__ == "__main__":


    a = int(sys.argv[1])
    b = int(sys.argv[2])


    verify(a, b)
    
