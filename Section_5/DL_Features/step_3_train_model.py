import sys
import os
import math
import itertools
import numpy as np
import os
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.layers import Dense, Conv1D, Input, Reshape, Permute, Add, Flatten, BatchNormalization, Activation
from keras.regularizers import l2
from keras.models import Model
from tensorflow.keras import backend as K
import random
import tensorflow as tf
import time

from neural_model import cnn_model



model_name = "cnn_model"

train_func = cnn_model



wdir = './train_models/' + model_name + "/"

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
    


def make_checkpoint(datei):
  res = ModelCheckpoint(datei, monitor='val_loss', save_best_only = True)
  return(res)

def cyclic_lr(num_epochs, high_lr, low_lr):
  res = lambda i: low_lr + ((num_epochs-1) - i % num_epochs)/(num_epochs-1) * (high_lr - low_lr)
  return(res)





def gen_train_or_val_data(train_num, load_file_name):


    train_num_label_1 = train_num // 2
    train_X_label_1 = np.load(load_file_name)
    train_X_label_0 = np.random.choice([0, 1], size=(train_num_label_1, 16, 8))

    temp_1 = train_X_label_0[:, :, 0: 4] 
    temp_2 = train_X_label_0[:, :, 4: 8]
   

    x_3_6r_2 = temp_1[:, :, 0]
    x_2_6r_2 = temp_1[:, :, 1]
    x_1_6r_2 = temp_1[:, :, 2]
    x_0_6r_2 = temp_1[:, :, 3]

    x_3_6r_7 = temp_2[:, :, 0]
    x_2_6r_7 = temp_2[:, :, 1]
    x_1_6r_7 = temp_2[:, :, 2]
    x_0_6r_7 = temp_2[:, :, 3]


    train_X_column_1 = x_3_6r_2^x_3_6r_7
    train_X_column_2 = x_3_6r_2^x_1_6r_7
    train_X_column_3 = x_3_6r_2^x_0_6r_7 
    train_X_column_4 = x_2_6r_2^x_2_6r_7
    train_X_column_5 = x_2_6r_2^x_0_6r_7 
    train_X_column_6 = x_1_6r_2^x_3_6r_7
    train_X_column_7 = x_1_6r_2^x_1_6r_7
    train_X_column_8 = x_1_6r_2^x_0_6r_7 
    train_X_column_9 = x_0_6r_2 ^x_3_6r_7
    train_X_column_10 = x_0_6r_2 ^x_2_6r_7
    train_X_column_11 = x_0_6r_2 ^x_1_6r_7
    train_X_column_12 = x_0_6r_2 ^x_0_6r_7 
    train_X_column_13 = x_3_6r_2^x_2_6r_2^x_3_6r_7
    train_X_column_14 = x_3_6r_2^x_2_6r_2^x_1_6r_7
    train_X_column_15 = x_3_6r_2^x_2_6r_2^x_0_6r_7 
    train_X_column_16 = x_3_6r_2^x_1_6r_2^x_3_6r_7
    train_X_column_17 = x_3_6r_2^x_1_6r_2^x_1_6r_7
    train_X_column_18 = x_3_6r_2^x_1_6r_2^x_0_6r_7 
    train_X_column_19 = x_3_6r_2^x_0_6r_2 ^x_3_6r_7
    train_X_column_20 = x_3_6r_2^x_0_6r_2 ^x_2_6r_7
    train_X_column_21 = x_3_6r_2^x_0_6r_2 ^x_1_6r_7
    train_X_column_22 = x_3_6r_2^x_0_6r_2 ^x_0_6r_7 
    train_X_column_23 = x_3_6r_2^x_3_6r_7^x_2_6r_7
    train_X_column_24 = x_3_6r_2^x_3_6r_7^x_1_6r_7
    train_X_column_25 = x_3_6r_2^x_3_6r_7^x_0_6r_7 
    train_X_column_26 = x_3_6r_2^x_2_6r_7^x_1_6r_7
    train_X_column_27 = x_3_6r_2^x_2_6r_7^x_0_6r_7 
    train_X_column_28 = x_3_6r_2^x_1_6r_7^x_0_6r_7 
    train_X_column_29 = x_2_6r_2^x_1_6r_2^x_3_6r_7
    train_X_column_30 = x_2_6r_2^x_1_6r_2^x_2_6r_7
    train_X_column_31 = x_2_6r_2^x_1_6r_2^x_1_6r_7
    train_X_column_32 = x_2_6r_2^x_1_6r_2^x_0_6r_7 
    train_X_column_33 = x_2_6r_2^x_0_6r_2 ^x_3_6r_7
    train_X_column_34 = x_2_6r_2^x_0_6r_2 ^x_2_6r_7
    train_X_column_35 = x_2_6r_2^x_0_6r_2 ^x_1_6r_7
    train_X_column_36 = x_2_6r_2^x_0_6r_2 ^x_0_6r_7 
    train_X_column_37 = x_2_6r_2^x_3_6r_7^x_0_6r_7 
    train_X_column_38 = x_2_6r_2^x_2_6r_7^x_1_6r_7
    train_X_column_39 = x_2_6r_2^x_2_6r_7^x_0_6r_7 
    train_X_column_40 = x_2_6r_2^x_1_6r_7^x_0_6r_7 
    train_X_column_41 = x_1_6r_2^x_0_6r_2 ^x_3_6r_7
    train_X_column_42 = x_1_6r_2^x_0_6r_2 ^x_2_6r_7
    train_X_column_43 = x_1_6r_2^x_0_6r_2 ^x_1_6r_7
    train_X_column_44 = x_1_6r_2^x_0_6r_2 ^x_0_6r_7 
    train_X_column_45 = x_1_6r_2^x_3_6r_7^x_2_6r_7
    train_X_column_46 = x_1_6r_2^x_3_6r_7^x_1_6r_7
    train_X_column_47 = x_1_6r_2^x_3_6r_7^x_0_6r_7 
    train_X_column_48 = x_1_6r_2^x_2_6r_7^x_1_6r_7
    train_X_column_49 = x_1_6r_2^x_2_6r_7^x_0_6r_7 
    train_X_column_50 = x_1_6r_2^x_1_6r_7^x_0_6r_7 
    train_X_column_51 = x_0_6r_2 ^x_3_6r_7^x_2_6r_7
    train_X_column_52 = x_0_6r_2 ^x_3_6r_7^x_1_6r_7
    train_X_column_53 = x_0_6r_2 ^x_3_6r_7^x_0_6r_7 
    train_X_column_54 = x_0_6r_2 ^x_2_6r_7^x_1_6r_7
    train_X_column_55 = x_0_6r_2 ^x_2_6r_7^x_0_6r_7 
    train_X_column_56 = x_0_6r_2 ^x_1_6r_7^x_0_6r_7 
    train_X_column_57 = x_3_6r_2^x_2_6r_2^x_1_6r_2^x_3_6r_7
    train_X_column_58 = x_3_6r_2^x_2_6r_2^x_1_6r_2^x_2_6r_7
    train_X_column_59 = x_3_6r_2^x_2_6r_2^x_1_6r_2^x_1_6r_7
    train_X_column_60 = x_3_6r_2^x_2_6r_2^x_1_6r_2^x_0_6r_7 
    train_X_column_61 = x_3_6r_2^x_2_6r_2^x_0_6r_2 ^x_3_6r_7
    train_X_column_62 = x_3_6r_2^x_2_6r_2^x_0_6r_2 ^x_2_6r_7
    train_X_column_63 = x_3_6r_2^x_2_6r_2^x_0_6r_2 ^x_1_6r_7
    train_X_column_64 = x_3_6r_2^x_2_6r_2^x_0_6r_2 ^x_0_6r_7 
    train_X_column_65 = x_3_6r_2^x_2_6r_2^x_3_6r_7^x_2_6r_7
    train_X_column_66 = x_3_6r_2^x_2_6r_2^x_3_6r_7^x_1_6r_7
    train_X_column_67 = x_3_6r_2^x_2_6r_2^x_3_6r_7^x_0_6r_7 
    train_X_column_68 = x_3_6r_2^x_2_6r_2^x_2_6r_7^x_1_6r_7
    train_X_column_69 = x_3_6r_2^x_2_6r_2^x_2_6r_7^x_0_6r_7 
    train_X_column_70 = x_3_6r_2^x_2_6r_2^x_1_6r_7^x_0_6r_7 
    train_X_column_71 = x_3_6r_2^x_1_6r_2^x_0_6r_2 ^x_3_6r_7
    train_X_column_72 = x_3_6r_2^x_1_6r_2^x_0_6r_2 ^x_2_6r_7
    train_X_column_73 = x_3_6r_2^x_1_6r_2^x_0_6r_2 ^x_1_6r_7
    train_X_column_74 = x_3_6r_2^x_1_6r_2^x_0_6r_2 ^x_0_6r_7 
    train_X_column_75 = x_3_6r_2^x_1_6r_2^x_3_6r_7^x_2_6r_7
    train_X_column_76 = x_3_6r_2^x_1_6r_2^x_3_6r_7^x_1_6r_7
    train_X_column_77 = x_3_6r_2^x_1_6r_2^x_3_6r_7^x_0_6r_7 
    train_X_column_78 = x_3_6r_2^x_1_6r_2^x_2_6r_7^x_1_6r_7
    train_X_column_79 = x_3_6r_2^x_1_6r_2^x_2_6r_7^x_0_6r_7 
    train_X_column_80 = x_3_6r_2^x_1_6r_2^x_1_6r_7^x_0_6r_7 
    train_X_column_81 = x_3_6r_2^x_0_6r_2 ^x_3_6r_7^x_2_6r_7
    train_X_column_82 = x_3_6r_2^x_0_6r_2 ^x_3_6r_7^x_1_6r_7
    train_X_column_83 = x_3_6r_2^x_0_6r_2 ^x_3_6r_7^x_0_6r_7 
    train_X_column_84 = x_3_6r_2^x_0_6r_2 ^x_2_6r_7^x_1_6r_7
    train_X_column_85 = x_3_6r_2^x_0_6r_2 ^x_2_6r_7^x_0_6r_7 
    train_X_column_86 = x_3_6r_2^x_0_6r_2 ^x_1_6r_7^x_0_6r_7 
    train_X_column_87 = x_3_6r_2^x_3_6r_7^x_2_6r_7^x_1_6r_7
    train_X_column_88 = x_3_6r_2^x_3_6r_7^x_2_6r_7^x_0_6r_7 
    train_X_column_89 = x_3_6r_2^x_3_6r_7^x_1_6r_7^x_0_6r_7 
    train_X_column_90 = x_3_6r_2^x_2_6r_7^x_1_6r_7^x_0_6r_7 
    train_X_column_91 = x_2_6r_2^x_1_6r_2^x_0_6r_2 ^x_3_6r_7
    train_X_column_92 = x_2_6r_2^x_1_6r_2^x_0_6r_2 ^x_2_6r_7
    train_X_column_93 = x_2_6r_2^x_1_6r_2^x_0_6r_2 ^x_1_6r_7
    train_X_column_94 = x_2_6r_2^x_1_6r_2^x_0_6r_2 ^x_0_6r_7 
    train_X_column_95 = x_2_6r_2^x_1_6r_2^x_3_6r_7^x_2_6r_7
    train_X_column_96 = x_2_6r_2^x_1_6r_2^x_3_6r_7^x_1_6r_7
    train_X_column_97 = x_2_6r_2^x_1_6r_2^x_3_6r_7^x_0_6r_7 
    train_X_column_98 = x_2_6r_2^x_1_6r_2^x_2_6r_7^x_1_6r_7
    train_X_column_99 = x_2_6r_2^x_1_6r_2^x_2_6r_7^x_0_6r_7 
    train_X_column_100 = x_2_6r_2^x_1_6r_2^x_1_6r_7^x_0_6r_7 
    train_X_column_101 = x_2_6r_2^x_0_6r_2 ^x_3_6r_7^x_2_6r_7
    train_X_column_102 = x_2_6r_2^x_0_6r_2 ^x_3_6r_7^x_1_6r_7
    train_X_column_103 = x_2_6r_2^x_0_6r_2 ^x_3_6r_7^x_0_6r_7 
    train_X_column_104 = x_2_6r_2^x_0_6r_2 ^x_2_6r_7^x_1_6r_7
    train_X_column_105 = x_2_6r_2^x_0_6r_2 ^x_2_6r_7^x_0_6r_7 
    train_X_column_106 = x_2_6r_2^x_0_6r_2 ^x_1_6r_7^x_0_6r_7 
    train_X_column_107 = x_2_6r_2^x_3_6r_7^x_2_6r_7^x_1_6r_7
    train_X_column_108 = x_2_6r_2^x_3_6r_7^x_2_6r_7^x_0_6r_7 
    train_X_column_109 = x_2_6r_2^x_3_6r_7^x_1_6r_7^x_0_6r_7 
    train_X_column_110 = x_2_6r_2^x_2_6r_7^x_1_6r_7^x_0_6r_7 
    train_X_column_111 = x_1_6r_2^x_0_6r_2 ^x_3_6r_7^x_2_6r_7
    train_X_column_112 = x_1_6r_2^x_0_6r_2 ^x_3_6r_7^x_1_6r_7
    train_X_column_113 = x_1_6r_2^x_0_6r_2 ^x_3_6r_7^x_0_6r_7 
    train_X_column_114 = x_1_6r_2^x_0_6r_2 ^x_2_6r_7^x_1_6r_7
    train_X_column_115 = x_1_6r_2^x_0_6r_2 ^x_2_6r_7^x_0_6r_7 
    train_X_column_116 = x_1_6r_2^x_0_6r_2 ^x_1_6r_7^x_0_6r_7 
    train_X_column_117 = x_1_6r_2^x_3_6r_7^x_2_6r_7^x_1_6r_7
    train_X_column_118 = x_1_6r_2^x_3_6r_7^x_2_6r_7^x_0_6r_7 
    train_X_column_119 = x_1_6r_2^x_3_6r_7^x_1_6r_7^x_0_6r_7 
    train_X_column_120 = x_1_6r_2^x_2_6r_7^x_1_6r_7^x_0_6r_7 
    train_X_column_121 = x_0_6r_2 ^x_3_6r_7^x_2_6r_7^x_1_6r_7
    train_X_column_122 = x_0_6r_2 ^x_3_6r_7^x_2_6r_7^x_0_6r_7 
    train_X_column_123 = x_0_6r_2 ^x_3_6r_7^x_1_6r_7^x_0_6r_7 
    train_X_column_124 = x_0_6r_2 ^x_2_6r_7^x_1_6r_7^x_0_6r_7 
    train_X_column_125 = x_3_6r_2^x_2_6r_2^x_1_6r_2^x_0_6r_2 ^x_3_6r_7
    train_X_column_126 = x_3_6r_2^x_2_6r_2^x_1_6r_2^x_0_6r_2 ^x_2_6r_7
    train_X_column_127 = x_3_6r_2^x_2_6r_2^x_1_6r_2^x_0_6r_2 ^x_1_6r_7
    train_X_column_128 = x_3_6r_2^x_2_6r_2^x_1_6r_2^x_0_6r_2 ^x_0_6r_7 
    train_X_column_129 = x_3_6r_2^x_2_6r_2^x_1_6r_2^x_3_6r_7^x_2_6r_7
    train_X_column_130 = x_3_6r_2^x_2_6r_2^x_1_6r_2^x_3_6r_7^x_1_6r_7
    train_X_column_131 = x_3_6r_2^x_2_6r_2^x_1_6r_2^x_3_6r_7^x_0_6r_7 
    train_X_column_132 = x_3_6r_2^x_2_6r_2^x_1_6r_2^x_2_6r_7^x_1_6r_7
    train_X_column_133 = x_3_6r_2^x_2_6r_2^x_1_6r_2^x_2_6r_7^x_0_6r_7 
    train_X_column_134 = x_3_6r_2^x_2_6r_2^x_1_6r_2^x_1_6r_7^x_0_6r_7 
    train_X_column_135 = x_3_6r_2^x_2_6r_2^x_0_6r_2 ^x_3_6r_7^x_2_6r_7
    train_X_column_136 = x_3_6r_2^x_2_6r_2^x_0_6r_2 ^x_3_6r_7^x_1_6r_7
    train_X_column_137 = x_3_6r_2^x_2_6r_2^x_0_6r_2 ^x_3_6r_7^x_0_6r_7 
    train_X_column_138 = x_3_6r_2^x_2_6r_2^x_0_6r_2 ^x_2_6r_7^x_1_6r_7
    train_X_column_139 = x_3_6r_2^x_2_6r_2^x_0_6r_2 ^x_2_6r_7^x_0_6r_7 
    train_X_column_140 = x_3_6r_2^x_2_6r_2^x_0_6r_2 ^x_1_6r_7^x_0_6r_7 
    train_X_column_141 = x_3_6r_2^x_2_6r_2^x_3_6r_7^x_2_6r_7^x_1_6r_7
    train_X_column_142 = x_3_6r_2^x_2_6r_2^x_3_6r_7^x_2_6r_7^x_0_6r_7 
    train_X_column_143 = x_3_6r_2^x_2_6r_2^x_3_6r_7^x_1_6r_7^x_0_6r_7 
    train_X_column_144 = x_3_6r_2^x_2_6r_2^x_2_6r_7^x_1_6r_7^x_0_6r_7 
    train_X_column_145 = x_3_6r_2^x_1_6r_2^x_0_6r_2 ^x_3_6r_7^x_2_6r_7
    train_X_column_146 = x_3_6r_2^x_1_6r_2^x_0_6r_2 ^x_3_6r_7^x_1_6r_7
    train_X_column_147 = x_3_6r_2^x_1_6r_2^x_0_6r_2 ^x_3_6r_7^x_0_6r_7 
    train_X_column_148 = x_3_6r_2^x_1_6r_2^x_0_6r_2 ^x_2_6r_7^x_1_6r_7
    train_X_column_149 = x_3_6r_2^x_1_6r_2^x_0_6r_2 ^x_2_6r_7^x_0_6r_7 
    train_X_column_150 = x_3_6r_2^x_1_6r_2^x_0_6r_2 ^x_1_6r_7^x_0_6r_7 
    train_X_column_151 = x_3_6r_2^x_1_6r_2^x_3_6r_7^x_2_6r_7^x_1_6r_7
    train_X_column_152 = x_3_6r_2^x_1_6r_2^x_3_6r_7^x_2_6r_7^x_0_6r_7 
    train_X_column_153 = x_3_6r_2^x_1_6r_2^x_3_6r_7^x_1_6r_7^x_0_6r_7 
    train_X_column_154 = x_3_6r_2^x_1_6r_2^x_2_6r_7^x_1_6r_7^x_0_6r_7 
    train_X_column_155 = x_3_6r_2^x_0_6r_2 ^x_3_6r_7^x_2_6r_7^x_1_6r_7
    train_X_column_156 = x_3_6r_2^x_0_6r_2 ^x_3_6r_7^x_2_6r_7^x_0_6r_7 
    train_X_column_157 = x_3_6r_2^x_0_6r_2 ^x_3_6r_7^x_1_6r_7^x_0_6r_7 
    train_X_column_158 = x_3_6r_2^x_0_6r_2 ^x_2_6r_7^x_1_6r_7^x_0_6r_7 
    train_X_column_159 = x_3_6r_2^x_3_6r_7^x_2_6r_7^x_1_6r_7^x_0_6r_7 
    train_X_column_160 = x_2_6r_2^x_1_6r_2^x_0_6r_2 ^x_3_6r_7^x_2_6r_7
    train_X_column_161 = x_2_6r_2^x_1_6r_2^x_0_6r_2 ^x_3_6r_7^x_1_6r_7
    train_X_column_162 = x_2_6r_2^x_1_6r_2^x_0_6r_2 ^x_3_6r_7^x_0_6r_7 
    train_X_column_163 = x_2_6r_2^x_1_6r_2^x_0_6r_2 ^x_2_6r_7^x_1_6r_7
    train_X_column_164 = x_2_6r_2^x_1_6r_2^x_0_6r_2 ^x_2_6r_7^x_0_6r_7 
    train_X_column_165 = x_2_6r_2^x_1_6r_2^x_0_6r_2 ^x_1_6r_7^x_0_6r_7 
    train_X_column_166 = x_2_6r_2^x_1_6r_2^x_3_6r_7^x_2_6r_7^x_1_6r_7
    train_X_column_167 = x_2_6r_2^x_1_6r_2^x_3_6r_7^x_2_6r_7^x_0_6r_7 
    train_X_column_168 = x_2_6r_2^x_1_6r_2^x_3_6r_7^x_1_6r_7^x_0_6r_7 
    train_X_column_169 = x_2_6r_2^x_1_6r_2^x_2_6r_7^x_1_6r_7^x_0_6r_7 
    train_X_column_170 = x_2_6r_2^x_0_6r_2 ^x_3_6r_7^x_2_6r_7^x_1_6r_7
    train_X_column_171 = x_2_6r_2^x_0_6r_2 ^x_3_6r_7^x_2_6r_7^x_0_6r_7 
    train_X_column_172 = x_2_6r_2^x_0_6r_2 ^x_3_6r_7^x_1_6r_7^x_0_6r_7 
    train_X_column_173 = x_2_6r_2^x_0_6r_2 ^x_2_6r_7^x_1_6r_7^x_0_6r_7 
    train_X_column_174 = x_2_6r_2^x_3_6r_7^x_2_6r_7^x_1_6r_7^x_0_6r_7 
    train_X_column_175 = x_1_6r_2^x_0_6r_2 ^x_3_6r_7^x_2_6r_7^x_1_6r_7
    train_X_column_176 = x_1_6r_2^x_0_6r_2 ^x_3_6r_7^x_2_6r_7^x_0_6r_7 
    train_X_column_177 = x_1_6r_2^x_0_6r_2 ^x_3_6r_7^x_1_6r_7^x_0_6r_7 
    train_X_column_178 = x_1_6r_2^x_0_6r_2 ^x_2_6r_7^x_1_6r_7^x_0_6r_7 
    train_X_column_179 = x_1_6r_2^x_3_6r_7^x_2_6r_7^x_1_6r_7^x_0_6r_7 
    train_X_column_180 = x_0_6r_2 ^x_3_6r_7^x_2_6r_7^x_1_6r_7^x_0_6r_7 
    train_X_column_181 = x_3_6r_2^x_2_6r_2^x_1_6r_2^x_0_6r_2 ^x_3_6r_7^x_2_6r_7
    train_X_column_182 = x_3_6r_2^x_2_6r_2^x_1_6r_2^x_0_6r_2 ^x_3_6r_7^x_1_6r_7
    train_X_column_183 = x_3_6r_2^x_2_6r_2^x_1_6r_2^x_0_6r_2 ^x_3_6r_7^x_0_6r_7 
    train_X_column_184 = x_3_6r_2^x_2_6r_2^x_1_6r_2^x_0_6r_2 ^x_2_6r_7^x_1_6r_7
    train_X_column_185 = x_3_6r_2^x_2_6r_2^x_1_6r_2^x_0_6r_2 ^x_2_6r_7^x_0_6r_7 
    train_X_column_186 = x_3_6r_2^x_2_6r_2^x_1_6r_2^x_0_6r_2 ^x_1_6r_7^x_0_6r_7 
    train_X_column_187 = x_3_6r_2^x_2_6r_2^x_1_6r_2^x_3_6r_7^x_2_6r_7^x_1_6r_7
    train_X_column_188 = x_3_6r_2^x_2_6r_2^x_1_6r_2^x_3_6r_7^x_2_6r_7^x_0_6r_7 
    train_X_column_189 = x_3_6r_2^x_2_6r_2^x_1_6r_2^x_3_6r_7^x_1_6r_7^x_0_6r_7 
    train_X_column_190 = x_3_6r_2^x_2_6r_2^x_1_6r_2^x_2_6r_7^x_1_6r_7^x_0_6r_7 
    train_X_column_191 = x_3_6r_2^x_2_6r_2^x_0_6r_2 ^x_3_6r_7^x_2_6r_7^x_1_6r_7
    train_X_column_192 = x_3_6r_2^x_2_6r_2^x_0_6r_2 ^x_3_6r_7^x_2_6r_7^x_0_6r_7 
    train_X_column_193 = x_3_6r_2^x_2_6r_2^x_0_6r_2 ^x_3_6r_7^x_1_6r_7^x_0_6r_7 
    train_X_column_194 = x_3_6r_2^x_2_6r_2^x_0_6r_2 ^x_2_6r_7^x_1_6r_7^x_0_6r_7 
    train_X_column_195 = x_3_6r_2^x_2_6r_2^x_3_6r_7^x_2_6r_7^x_1_6r_7^x_0_6r_7 
    train_X_column_196 = x_3_6r_2^x_1_6r_2^x_0_6r_2 ^x_3_6r_7^x_2_6r_7^x_1_6r_7
    train_X_column_197 = x_3_6r_2^x_1_6r_2^x_0_6r_2 ^x_3_6r_7^x_2_6r_7^x_0_6r_7 
    train_X_column_198 = x_3_6r_2^x_1_6r_2^x_0_6r_2 ^x_3_6r_7^x_1_6r_7^x_0_6r_7 
    train_X_column_199 = x_3_6r_2^x_1_6r_2^x_0_6r_2 ^x_2_6r_7^x_1_6r_7^x_0_6r_7 
    train_X_column_200 = x_3_6r_2^x_1_6r_2^x_3_6r_7^x_2_6r_7^x_1_6r_7^x_0_6r_7 
    train_X_column_201 = x_3_6r_2^x_0_6r_2 ^x_3_6r_7^x_2_6r_7^x_1_6r_7^x_0_6r_7 
    train_X_column_202 = x_2_6r_2^x_1_6r_2^x_0_6r_2 ^x_3_6r_7^x_2_6r_7^x_1_6r_7
    train_X_column_203 = x_2_6r_2^x_1_6r_2^x_0_6r_2 ^x_3_6r_7^x_2_6r_7^x_0_6r_7 
    train_X_column_204 = x_2_6r_2^x_1_6r_2^x_0_6r_2 ^x_3_6r_7^x_1_6r_7^x_0_6r_7 
    train_X_column_205 = x_2_6r_2^x_1_6r_2^x_0_6r_2 ^x_2_6r_7^x_1_6r_7^x_0_6r_7 
    train_X_column_206 = x_2_6r_2^x_1_6r_2^x_3_6r_7^x_2_6r_7^x_1_6r_7^x_0_6r_7 
    train_X_column_207 = x_2_6r_2^x_0_6r_2 ^x_3_6r_7^x_2_6r_7^x_1_6r_7^x_0_6r_7 
    train_X_column_208 = x_1_6r_2^x_0_6r_2 ^x_3_6r_7^x_2_6r_7^x_1_6r_7^x_0_6r_7 
    train_X_column_209 = x_3_6r_2^x_2_6r_2^x_1_6r_2^x_0_6r_2 ^x_3_6r_7^x_2_6r_7^x_1_6r_7
    train_X_column_210 = x_3_6r_2^x_2_6r_2^x_1_6r_2^x_0_6r_2 ^x_3_6r_7^x_2_6r_7^x_0_6r_7 
    train_X_column_211 = x_3_6r_2^x_2_6r_2^x_1_6r_2^x_0_6r_2 ^x_3_6r_7^x_1_6r_7^x_0_6r_7 
    train_X_column_212 = x_3_6r_2^x_2_6r_2^x_1_6r_2^x_0_6r_2 ^x_2_6r_7^x_1_6r_7^x_0_6r_7 
    train_X_column_213 = x_3_6r_2^x_2_6r_2^x_1_6r_2^x_3_6r_7^x_2_6r_7^x_1_6r_7^x_0_6r_7 
    train_X_column_214 = x_3_6r_2^x_2_6r_2^x_0_6r_2 ^x_3_6r_7^x_2_6r_7^x_1_6r_7^x_0_6r_7 
    train_X_column_215 = x_3_6r_2^x_1_6r_2^x_0_6r_2 ^x_3_6r_7^x_2_6r_7^x_1_6r_7^x_0_6r_7 
    train_X_column_216 = x_2_6r_2^x_1_6r_2^x_0_6r_2 ^x_3_6r_7^x_2_6r_7^x_1_6r_7^x_0_6r_7 
    train_X_column_217 = x_3_6r_2^x_2_6r_2^x_1_6r_2^x_0_6r_2 ^x_3_6r_7^x_2_6r_7^x_1_6r_7^x_0_6r_7 



    train_Y_label_1 = np.ones((train_num_label_1, 1)) 
    train_Y_label_0 = np.zeros((train_num_label_1, 1))



    train_X_label_0 = np.concatenate([train_X_column_1, train_X_column_2, train_X_column_3, train_X_column_4, train_X_column_5, train_X_column_6, train_X_column_7, train_X_column_8, train_X_column_9, train_X_column_10, train_X_column_11, train_X_column_12, train_X_column_13, train_X_column_14, train_X_column_15, train_X_column_16, train_X_column_17, train_X_column_18, train_X_column_19, train_X_column_20, train_X_column_21, train_X_column_22, train_X_column_23, train_X_column_24, train_X_column_25, train_X_column_26, train_X_column_27, train_X_column_28, train_X_column_29, train_X_column_30, train_X_column_31, train_X_column_32, train_X_column_33, train_X_column_34, train_X_column_35, train_X_column_36, train_X_column_37, train_X_column_38, train_X_column_39, train_X_column_40, train_X_column_41, train_X_column_42, train_X_column_43, train_X_column_44, train_X_column_45, train_X_column_46, train_X_column_47, train_X_column_48, train_X_column_49, train_X_column_50, train_X_column_51, train_X_column_52, train_X_column_53, train_X_column_54, train_X_column_55, train_X_column_56, train_X_column_57, train_X_column_58, train_X_column_59, train_X_column_60, train_X_column_61, train_X_column_62, train_X_column_63, train_X_column_64, train_X_column_65, train_X_column_66, train_X_column_67, train_X_column_68, train_X_column_69, train_X_column_70, train_X_column_71, train_X_column_72, train_X_column_73, train_X_column_74, train_X_column_75, train_X_column_76, train_X_column_77, train_X_column_78, train_X_column_79, train_X_column_80, train_X_column_81, train_X_column_82, train_X_column_83, train_X_column_84, train_X_column_85, train_X_column_86, train_X_column_87, train_X_column_88, train_X_column_89, train_X_column_90, train_X_column_91, train_X_column_92, train_X_column_93, train_X_column_94, train_X_column_95, train_X_column_96, train_X_column_97, train_X_column_98, train_X_column_99, train_X_column_100, train_X_column_101, train_X_column_102, train_X_column_103, train_X_column_104, train_X_column_105, train_X_column_106, train_X_column_107, train_X_column_108, train_X_column_109, train_X_column_110, train_X_column_111, train_X_column_112, train_X_column_113, train_X_column_114, train_X_column_115, train_X_column_116, train_X_column_117, train_X_column_118, train_X_column_119, train_X_column_120, train_X_column_121, train_X_column_122, train_X_column_123, train_X_column_124, train_X_column_125, train_X_column_126, train_X_column_127, train_X_column_128, train_X_column_129, train_X_column_130, train_X_column_131, train_X_column_132, train_X_column_133, train_X_column_134, train_X_column_135, train_X_column_136, train_X_column_137, train_X_column_138, train_X_column_139, train_X_column_140, train_X_column_141, train_X_column_142, train_X_column_143, train_X_column_144, train_X_column_145, train_X_column_146, train_X_column_147, train_X_column_148, train_X_column_149, train_X_column_150, train_X_column_151, train_X_column_152, train_X_column_153, train_X_column_154, train_X_column_155, train_X_column_156, train_X_column_157, train_X_column_158, train_X_column_159, train_X_column_160, train_X_column_161, train_X_column_162, train_X_column_163, train_X_column_164, train_X_column_165, train_X_column_166, train_X_column_167, train_X_column_168, train_X_column_169, train_X_column_170, train_X_column_171, train_X_column_172, train_X_column_173, train_X_column_174, train_X_column_175, train_X_column_176, train_X_column_177, train_X_column_178, train_X_column_179, train_X_column_180, train_X_column_181, train_X_column_182, train_X_column_183, train_X_column_184, train_X_column_185, train_X_column_186, train_X_column_187, train_X_column_188, train_X_column_189, train_X_column_190, train_X_column_191, train_X_column_192, train_X_column_193, train_X_column_194, train_X_column_195, train_X_column_196, train_X_column_197, train_X_column_198, train_X_column_199, train_X_column_200, train_X_column_201, train_X_column_202, train_X_column_203, train_X_column_204, train_X_column_205, train_X_column_206, train_X_column_207, train_X_column_208, train_X_column_209, train_X_column_210, train_X_column_211, train_X_column_212, train_X_column_213, train_X_column_214, train_X_column_215, train_X_column_216, train_X_column_217,
                               ], axis=1)
    

    train_X = np.concatenate((train_X_label_1, train_X_label_0), axis = 0)

    train_Y = np.concatenate((train_Y_label_1, train_Y_label_0), axis = 0)



    train_indices = np.random.permutation(len(train_X))



    train_X_random = train_X[train_indices]
    train_Y_random = train_Y[train_indices]



    return  train_X_random, train_Y_random



def train_one_model(train_num, train_func, repeat_num, file_name_1, file_name_2):

    



    train_num_record = int(math.log2(train_num))

    train_X, train_Y = gen_train_or_val_data(train_num, file_name_1)

    val_X, val_Y = gen_train_or_val_data(train_num, file_name_2)


    num_epochs = 100
    bs = 512

    input_len = len(train_X[0])
    net =  train_func(input_len = input_len)

    net.compile(optimizer='adam',loss='mse',metrics=['accuracy', TPRMetric(), TNRMetric()])


    import time
    t = int(time.time())
    check = ModelCheckpoint(wdir + f'{model_name}_train_data_{train_num_record}_{repeat_num}_{t}'  + '.h5', monitor='val_loss', save_best_only=True)
    

    lr = LearningRateScheduler(cyclic_lr(10,0.002, 0.0001))
    h = net.fit(train_X,train_Y,epochs=num_epochs,batch_size=bs,validation_data=(val_X, val_Y), callbacks=[lr,check])
    print(h.history.keys())


    best_val_acc_epoch = np.argmax(h.history['val_accuracy'])

    best_train_acc = h.history['accuracy'][best_val_acc_epoch]  
    best_train_tpr = h.history['tpr'][best_val_acc_epoch]  
    best_train_tnr = h.history['tnr'][best_val_acc_epoch] 

    best_val_acc = h.history['val_accuracy'][best_val_acc_epoch]
    best_val_tpr = h.history['val_tpr'][best_val_acc_epoch]
    best_val_tnr = h.history['val_tnr'][best_val_acc_epoch]


    write = open(wdir + model_name + "_record.txt",'a')
    
    write.write(f"neural_model: {train_func.__name__}, repeat_num : {repeat_num}, train_num: 2^ {train_num_record}\n")


    write.write("Best valid accuracy: "  + str(best_val_acc) + " ; ")
    write.write("Best valid tpr: " + str(best_val_tpr) + " ; ")
    write.write("Best valid tnr: " + str(best_val_tnr) + " ; ")
    write.write("\n")

    write.write("Best train accuracy: " + str(best_train_acc) + " ; ")
    write.write("Best train tpr: " + str(best_train_tpr) + " ; ")
    write.write("Best train tnr: " + str(best_train_tnr) + " ; ")
    write.write("\n")
    write.write("\n")
    write.close()



if __name__ == "__main__":

    

    start_time = time.time()


    file_name_1 = "cipher_txt_npy/" + ""

    file_name_2 = "cipher_txt_npy/" + ""
 

   
    for train_num_up in [20]:
        train_num = pow(2, train_num_up)
        for repeat_num in range(1):
            train_one_model(train_num, train_func, repeat_num, file_name_1, file_name_2)


    end_time = time.time()







