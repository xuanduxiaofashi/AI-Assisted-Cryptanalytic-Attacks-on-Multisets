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
import  time

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

    train_X_label_0 = np.random.choice([0, 1], size=(train_num_label_1, 16, 12))

    train_X_column_1 = train_X_label_0[:, :, 0] ^ train_X_label_0[:, :, 4] ^ train_X_label_0[:, :, 8]
    train_X_column_2 = train_X_label_0[:, :, 1] ^ train_X_label_0[:, :, 5] ^ train_X_label_0[:, :, 9]
    train_X_column_3 = train_X_label_0[:, :, 2] ^ train_X_label_0[:, :, 6] ^ train_X_label_0[:, :, 10]
    train_X_column_4 = train_X_label_0[:, :, 3] ^ train_X_label_0[:, :, 7] ^ train_X_label_0[:, :, 11]
    temp_1 = train_X_label_0[:, :, 0: 4] ^ train_X_label_0[:, :, 8: 12]
    temp_2 = train_X_label_0[:, :, 4: 8]

    

    x_3_6r_2 = train_X_label_0[:, :, 0]
    x_2_6r_2 = train_X_label_0[:, :, 1]
    x_1_6r_2 = train_X_label_0[:, :, 2]
    x_0_6r_2 = train_X_label_0[:, :, 3]

    x_3_6r_7 = temp_2[:, :, 0]
    x_2_6r_7 = temp_2[:, :, 1]
    x_1_6r_7 = temp_2[:, :, 2]
    x_0_6r_7 = temp_2[:, :, 3]

    x_3_6r_14 = train_X_label_0[:, :, 0 + 8]
    x_2_6r_14 = train_X_label_0[:, :, 1 + 8]
    x_1_6r_14 = train_X_label_0[:, :, 2 + 8]
    x_0_6r_14 = train_X_label_0[:, :, 3 + 8]

    train_X_column_5 = x_3_6r_2 ^  x_3_6r_7 ^ x_2_6r_7 ^  x_3_6r_14 
    train_X_column_6 = x_2_6r_2 ^  x_2_6r_7 ^ x_1_6r_7 ^  x_2_6r_14   
    train_X_column_7 = x_1_6r_2 ^  x_3_6r_7 ^ x_1_6r_7 ^  x_1_6r_14  
    train_X_column_8 = x_0_6r_2 ^  x_2_6r_7 ^ x_0_6r_7 ^  x_0_6r_14 
    train_X_column_9 = x_2_6r_2 ^ x_0_6r_2 ^ x_0_6r_7 ^ x_2_6r_14 ^  x_0_6r_14 
    train_X_column_10 = x_3_6r_2 ^ x_1_6r_2 ^ x_1_6r_7 ^ x_3_6r_14 ^  x_1_6r_14 
    train_X_column_11 = x_3_6r_2 ^ x_2_6r_2 ^ x_3_6r_7  ^ x_2_6r_7 ^ x_3_6r_14 ^  x_2_6r_14
    train_X_column_12 = x_3_6r_2 ^ x_0_6r_2 ^ x_3_6r_7  ^ x_0_6r_7 ^ x_3_6r_14 ^  x_0_6r_14 
    train_X_column_13 = x_2_6r_2 ^ x_1_6r_2 ^ x_2_6r_7  ^ x_1_6r_7 ^ x_2_6r_14 ^  x_1_6r_14 
    train_X_column_14 = x_1_6r_2 ^ x_0_6r_2 ^ x_1_6r_7  ^ x_0_6r_7 ^ x_1_6r_14 ^  x_0_6r_14
    train_X_column_15 = x_2_6r_2 ^ x_0_6r_2 ^ x_2_6r_7  ^ x_0_6r_7 ^ x_2_6r_14 ^  x_0_6r_14
    train_X_column_16 = x_3_6r_2 ^ x_1_6r_2 ^ x_0_6r_2 ^ x_3_6r_7  ^ x_1_6r_7  ^ x_0_6r_7 ^ x_3_6r_14 ^  x_1_6r_14 ^ x_0_6r_14 
    train_X_column_17 = x_3_6r_2 ^ x_2_6r_2 ^ x_1_6r_2  ^ x_0_6r_2 ^ x_3_6r_7 ^  x_2_6r_7   ^ x_1_6r_7  ^ x_0_6r_7 ^ x_3_6r_14 ^  x_2_6r_14 ^ x_1_6r_14 ^ x_0_6r_14 
    train_X_column_18 = x_3_6r_2 ^ x_2_6r_2 ^ x_1_6r_2  ^ x_0_6r_2 ^ x_3_6r_7  ^ x_3_6r_14 ^  x_2_6r_14 ^ x_1_6r_14 ^ x_0_6r_14 
    train_X_column_19 = x_3_6r_2 ^ x_2_6r_2 ^ x_1_6r_2  ^ x_3_6r_7 ^  x_2_6r_7 ^ x_1_6r_7   ^ x_3_6r_14 ^  x_2_6r_14 ^ x_1_6r_14 
    train_X_column_20 = x_3_6r_2 ^ x_2_6r_2 ^ x_0_6r_2   ^ x_3_6r_7   ^ x_0_6r_7 ^ x_3_6r_14 ^  x_2_6r_14  ^ x_0_6r_14 
    train_X_column_21 =  x_3_6r_2 ^ x_2_6r_2   ^ x_0_6r_2 ^ x_3_6r_7 ^  x_2_6r_7   ^ x_0_6r_7 ^ x_3_6r_14 ^  x_2_6r_14  ^ x_0_6r_14 
    train_X_column_22 = x_1_6r_2 ^  x_3_6r_7 ^  x_1_6r_7 ^ x_0_6r_7 ^  x_1_6r_14
    train_X_column_23 = x_1_6r_2 ^  x_3_6r_7 ^  x_2_6r_7  ^  x_1_6r_7 ^ x_0_6r_7 ^  x_1_6r_14
    train_X_column_24 = x_2_6r_2 ^ x_1_6r_2  ^ x_0_6r_2  ^  x_2_6r_7   ^ x_1_6r_7  ^ x_0_6r_7 ^   x_2_6r_14 ^   x_1_6r_14 ^   x_0_6r_14
    train_X_column_25 = x_3_6r_2   ^ x_0_6r_2 ^ x_3_6r_7 ^  x_2_6r_7     ^ x_0_6r_7 ^ x_3_6r_14 ^ x_0_6r_14 
    train_X_column_26 = x_3_6r_2  ^ x_1_6r_2   ^ x_3_6r_7    ^ x_1_6r_7   ^ x_3_6r_14 ^ x_1_6r_14
    train_X_column_27 = x_3_6r_2  ^ x_1_6r_2  ^ x_0_6r_2 ^ x_3_6r_7 ^  x_2_6r_7    ^ x_3_6r_14  ^ x_1_6r_14 ^ x_0_6r_14
    train_X_column_28 = x_3_6r_2 ^ x_2_6r_2 ^ x_3_6r_7  ^ x_3_6r_14 ^  x_2_6r_14
    train_X_column_29 = x_3_6r_2 ^ x_2_6r_2 ^ x_1_6r_2  ^ x_0_6r_2 ^  x_2_6r_7    ^ x_3_6r_14 ^  x_2_6r_14 ^ x_1_6r_14 ^ x_0_6r_14 
    train_X_column_30 = x_3_6r_2   ^ x_0_6r_2 ^  x_0_6r_7    ^ x_3_6r_14 ^ x_0_6r_14
    train_X_column_31 = x_2_6r_2   ^ x_1_6r_2 ^  x_2_6r_7    ^ x_2_6r_14 ^ x_1_6r_14 
    train_X_column_32 = x_1_6r_2   ^ x_0_6r_2 ^  x_1_6r_7    ^ x_1_6r_14 ^ x_0_6r_14 
    train_X_column_33 =  x_1_6r_2   ^ x_1_6r_7 ^  x_0_6r_7    ^ x_1_6r_14   
    train_X_column_34 = x_0_6r_2   ^ x_3_6r_7 ^  x_0_6r_7    ^ x_0_6r_14  
    train_X_column_35 =  x_3_6r_2   ^ x_2_6r_2 ^  x_1_6r_2  ^ x_2_6r_7 ^ x_3_6r_14   ^ x_2_6r_14 ^  x_1_6r_14 
    train_X_column_36 = x_3_6r_2   ^ x_2_6r_2 ^  x_0_6r_2  ^ x_0_6r_7 ^ x_3_6r_14   ^ x_2_6r_14 ^  x_0_6r_14 
    train_X_column_37 = x_3_6r_2   ^ x_2_6r_2 ^  x_1_6r_7  ^ x_0_6r_7   ^ x_3_6r_14 ^  x_2_6r_14 
    train_X_column_38 = x_3_6r_2   ^ x_1_6r_2 ^  x_0_6r_2  ^ x_1_6r_7   ^ x_3_6r_14 ^  x_1_6r_14  ^  x_0_6r_14


    
    train_Y_label_1 = np.ones((train_num_label_1, 1)) 
    train_Y_label_0 = np.zeros((train_num_label_1, 1))



    train_X_label_0 = np.concatenate([train_X_column_1, train_X_column_2, train_X_column_3, train_X_column_4, train_X_column_5, train_X_column_6, train_X_column_7,
                               train_X_column_8, train_X_column_9, train_X_column_10, train_X_column_11,   train_X_column_12, 
                               train_X_column_13, train_X_column_14, train_X_column_15, train_X_column_16, train_X_column_17, train_X_column_18, 
                               train_X_column_19, train_X_column_20, train_X_column_21, train_X_column_22,train_X_column_23, train_X_column_24,
                               train_X_column_25,train_X_column_26, train_X_column_27, train_X_column_28, train_X_column_29,
                               train_X_column_30,train_X_column_31, train_X_column_32, train_X_column_33, train_X_column_34,
                               train_X_column_35,train_X_column_36, train_X_column_37, train_X_column_38,
                               ], axis=1)

    train_X = np.concatenate((train_X_label_1, train_X_label_0), axis = 0)

    train_Y = np.concatenate((train_Y_label_1, train_Y_label_0), axis = 0)



    train_indices = np.random.permutation(len(train_X))


    train_X_random = train_X[train_indices]
    train_Y_random = train_Y[train_indices]


    return  train_X_random, train_Y_random

def train_one_model(train_num,  train_func, repeat_num, file_name_1, file_name_2):

    train_num_record = int(math.log2(train_num))

    train_X, train_Y = gen_train_or_val_data(train_num, file_name_1)

    val_X, val_Y = gen_train_or_val_data(train_num, file_name_2)



    num_epochs = 30
    bs = 256

    input_len = len(train_X[0])
    net =  train_func(input_len = input_len)

    net.compile(optimizer='adam',loss='mse',metrics=['accuracy', TPRMetric(), TNRMetric()])


    check = ModelCheckpoint(wdir + f'{model_name}_train_data_{train_num_record}_{repeat_num}'  + '.h5', monitor='val_loss', save_best_only=True)
    

    lr = LearningRateScheduler(cyclic_lr(10,0.002, 0.0001))
    h = net.fit(train_X,train_Y,epochs=num_epochs,batch_size=bs,validation_data=(val_X, val_Y), callbacks=[lr,check])

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

    

    for train_num_up in [19]:
        train_num = pow(2, train_num_up)
        for repeat_num in range(1):
            train_one_model(train_num,  train_func, repeat_num, file_name_1, file_name_2)


    end_time = time.time()







