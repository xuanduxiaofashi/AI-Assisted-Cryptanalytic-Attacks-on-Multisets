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
from neural_model import  three_layer_mlp
import time

model_name = "three_layer_mlp"

train_func = three_layer_mlp

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






def gen_train_or_val_data(train_num,  load_file_name):



    train_num_label_1 = train_num // 2


    train_X_label_0 = np.random.choice([0, 1], size=(train_num_label_1  , 16 , 32))

    train_X_cell_2 = train_X_label_0[:, :, 0: 4];train_X_cell_5 = train_X_label_0[:, :, 4: 8]
    train_X_cell_6 = train_X_label_0[:, :, 8: 12];train_X_cell_7 = train_X_label_0[:, :, 12: 16];train_X_cell_9 = train_X_label_0[:, :, 16: 20]
    train_X_cell_10 = train_X_label_0[:, :, 16: 20]; train_X_cell_13 = train_X_label_0[:, :, 16: 20]; train_X_cell_14 = train_X_label_0[:, :, 16: 20]

    x_3_6r_2 = train_X_cell_2[:, :, 0];x_2_6r_2 = train_X_cell_2[:, :, 1];x_1_6r_2 = train_X_cell_2[:, :, 2];x_0_6r_2 = train_X_cell_2[:, :, 3]
    x_3_6r_6 = train_X_cell_6[:, :, 0];x_2_6r_6 = train_X_cell_6[:, :, 1];x_1_6r_6 = train_X_cell_6[:, :, 2];x_0_6r_6 = train_X_cell_6[:, :, 3]
    x_3_6r_7 = train_X_cell_7[:, :, 0];x_2_6r_7 = train_X_cell_7[:, :, 1];x_1_6r_7 = train_X_cell_7[:, :, 2];x_0_6r_7 = train_X_cell_7[:, :, 3]
    x_3_6r_10 = train_X_cell_10[:, :, 0];x_2_6r_10 = train_X_cell_10[:, :, 1];x_1_6r_10 = train_X_cell_10[:, :, 2];x_0_6r_10 = train_X_cell_10[:, :, 3]
    x_3_6r_14 = train_X_cell_14[:, :, 0];x_2_6r_14 = train_X_cell_14[:, :, 1];x_1_6r_14 = train_X_cell_14[:, :, 2];x_0_6r_14 = train_X_cell_14[:, :, 3]



    train_X_cell_5_xor  = np.bitwise_xor.reduce(train_X_cell_5,  axis=1).reshape(-1, 1, 4);train_X_cell_9_xor = np.bitwise_xor.reduce(train_X_cell_9, axis=1).reshape(-1, 1, 4)
    train_X_cell_13_xor = np.bitwise_xor.reduce(train_X_cell_13, axis=1).reshape(-1, 1, 4);train_X_cell_6_xor  = np.bitwise_xor.reduce(train_X_cell_6,  axis=1).reshape(-1, 1, 4)
    train_X_cell_10_xor = np.bitwise_xor.reduce(train_X_cell_10, axis=1).reshape(-1, 1, 4);train_X_cell_14_xor = np.bitwise_xor.reduce(train_X_cell_14, axis=1).reshape(-1, 1, 4)

    train_X_xor_1 = train_X_cell_5_xor ^ train_X_cell_9_xor ^ train_X_cell_13_xor
    train_X_xor_2 = train_X_cell_6_xor ^ train_X_cell_10_xor ^ train_X_cell_14_xor
    train_X_column_integral_1 = train_X_xor_1[:, :, 0]
    train_X_column_integral_2 = train_X_xor_1[:, :, 2]
    train_X_column_integral_3 = train_X_xor_1[:, :, 0] ^ train_X_xor_1[:, :, 2]

    train_X_column_integral_4 = train_X_xor_2[:, :, 0]
    train_X_column_integral_5 = train_X_xor_2[:, :, 2]
    train_X_column_integral_6 = train_X_xor_2[:, :, 0] ^ train_X_xor_2[:, :, 2]


    

    train_X_label_0 = np.concatenate([  train_X_column_integral_1, train_X_column_integral_2, train_X_column_integral_3, 
                                      train_X_column_integral_4, train_X_column_integral_5, train_X_column_integral_6, 
                        ], axis=1)
    

    train_Y_label_1 = np.ones((train_num_label_1, 1)) 
    train_Y_label_0 = np.zeros((train_num_label_1, 1))

    train_X_label_1 = np.load(load_file_name)


    train_X = np.concatenate((train_X_label_1, train_X_label_0), axis = 0)



    train_Y = np.concatenate((train_Y_label_1, train_Y_label_0), axis = 0)


    train_indices = np.random.permutation(len(train_X))


    train_X_random = train_X[train_indices]
    train_Y_random = train_Y[train_indices]
    

    return  train_X_random, train_Y_random


def train_one_model(train_num,  train_func, repeat_num, file_name_1, file_name_2):
   
    train_num_record = int(math.log2(train_num))
    
    train_X, train_Y = gen_train_or_val_data(train_num, file_name_1)

    val_X, val_Y = gen_train_or_val_data(train_num,  file_name_2)


    num_epochs = 30
    bs = 200

    net =  train_func(6)

    net.compile(optimizer='adam',loss='mse',metrics=['accuracy', TPRMetric(), TNRMetric()])

    
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
 
  
    for train_num_up in [19]:
        train_num = pow(2, train_num_up)
        for repeat_num in range(1):
            train_one_model(train_num,  train_func, repeat_num, file_name_1, file_name_2)


    end_time = time.time()

  

