import numpy as np
import sys
import math
import os
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.layers import Dense, Conv1D, Input, Reshape, Permute, Add, Flatten, BatchNormalization, Activation
from keras.regularizers import l2
from keras.models import Model
from tensorflow.keras import backend as K
import random
import tensorflow as tf
from neural_model import cnn_model
from keras.utils import get_custom_objects
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  

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



model_name = "cnn_model"

train_func = cnn_model


def process_input_data(input_array, guess_key):

    input_array_reshape = input_array.reshape(-1, 16, 12).copy()

    guess_key_2 = (guess_key >> 4) & 0xF  
    guess_key_7 = guess_key& 0xF          
    array1 = input_array_reshape[:, :, 0: 4]
    array2 = input_array_reshape[:, :, 4: 8]
    array3 = input_array_reshape[:, :, 8: 12]
    sbox_inv = np.array([3, 4,  6, 8, 12, 10, 1, 14, 9, 2, 5, 7, 0, 11, 13, 15])


    array1_value = (8 * array1[:, :, 0] + 4 * array1[:, :, 1] + 2 * array1[:, :, 2]+ 1 * array1[:, :, 3]).astype(int)
    array2_value = (8 * array2[:, :, 0] + 4 * array2[:, :, 1] + 2 * array2[:, :, 2]+ 1 * array2[:, :, 3]).astype(int)
    array3_value = (8 * array3[:, :, 0] + 4 * array3[:, :, 1] + 2 * array3[:, :, 2]+ 1 * array3[:, :, 3]).astype(int)

    array1_value = sbox_inv[array1_value ^ guess_key_2]
    array2_value = sbox_inv[array2_value ^ guess_key_7]
    array3_value = sbox_inv[array3_value]
    
    array1_value_columns = ((array1_value[..., None] >> np.arange(3, -1, -1)) & 1).astype(np.uint8)
    array2_value_columns = ((array2_value[..., None] >> np.arange(3, -1, -1)) & 1).astype(np.uint8)
    array3_value_columns = ((array3_value[..., None] >> np.arange(3, -1, -1)) & 1).astype(np.uint8)


    merged = np.concatenate([array1_value_columns, array2_value_columns, array3_value_columns], axis=2)

    input_array = merged
    train_X_column_1 = input_array[:, :, 0] ^ input_array[:, :, 4] ^ input_array[:, :, 8]
    train_X_column_2 = input_array[:, :, 1] ^ input_array[:, :, 5] ^ input_array[:, :, 9]

    temp_1 = input_array[:, :, 0: 4] ^ input_array[:, :, 8: 12]
    temp_2 = input_array[:, :, 4: 8]
    
    temp_1_3 = temp_1[:, :, 0];temp_1_2 = temp_1[:, :, 1];temp_1_1 = temp_1[:, :, 2];temp_1_0 = temp_1[:, :, 3]

    cp_x_3 = (temp_1_2 ^ temp_1_3 ^ temp_1_1&temp_1_0 ^temp_1_3&temp_1_0 ^ temp_1_3&temp_1_1 ^ temp_1_2&temp_1_1 ^ temp_1_3 & temp_1_2&temp_1_1)
    cp_x_2 = ( temp_1_0 ^ temp_1_1 ^ temp_1_2 ^ temp_1_3 & temp_1_0 ^ temp_1_3 & temp_1_2)
    cp_x_1 = temp_1_0 ^ temp_1_3 & temp_1_2 ^ temp_1_3 ^ temp_1_2
    cp_x_0 = temp_1_0 ^  temp_1_1 ^  temp_1_2 ^  temp_1_1 &  temp_1_0 ^  temp_1_2 &  temp_1_0 ^  temp_1_3 &  temp_1_1 ^  temp_1_3 &  temp_1_2 &  temp_1_0 ^  temp_1_3 &  temp_1_2 &  temp_1_1
    

    x_3_6r_2 = input_array[:, :, 0];x_2_6r_2 = input_array[:, :, 1];x_1_6r_2 = input_array[:, :, 2];x_0_6r_2 = input_array[:, :, 3]

    x_3_6r_7 = temp_2[:, :, 0];x_2_6r_7 = temp_2[:, :, 1];x_1_6r_7 = temp_2[:, :, 2];x_0_6r_7 = temp_2[:, :, 3]

    x_3_6r_14 = input_array[:, :, 0 + 8];x_2_6r_14 = input_array[:, :, 1 + 8];x_1_6r_14 = input_array[:, :, 2 + 8];x_0_6r_14 = input_array[:, :, 3 + 8]

    train_X_column_6 = x_3_6r_7 ^cp_x_0
    train_X_column_7 = x_3_6r_7 ^ cp_x_0 ^ cp_x_2
    train_X_column_8 = x_3_6r_7 ^ cp_x_0 ^ cp_x_3
    train_X_column_9 = x_3_6r_7 ^ cp_x_0 ^ cp_x_2 ^ cp_x_3
    train_X_column_10 = x_2_6r_7 ^cp_x_3 
    train_X_column_11 = x_2_6r_7 ^ cp_x_3 ^ cp_x_1
    train_X_column_12 = x_2_6r_7 ^ cp_x_3 ^ cp_x_2
    train_X_column_13 = x_2_6r_7 ^ cp_x_3 ^ cp_x_2^ cp_x_1
    
    train_X_column_20 = x_3_6r_2 ^ x_2_6r_2 ^ x_3_6r_7  ^ x_2_6r_7 ^ x_3_6r_14 ^  x_2_6r_14
    train_X_column_25 = x_3_6r_2 ^ x_1_6r_2 ^ x_0_6r_2 ^ x_3_6r_7  ^ x_1_6r_7  ^ x_0_6r_7 ^ x_3_6r_14 ^  x_1_6r_14 ^ x_0_6r_14 
    
    process_array = np.concatenate([train_X_column_1, train_X_column_2, 
                                train_X_column_6, train_X_column_7, train_X_column_8, train_X_column_9, train_X_column_10, 
                                train_X_column_11, train_X_column_12, train_X_column_13, train_X_column_20,   train_X_column_25,  
                                ], axis=1)
    
    process_array = process_array.reshape((-1, 4 * 192))
    return process_array



if __name__ == "__main__":
    candidate_num =  16

    step_num = 5
    all_credits = np.zeros((256, ))

    key_candidate_list = np.random.choice(256, size=candidate_num, replace=False)
    mean_table = np.load("average_list_diff.npy").flatten()
    std_table = np.load("std_list_diff.npy").flatten()
   
    print("Random array:", key_candidate_list)
    for i in range(step_num):

        all_credits_values_for_candidate = np.zeros((256, ))

        for key_candidate in key_candidate_list:

            file_name = "ciper_txt_gen_16/cipher_8r_gen_1.txt"

            X_val = np.loadtxt(file_name)
        
            process_X_val = process_input_data(X_val, key_candidate)
            
            model_name = "train_u_v_models/" + "cnn_model/" + "cnn_model.h5"
            
            model = tf.keras.models.load_model(model_name)

            neural_scores = model.predict(process_X_val,  verbose=0)

            neural_scores = np.mean(neural_scores)

            
            all_credits[key_candidate] = neural_scores

            for i in range(256): 
                index = i ^ key_candidate
                all_credits_values_for_candidate[i] = (all_credits_values_for_candidate[i] + 
                                                pow( (neural_scores - mean_table[index]) , 2))

            

        
        lowest_indices = np.argsort(all_credits_values_for_candidate)[:candidate_num]

        key_candidate_list = lowest_indices
        print(lowest_indices)
        binary_strings = [format(i, '08b') for i in lowest_indices]
        print(binary_strings)

        