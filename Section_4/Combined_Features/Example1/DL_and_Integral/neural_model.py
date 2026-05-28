from keras.layers import Dense, Conv1D, Input, Reshape, Permute, Add, Flatten, BatchNormalization, Activation
from keras.regularizers import l2
from keras.models import Model
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Reshape, Conv1D, BatchNormalization
from tensorflow.keras.layers import Activation, GlobalMaxPooling1D, Dense, Dropout, MaxPooling1D

from tensorflow.keras.regularizers import l2



def three_layer_mlp(input_len, num_outputs=1, reg_param=0.0001,final_activation='sigmoid'):
  inp = Input(shape=(input_len, ))
  d1, d2 , d3 = 40, 20, 40
  dense1 = Dense(d1,kernel_regularizer=l2(reg_param))(inp)
  dense1 = BatchNormalization()(dense1)
  dense1 = Activation('relu')(dense1)

  dense2 = Dense(d2, kernel_regularizer=l2(reg_param))(dense1)
  dense2 = BatchNormalization()(dense2)
  dense2 = Activation('relu')(dense2)

  dense3 = Dense(d3, kernel_regularizer=l2(reg_param))(dense2)
  dense3 = BatchNormalization()(dense3)
  dense3 = Activation('relu')(dense3)


  out = Dense(num_outputs, activation=final_activation, kernel_regularizer=l2(reg_param))(dense3)
  model = Model(inputs=inp, outputs=out)
  return(model)
