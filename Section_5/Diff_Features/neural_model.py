from keras.layers import Dense, Conv1D, Input, Reshape, Permute, Add, Flatten, BatchNormalization, Activation
from keras.regularizers import l2
from keras.models import Model
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Reshape, Conv1D, BatchNormalization
from tensorflow.keras.layers import Activation, GlobalMaxPooling1D, Dense, Dropout, MaxPooling1D
from tensorflow.keras.regularizers import l2


def new_model(input_len= 64):
  model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(input_len,)),
    layers.BatchNormalization(),


    layers.Dense(64, activation='relu'),
    layers.BatchNormalization(),


    layers.Dense(32, activation='relu'),

    layers.Dense(1, activation='sigmoid')
  ])

  return model
