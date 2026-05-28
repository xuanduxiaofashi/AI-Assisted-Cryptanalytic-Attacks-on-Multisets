from keras.layers import Dense, Conv1D, Input, Reshape, Permute, Add, Flatten, BatchNormalization, Activation
from keras.regularizers import l2
from keras.models import Model
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Reshape, Conv1D, BatchNormalization
from tensorflow.keras.layers import Activation, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.regularizers import l2


def cnn_model(input_len=4096, num_outputs=1, reg_param=1e-4, final_activation='sigmoid'):
    
    inp = Input(shape=(input_len,), name='input_bits')
    x = Reshape((input_len, 1))(inp)


    x = Conv1D(filters=64, kernel_size=16, strides=16, padding='valid',
               kernel_regularizer=l2(reg_param))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)


    x = Conv1D(filters=64, kernel_size=3, padding='same',
               kernel_regularizer=l2(reg_param))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = GlobalMaxPooling1D()(x)

    x = Dense(256, kernel_regularizer=l2(reg_param))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(128, kernel_regularizer=l2(reg_param))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)


    out = Dense(num_outputs, activation=final_activation,
                kernel_regularizer=l2(reg_param))(x)

    model = Model(inputs=inp, outputs=out)
    return model
  


