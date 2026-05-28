from keras.layers import Dense, Conv1D, Input, Reshape, Permute, Add, Flatten, BatchNormalization, Activation
from keras.regularizers import l2
from keras.models import Model
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Reshape, Conv1D, BatchNormalization
from tensorflow.keras.layers import Activation, GlobalMaxPooling1D, Dense, Dropout, MaxPooling1D
from tensorflow.keras.regularizers import l2


def new_transformer_model_from_64(input_shape=(120, 64), num_heads=4):
    inputs = tf.keras.Input(shape=input_shape)  
    x = inputs
    attn_output = layers.MultiHeadAttention(num_heads=num_heads, key_dim=64)(x, x)
    x = layers.Add()([x, attn_output])
    x = layers.LayerNormalization()(x)

    ff = layers.Dense(128, activation='relu')(x)
    ff = layers.Dense(64)(ff)
    x = layers.Add()([x, ff])
    x = layers.LayerNormalization()(x)

    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs, outputs)
    return model

