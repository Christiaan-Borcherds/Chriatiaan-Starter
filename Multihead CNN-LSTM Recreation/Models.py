import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import sklearn.metrics
from tensorflow.keras.utils import plot_model


#  input_shape == input_shape of each head (sensor data)
#  Here, input shape is of the form (128,3) for each head
#  seq == sequence no. of head
#  no activations are applied after any layer ( => identity )

def cnnhead_(input_shape, seq):
    input_layer = keras.Input(shape=input_shape, name='title_' + str(seq))

    cnn = layers.Conv1D(24, 2, 1, "same", name='Conv1D_' + str(seq) + '_1')(input_layer)
    cnn = layers.LayerNormalization(name='layernorm_' + str(seq) + '_1')(cnn)
    cnn = layers.Dropout(rate=0.5, name='dropout_' + str(seq) + '_1')(cnn)

    cnn = layers.Conv1D(48, 2, 1, "same", name='Conv1D_' + str(seq) + '_2')(cnn)
    cnn = layers.LayerNormalization(name='layernorm_' + str(seq) + '_2')(cnn)
    cnn = layers.Dropout(rate=0.5, name='dropout_' + str(seq) + '_2')(cnn)

    cnn = layers.Conv1D(48, 2, 1, "same", name='Conv1D_' + str(seq) + '_3')(cnn)
    cnn = layers.LayerNormalization(name='layernorm_' + str(seq) + '_3')(cnn)

    return input_layer, cnn


# Concatenates last layer of each cnn head together in third dimension (channels)

def concatenate_(heads):
    final_layers = []
    for i,j in heads:
        final_layers.append(j)
    return layers.concatenate(final_layers,name = 'concatenate')



# builds last lstm block

def lstm_(input_, number_of_classes):
    x = layers.LSTM(4*number_of_classes,return_sequences = True,name = 'lstm_1')(input_)
    x = layers.LayerNormalization()(x)
    x = layers.LSTM(4*number_of_classes,return_sequences = True,name = 'lstm_2')(x)
    x = layers.LayerNormalization()(x)
    x = layers.LSTM(2*number_of_classes,name = 'lstm_3')(x)
    x = layers.LayerNormalization()(x)
    x = layers.Dense(2*number_of_classes,name = 'dense_1')(x)
    x = layers.Dense(number_of_classes,name = 'dense_2',activation = 'softmax')(x)
    return x


def model_(heads,dense_):
    return keras.Model([i for i,j in heads],dense_,name = 'model')


# xtrain is the form (row,window,sensor,axis)
# (None,128,2,3) here

def build_model(xtrain, ytrain):
    heads = []

    shape = xtrain.shape

    for i in range(len(xtrain[0][0])):
        heads.append(cnnhead_((shape[1], shape[3]), i + 1))

    x = concatenate_(heads)

    x = lstm_(x, ytrain.shape[1])

    model = model_(heads, x)

    return model


