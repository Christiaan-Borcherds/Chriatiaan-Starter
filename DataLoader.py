from config import *

import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import sklearn.metrics
from tensorflow.keras.utils import plot_model


# MAPPING
label_to_class = {
    1  : 'WALKING',
    2  : 'WALKING_UPSTAIRS',
    3  : 'WALKING_DOWNSTAIRS',
    4  : 'SITTING',
    5  : 'STANDING',
    6  : 'LAYING',
    7  : 'STAND_TO_SIT',
    8  : 'SIT_TO_STAND',
    9  : 'SIT_TO_LIE',
    10 : 'LIE_TO_SIT',
    11 : 'STAND_TO_LIE',
    12 : 'LIE_TO_STAND',
    np.nan : np.nan
}
class_to_label = {
    'WALKING' : 1,
    'WALKING_UPSTAIRS' : 2,
    'WALKING_DOWNSTAIRS' : 3,
    'SITTING' : 4,
    'STANDING' : 5,
    'LAYING' : 6,
    'STAND_TO_SIT' : 7,
    'SIT_TO_STAND' : 8,
    'SIT_TO_LIE' : 9,
    'LIE_TO_SIT' : 10,
    'STAND_TO_LIE' : 11,
    'LIE_TO_STAND' : 12,
    np.nan : np.nan
}

# Function to draw bar graph of classes corresponding to their frequencies in occurence
def draw_bar(ydata):
    print('Frequencies :- ', ydata.sum(axis=0))

    x = np.arange(1, len(ydata[0]) + 1, 1);
    y = ydata.sum(axis=0)

    plt.figure(figsize=(12.8, 3))
    plt.xlabel('Class Label', fontdict={'size': 15})
    plt.ylabel('Frequency', fontdict={'size': 15})
    bar = plt.bar(x, y)

    for idx, rect in enumerate(bar):
        plt.text(
            rect.get_x() + rect.get_width() / 2.0,
            rect.get_height(), int(y[idx]),
            ha='center',
            va='bottom'
        )

    plt.show()

def draw_bar_sets(ytrain, ytest, yval):

    # Compute frequencies
    datasets = [ytrain, ytest, yval]
    titles = ['Train', 'Test', 'Validation']

    fig, axes = plt.subplots(
        3, 1,
        figsize=(14, 8),
        constrained_layout=True
    )

    for i, (ydata, ax) in enumerate(zip(datasets, axes)):
        freq = ydata.sum(axis=0)
        print(f'{titles[i]} Frequencies :- ', freq)

        x = np.arange(1, len(freq) + 1)

        bars = ax.bar(x, freq)

        ax.set_title(f'{titles[i]} Set')
        ax.set_ylabel('Frequency')
        ax.set_xticks(x)

        # Only bottom plot gets x-label to avoid clutter
        if i == 2:
            ax.set_xlabel('Class Label')

        # Add value labels on bars
        for idx, rect in enumerate(bars):
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                rect.get_height(),
                int(freq[idx]),
                ha='center',
                va='bottom',
                fontsize=8
            )

        ax.grid(axis='y', alpha=0.3)

    fig.suptitle('Class Distribution (Train / Test / Validation)', fontsize=14)

    plt.show()


# Function to draw xtrain time_series sensor data for first  instance of activity_label(class)
# x == [ 0th milisecond data , 20th ms data , 40th ms, ...] for 50 Hz rate i.e. 20 ms for each timestamp
# length = window_size
# row stores the index of first data_point that belongs to class == activity_label

def draw_wave(xdata, ydata, activity_label):
    # row = 0
    # while (ydata[row].argmax() + 1 != activity_label): row = row + 1;
    #
    # length = xdata.shape[1]
    # sensor = xdata.shape[2]
    # channel = xdata.shape[3]
    #
    # x = np.linspace(0, (20) * (length - 1) / 1000, length)
    #
    # plt.figure(figsize=(12.8, 2))
    # plt.plot(x, xdata[row, :, 0, 0])
    # plt.plot(x, xdata[row, :, 0, 1])
    # plt.plot(x, xdata[row, :, 0, 2])
    # plt.legend(['acc_x', 'acc_y', 'acc_z'])
    # plt.show()
    #
    # plt.figure(figsize=(12.8, 2))
    # plt.plot(x, xdata[row, :, 1, 0])
    # plt.plot(x, xdata[row, :, 1, 1])
    # plt.plot(x, xdata[row, :, 1, 2])
    # plt.legend(['gyro_x', 'gyro_y', 'gyro_z'])
    # plt.xlabel('Time in seconds :- ( Instance of ' + label_to_class[activity_label] + ' data )', fontdict={'size': 15})
    # plt.show()
    row = 0
    while ydata[row].argmax() + 1 != activity_label:
        row += 1

    length = xdata.shape[1]
    x = np.linspace(0, 20 * (length - 1) / 1000, length)

    fig, axes = plt.subplots(
        2, 1,
        figsize=(14, 6),
        sharex=True,
        constrained_layout=True
    )

    # Accelerometer subplot
    axes[0].plot(x, xdata[row, :, 0, 0], label='acc_x')
    axes[0].plot(x, xdata[row, :, 0, 1], label='acc_y')
    axes[0].plot(x, xdata[row, :, 0, 2], label='acc_z')
    axes[0].set_title('Accelerometer')
    axes[0].set_ylabel('Amplitude')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)

    # Gyroscope subplot
    axes[1].plot(x, xdata[row, :, 1, 0], label='gyro_x')
    axes[1].plot(x, xdata[row, :, 1, 1], label='gyro_y')
    axes[1].plot(x, xdata[row, :, 1, 2], label='gyro_z')
    axes[1].set_title('Gyroscope')
    axes[1].set_xlabel(f"Time in seconds (Instance of {label_to_class[activity_label]} data)")
    axes[1].set_ylabel('Amplitude')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"Waveform for Activity: {label_to_class[activity_label]}", fontsize=14)
    plt.show()



# (i).   Removing data-points where y and x values is null

# Other methods can be
# ffill (forward fill) => fills using forward points
# bfill (backward fill) => using backward points
# interpolate

def remove_null(xdata, ydata):
    xdata = xdata[np.where(np.isfinite(ydata))]
    ydata = ydata[np.where(np.isfinite(ydata))]
    ydata = ydata[np.where(np.isfinite(xdata).all(axis=1).all(axis=1).all(axis=1))]
    xdata = xdata[np.where(np.isfinite(xdata).all(axis=1).all(axis=1).all(axis=1))]

    return xdata, ydata


# normalize xdata using sklearn.preprocessing.StandardScaler and returns
# scaler object to use it furthur for testing data

# Each axis of each sensor has different min, max, I scaled according to them seperately
# Initial shape == (None,128,2,3)
# changed to (None , 6) :-
# reshape to (None,128,6) -> swapaxis(0,2) -> reshape(6,-1) -> transpose
# Fit scaler OR transform according to scaler

# Reverse above process to get back oiginal data
# transpose -> reshape(6,128,None) -> swapaxes(0,2) -> reshape(None,128,2,3)

def get_scaler(xdata):
    row = xdata.shape[0]
    timestamp = xdata.shape[1]
    sensor = xdata.shape[2]
    axis = xdata.shape[3]

    scaler = MinMaxScaler(feature_range=(-1, 1))
    xdata = xdata.reshape(row, timestamp, sensor * axis)
    xdata = np.swapaxes(xdata, 0, 2).reshape(sensor * axis, -1).T
    scaler.fit(xdata)
    return scaler


def scale_data(xdata, scaler):
    row = xdata.shape[0]
    timestamp = xdata.shape[1]
    sensor = xdata.shape[2]
    axis = xdata.shape[3]

    xdata = xdata.reshape(row, timestamp, sensor * axis)
    xdata = np.swapaxes(xdata, 0, 2).reshape(sensor * axis, -1).T
    xdata = scaler.transform(xdata)
    xdata = xdata.T.reshape(sensor * axis, timestamp, row)
    xdata = np.swapaxes(xdata, 0, 2).reshape(row, timestamp, sensor, axis)

    return xdata


# takes in location, exp no., user no., start and end(end point is excluded from reading i.e lastpoint+1) point
# ,overlap array, and returns xdata and ydata

def create_windows(location, exp, user, start, end, activity, length, overlap):
    acc_file = location + '/acc_exp' + str(exp).zfill(2) + '_user' + str(user).zfill(2) + '.txt'
    gyro_file = location + '/gyro_exp' + str(exp).zfill(2) + '_user' + str(user).zfill(2) + '.txt'

    acc_data = np.loadtxt(acc_file)
    gyro_data = np.loadtxt(gyro_file)

    xtrain = []
    ytrain = []

    while (start + length <= end):

        stop = start + length
        window = []

        while start != stop:
            window.append([acc_data[start], gyro_data[start]])
            start += 1

        xtrain.append(window)
        ytrain.append(activity)

        start = stop - overlap[activity - 1]

    return xtrain, ytrain


# location == location of file
# lenght == lenght of window
# overlap == array of overlaps of size == number of unique activities
# overlap depends on activity so as to extract more data from a particular class if needed


# (i).   Loading labels.txt as labels
# (ii).  Iterating in labels and calling create_windows on acceleration file, extending returned data in xtrain, ytrain
# (iii). Iterating in labels and calling create_windows on gyroscope file, extending returned data in xtrain, ytrain

def prepare_data(location, length=128, overlap=[64] * 12):
    xdata = []
    ydata = []

    labels = np.loadtxt(location + '/labels.txt', dtype='uint32')

    count =0
    for exp, user, activity, start, end in labels:
        count +=1
        xtemp, ytemp = create_windows(location, exp, user, start, end + 1, activity, length, overlap)
        xdata.extend(xtemp)
        ydata.extend(ytemp)
        print(f'Window no: {count+1}')

    return np.array(xdata), np.array(ydata)


# (i). Finds max element index sets its 1 and sets remaining 0
#      for each row

def to_categorical(ydata):
    for i in range(len(ydata)):
        j = ydata[i].argmax()
        for k in range(len(ydata[i])):
            ydata[i][k] = (k == j)
    return ydata


# (i).  OneHotEncoding ydata
# (ii). Converting sparsh matrix ydata into dense form and then matrix into numpy array

def one_hot_encoded(ydata):
    ydata = OneHotEncoder().fit_transform(ydata.reshape(len(ydata),1))
    ydata = np.asarray(ydata.todense())
    return ydata



# # DATA PREP
#
# # Preparing data, xtrain, ytrain
# # Last six classes [7 to 12] has very less weightage in data since they are extra classes added
# # , made from original six classes
# # so, I took more overlapping in them to get slightly more data
#
# xtrain,ytrain = prepare_data('/home/christiaan/Documents/MUST/Starter Project/Datasets/HAPT/RawData',128,[64,64,64,64,64,64,120,120,120,120,120,120])
#
# # Saving xdata and ydata temporarily for using again if needed
#
# with open('Data/xdata','wb') as f:
#     pickle.dump(xtrain,f)
# with open('Data/ydata','wb') as f:
#     pickle.dump(ytrain,f)

# HANDLING missing Data and norm

# xtrain,ytrain = remove_null(xtrain,ytrain)
#
# # splitting into training (70%) testing (15%) and validation (15%) set
# xtrain,xtest,ytrain,ytest = train_test_split(xtrain,ytrain,test_size = 0.3)
# xtest,xval,ytest,yval = train_test_split(xtest,ytest,test_size = 0.5)
#
# print(f'{xtrain.shape} \n {ytrain.shape} \n {xtest.shape} \n {ytest.shape} \n {xval.shape} \n {yval.shape} ')
#
# # (i).  Get scaler object
# # (ii). Scaling xtrain and xtest
#
# scaler = get_scaler(xtrain)
# xtrain = scale_data(xtrain,scaler)
# xtest  = scale_data(xtest,scaler)
# xval   = scale_data(xval,scaler)

# One hot encoding y values

# Note that when using Cross Entropy Loss, the labels should not be one-hot encoded!!
# labels = [0, 1, 2, ..., 11]
# shape = (N,)
# dtype = torch.long

# need to adjust label index! - pytorch expects [0,1...11]
# ytrain = ytrain - 1

# example:
# outputs = model(x)   # shape: (32, 12)
# labels  = tensor([2, 0, 5, 1, ...])  # shape: (32,)
# loss = criterion(outputs, labels)

# no softmax with  CrossEntropyLoss

# ytrain = one_hot_encoded(ytrain)
# ytest = one_hot_encoded(ytest)
# yval = one_hot_encoded(yval)
#
# print(f'{xtrain.shape} \n {ytrain.shape} \n {xtest.shape} \n {ytest.shape} \n {xval.shape} \n {yval.shape}')

# TRAIN
# (9355, 128, 2, 3) -> Num samples (windows) , window length, sensor type (gyro + acc) , axes (x, y, z)
# (9355, 12)

# TEST
# (2005, 128, 2, 3)
# (2005, 12)

# VAL
# (2005, 128, 2, 3)
# (2005, 12)

