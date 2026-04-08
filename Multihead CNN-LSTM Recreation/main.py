import os
# print("LD_LIBRARY_PATH =", os.environ.get("LD_LIBRARY_PATH"))


import wandb
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"TensorFlow GPU detected: {gpus}")
    except RuntimeError as e:
        print(f"TensorFlow GPU setup error: {e}")
else:
    print("No TensorFlow GPU detected. Training will run on CPU.")

import DataLoader as DL
import Models
from config import *

from sklearn.model_selection import train_test_split
import pickle
from tensorflow.keras.utils import plot_model
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import sklearn.metrics
import seaborn as sb




ReloadFromRawData = False
VisualizeRawDataDistribution = False
TrainModel = True


if ReloadFromRawData:
    # DATA PREP

    # Preparing data, xtrain, ytrain
    # Last six classes [7 to 12] has very less weightage in data since they are extra classes added
    # , made from original six classes
    # so, I took more overlapping in them to get slightly more data

    xData,yData = DL.prepare_data(DatasetDir,128,[64,64,64,64,64,64,120,120,120,120,120,120])

    # Saving xdata and ydata temporarily for using again if needed

    with open('Data/xdata','wb') as f:
        pickle.dump(xData,f)
    with open('Data/ydata','wb') as f:
        pickle.dump(yData,f)

else:
    # To load temporarily saved prepared data
    with open('Data/xdata','rb') as f:
        xData = pickle.load(f)
    with open('Data/ydata','rb') as f:
        yData = pickle.load(f)


xData,yData = DL.remove_null(xData,yData)

# splitting into training (70%) testing (15%) and validation (15%) set
xtrain,xtest , ytrain,ytest = train_test_split(xData,yData,test_size = 0.3)
xtest,xval,ytest,yval = train_test_split(xtest,ytest,test_size = 0.5)

# print(f'{xtrain.shape} \n {ytrain.shape} \n {xtest.shape} \n {ytest.shape} \n {xval.shape} \n {yval.shape} ')

# (i).  Get scaler object
# (ii). Scaling xtrain and xtest

scaler = DL.get_scaler(xtrain)
xtrain = DL.scale_data(xtrain,scaler)
xtest  = DL.scale_data(xtest,scaler)
xval   = DL.scale_data(xval,scaler)


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

ytrain = DL.one_hot_encoded(ytrain)
ytest = DL.one_hot_encoded(ytest)
yval = DL.one_hot_encoded(yval)

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

if VisualizeRawDataDistribution:

    # Distrubution Visualisation
    DL.draw_bar_sets(ytrain, ytest, yval)

    # print('For training data :- ')
    # DL.draw_bar(ytrain)
    # print('For testing data :- ')
    # DL.draw_bar(ytest)
    # print('For validation data :- ')
    # DL.draw_bar(yval)

    # Visualizing sensors data for activities in train
    for i in range(12):
        DL.draw_wave(xtrain,ytrain,i+1)



if TrainModel:
    print('Training model')
    model = Models.build_model(xtrain, ytrain)
    # plot_model(model, "multiheaded.png", show_shapes=True, dpi=60)
    # model.summary()

    # Hyperparameters:
    EPOCHS_STAGE1 = 20
    EPOCHS_STAGE2 = 10
    BATCH_SIZE = 100
    run_count = 7

    # Create new run
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="christiaanborcherds-north-west-university",
        notes="Commit message for run",
        # Set the wandb project where this run will be logged.
        project="Starter-HAPT",
        name=f"Run{run_count}",
        # Track hyperparameters and run metadata.
        config={
            "architecture": "Multihead CNN & LSTM",
            "dataset": "HAPT",
            "epochs_stage1": EPOCHS_STAGE1,
            "epochs_stage2": EPOCHS_STAGE2,
            "batch_size": BATCH_SIZE,
        },
    )

    # run.watch(model, log="all", log_freq=5)  # optional: weights/gradients

    import time


    class progress_print(keras.callbacks.Callback):
        def on_train_begin(self, logs=None):
            if not hasattr(self, "global_epoch"):
                self.global_epoch = 0

        def on_epoch_begin(self, epoch, logs=None):
            self.start = time.time()

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}

            acc = logs.get('accuracy')
            loss = logs.get('loss')
            val_acc = logs.get('val_accuracy')
            val_loss = logs.get('val_loss')

            # if epoch < 8 or (epoch + 1) % 10 == 0:
            print(
                f'Epoch {epoch + 1}/{EPOCHS_STAGE1+EPOCHS_STAGE2} - Time: {time.time() - self.start:.2f}s\n'
                f'loss: {loss} - accuracy: {acc} - val_loss: {val_loss} - val_accuracy: {val_acc}\n'
            )
            # print(
            #     'Epoch {}/{} - Time taken : {}s\nloss: {} - accuracy: {} - val_loss: {} - val_accuracy: {}\n'
            #     .format(epoch + 1, EPOCHS_STAGE1+EPOCHS_STAGE2, time.time() - self.start, logs['loss'], logs['accuracy'],
            #             logs['val_loss'], logs['val_accuracy'])
            # )

            wandb.log({
                "loss": loss,
                "accuracy": acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc
            },step=self.global_epoch)

            self.global_epoch += 1




    # For 20 epochs

    lr_scheduler = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.001,
        decay_steps=int((xtrain.shape[0] + BATCH_SIZE) / BATCH_SIZE),
        decay_rate=0.99
    )

    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=lr_scheduler),
                  metrics=['accuracy'])

    ytrain = ytrain.astype('float32')
    yval = yval.astype('float32')

    common_callbacks = [progress_print()]

    history1 = model.fit(
        {'title_1': xtrain[:, :, 0, :],
         'title_2': xtrain[:, :, 1, :],
         },
        ytrain,
        epochs=EPOCHS_STAGE1,
        batch_size=BATCH_SIZE,
        validation_data=(
            {'title_1': xval[:, :, 0, :],
             'title_2': xval[:, :, 1, :]},
            yval
        ),
        verbose=0,
        callbacks=common_callbacks
        # initial_epoch = 0
    )

    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adagrad(), metrics=['accuracy'])

    # EPOCHS += 10

    history2 = model.fit(
        {'title_1': xtrain[:, :, 0, :], 'title_2': xtrain[:, :, 1, :],
         },
        ytrain,
        epochs=EPOCHS_STAGE1+EPOCHS_STAGE2,
        batch_size=BATCH_SIZE,
        validation_data=(
            {'title_1': xval[:, :, 0, :], 'title_2': xval[:, :, 1, :]},
            yval
        ),
        verbose=0,
        callbacks=common_callbacks,
        initial_epoch=EPOCHS_STAGE1
    )

    wandb.finish()

    plt.figure(figsize=(24, 6))

    # Visualizing training_loss and val_loss

    plt.subplot(1, 2, 1)
    plt.xlabel('Number of epochs')
    plt.grid(True, linewidth='0.5', linestyle='-.')
    plt.plot(history1.history['loss'] + history2.history['loss'], color='cyan')
    plt.plot(history1.history['val_loss'] + history2.history['val_loss'], color='orange')
    plt.legend(['training_loss', 'val_loss'])

    # Visualizing training_accuracy and val_accuracy

    plt.subplot(1, 2, 2)
    plt.xlabel('Number of epochs')
    plt.grid(True, linewidth='0.5', linestyle='-.')
    plt.plot(history1.history['accuracy'] + history2.history['accuracy'], color='cyan')
    plt.plot(history1.history['val_accuracy'] + history2.history['val_accuracy'], color='orange')
    plt.legend(['training_accuracy', 'val_accuracy'])

    plt.show()

    model.save_weights('Models/trained_weights.weights.h5')
    model.load_weights('Models/trained_weights.weights.h5')

    ytrain_pred = model.predict(
        {'title_1': xtrain[:, :, 0, :],
         'title_2': xtrain[:, :, 1, :],
         }
    )

    ytest_pred = model.predict(
        {'title_1': xtest[:, :, 0, :],
         'title_2': xtest[:, :, 1, :],
         }
    )

    # converts softmax ydata output into 0's and 1's

    ytrain_pred = DL.to_categorical(ytrain_pred)
    ytest_pred = DL.to_categorical(ytest_pred)

    train_cm = confusion_matrix(ytrain.argmax(axis=1), ytrain_pred.argmax(axis=1))
    test_cm = confusion_matrix(ytest.argmax(axis=1), ytest_pred.argmax(axis=1))

    sb.set(rc={'figure.figsize': (12.8, 9.6)})
    sb.heatmap(train_cm, annot=True, cmap='YlOrRd')
    plt.title('Train Confusion Matrix')
    plt.show()

    sb.set(rc={'figure.figsize': (12.8, 9.6)})
    sb.heatmap(test_cm, annot=True, cmap='YlOrRd')
    plt.title('Test Confusion Matrix')
    plt.show()

    print(sklearn.metrics.classification_report(ytest.argmax(axis=1), ytest_pred.argmax(axis=1)))

    model.save('Models/model.keras')

    with open('Models/scaler', 'wb') as f:
        pickle.dump(scaler, f)

    # To load model and scaler

    # model = keras.models.load_model('saved_Model/model')

    # with open('saved_model/scaler','rb') as f:
    #     scaler = pickle.load(f)







