import cv2
import numpy as np
import os
import io
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import time
from tqdm import tqdm
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import keras as k
import keras.backend as K
from sklearn.preprocessing import LabelEncoder
from keras.layers import *
from keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, Adadelta
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,log_loss
import time
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import TensorBoard
from keras.callbacks import ReduceLROnPlateau #Learning rate scheduler for when we reach plateaus
from sklearn.model_selection import KFold
# from IPython.display import HTML
# from base64 import b64encode
from sklearn.manifold import TSNE
from sklearn.utils import shuffle
from zipfile import ZipFile
import tensorflow as tf
import DataUtils


"""
This module implements sign language recognition state of the art methods.
 - Establish 3 recurrent neural network based architecture: LSTM, Bi-LSTM and 
    GRU.
 - Execute a 10 fold cross validation experiment. 
 - Methods evaluation and plots.
"""


def define_params(benchmarks):
    params = {}
    params['benchmark'] = benchmarks
    params['epochs'] = 100
    params['k folds']= 10
    params['batch size'] = 128
    return params

def create_model_benchmark_1_lstm(signs):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, activation='relu',
                   input_shape=(30,1530)))
    model.add(LSTM(192, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=True, activation='relu'))
    model.add(Dropout(0.1))
    model.add(LSTM(128, return_sequences=False, activation='relu'))
    model.add(Dense(signs.shape[0], activation='softmax'))
    model.summary()
    return model

def create_model_benchmark_1_bidirectional_lstm(signs):
    model = Sequential()
    model.add(Bidirectional(LSTM(128, return_sequences=True),input_shape=(30,
                                                                        1530)))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(Bidirectional(LSTM(64, return_sequences=False)))
    model.add(Dense(signs.shape[0], activation='softmax'))
    model.summary()
    return model

def create_model_benchmark_2_GRU(signs):
    model = Sequential()
    model.add(GRU(128, return_sequences=True,input_shape=(30,1530)))
    model.add(GRU(128, return_sequences=False))
    model.add(Dense(signs.shape[0], activation='softmax'))
    model.summary()
    return model

def define_callbacks():
    output_path = "/home/omerhof/sign_language_project/Outputs/Lstm" \
                  "/my_best_model.epoch{epoch:02d}.hdf5"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=output_path,
        save_best_only=True,
        monitor='val_categorical_accuracy',
        mode='max',
        verbose=1)
    return cp_callback


def select_model(banchmark_method, signs):
    if banchmark_method == '1a':
        model = create_model_benchmark_1_lstm(signs)
    elif banchmark_method == '1b':
        model = create_model_benchmark_1_bidirectional_lstm(signs)
    elif banchmark_method == '2a':
        model = create_model_benchmark_2_GRU(signs)
    model.compile(optimizer=Adam(0.0001), loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    return model


def train_model_using_k_fold(X_train, y_train, X_test, y_test, num_of_epochs, batch_size,
                             k_folds, signs, banchmark_method):
    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    model = None
    k = k_folds
    kfold = KFold(n_splits=k)
    accs, val_accs, losses, val_losses, acc_per_fold, loss_per_fold = [], [], [], [], [], []
    inputs = X_train
    targets = y_train
    callback = define_callbacks()
    for fold, (train_idx, val_idx) in enumerate(kfold.split(inputs, targets)):
        print('Fold {}'.format(fold + 1))
        model = select_model(banchmark_method,signs)
        history = model.fit(inputs[train_idx], targets[train_idx],
                            epochs=num_of_epochs,
                            batch_size=batch_size,
                            validation_data=(inputs[val_idx], targets[val_idx]),callbacks=[callback])
        accs.append(history.history['categorical_accuracy'])
        val_accs.append(history.history['val_categorical_accuracy'][-1])
        losses.append(history.history['loss'])
        val_losses.append(history.history['val_loss'])
        # Generate generalization metrics
        scores = model.evaluate(X_test, y_test, verbose=0)
        acc_per_fold.append(scores[1])
        loss_per_fold.append(scores[0])
        # Increase fold number
    print("Train accuracy: ", history.history['categorical_accuracy'])
    print("Train loss: ", history.history['loss'])
    print("Validation accuracy: ", history.history['val_categorical_accuracy'])
    print("Validation loss: ", history.history['val_loss'])

    return acc_per_fold, loss_per_fold, accs, val_accs, losses, val_losses, model

def train_model_using_train_test_split(X_train, y_train, X_test, y_test, num_of_epochs,
                                       batch_size, signs, banchmark_method):
    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    model = None
    callback = define_callbacks()
    model = select_model(banchmark_method, signs)
    history = model.fit(X_train, y_train,
                        epochs=num_of_epochs,
                        batch_size=batch_size,
                        validation_split = 0.2,
                        callbacks=[callback])
    train_accs = history.history['categorical_accuracy']
    val_accs = history.history['val_categorical_accuracy']
    train_losses = history.history['loss']
    val_losses = history.history['val_loss']
    # Generate generalization metrics
    scores = model.evaluate(X_test, y_test, verbose=0)
    test_acc = scores[1]
    test_loss = scores[0]
    return train_accs, val_accs, train_losses, val_losses, test_acc, test_loss

def plot_loss_acc(acc_per_fold, loss_per_fold, mean_acc, mean_loss,title):
    # print(mean_acc)
    plt.figure()
    acc_loss_list = [[]]
    for i in range(10):
        acc_loss_list.append([str(i+1),'Loss (X100)',loss_per_fold[i]])
        acc_loss_list.append([str(i+1),'Accuracy',acc_per_fold[i]])
    acc_loss_list.append(["Test Mean",'Loss (X100)',mean_loss ])
    acc_loss_list.append(["Test Mean", 'Accuracy',  mean_acc])
    acc_loss_dataframe=pd.DataFrame(acc_loss_list ,columns=['Folds/Test mean','Section','Score'])
    ax = None
    ax = sns.barplot(x="Folds/Test mean", y="Score", hue='Section', data=acc_loss_dataframe).set_title('Accuracy and Loss scores'+str(title))

def plot_subplots(acc, loss, val_acc, val_loss):
    fig, ax = plt.subplots(1,2,figsize=(12,4))
    ax[0].plot(acc)
    ax[0].plot(val_acc)
    ax[0].set_title('Model accuracy')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].legend(['Train', 'Validation'], loc='upper left')

    # Plot training & validation loss values
    ax[1].plot(loss)
    ax[1].plot(val_loss)
    ax[1].set_title('Model loss')
    ax[1].set_ylabel('Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].legend(['Train', 'Validation'], loc='upper left')
    plt.suptitle("Train - Validation comparision between Accuracy & Loss metrics (mean of all folds) ", fontsize=14)
    plt.show()

def compare_train_validation(acc_per_fold, loss_per_fold, accs, val_accs, losses, val_losses, model):
    np_accs= np.array(accs)
    means_accs = np_accs.mean(axis=0)
    np_losses= np.array(losses)
    means_losses = np_losses.mean(axis=0)
    np_val_accs= np.array(val_accs)
    means_val_accs = np_val_accs.mean(axis=0)
    np_val_losses= np.array(val_losses)
    means_val_losses = np_val_losses.mean(axis=0)
    plot_subplots(means_accs, means_losses, means_val_accs, means_val_losses)

def compare_validation_test(val_accs, val_losses, acc_per_fold, loss_per_fold):
    from statistics import mean
    list_val_accs = [item for sublist in val_accs for item in sublist]
    list_val_losses = [item for sublist in val_losses for item in sublist]
    plot_loss_acc(list_val_accs, list_val_losses, mean(acc_per_fold), mean(loss_per_fold)," for validation folds and test mean")
    plot_loss_acc(acc_per_fold, loss_per_fold, mean(acc_per_fold), mean(loss_per_fold)," for test folds and test mean")
    # print(acc_per_fold)
    # print( mean(acc_per_fold))

def create_heat_map(model, X_test, y_test,signs):
    plt1 = plt.figure(figsize=(10, 10))
    preds = model.predict(X_test, batch_size=128)
    preds_cat = np.argmax(preds,axis=1)
    y_test_not_dummies = np.argmax(y_test,axis=1)
    print('model accuracy on test set is: {0:.2f}%'.format(accuracy_score(y_test_not_dummies,preds_cat)*100))
    sns.heatmap(confusion_matrix(y_test_not_dummies,preds_cat),cmap='Greens',
                xticklabels=signs, yticklabels=signs ,annot=True, fmt='d')
    plt.xlabel('Prediction')
    plt.ylabel('True label')
    plt.title('Sign Language recognition LSTM model \n classification results on test set')
    plt.show()

def plot_test_accuracies(num_of_samples, test_res_dict):
    plt.clf()
    plt.figure()
    plt.axvline(x=10, color='r', linestyle='dotted')
    plt.axvline(x=30, color='r', linestyle='dotted')
    plt.plot(num_of_samples, test_res_dict['LSTM'], label = 'LSTM',color='blue')
    plt.plot(num_of_samples, test_res_dict['LSTM-augmented'], label = 'LSTM-augmented',
             color='blue',linestyle='dashed')
    plt.plot(num_of_samples, test_res_dict['Bi-LSTM'], label = 'BI-LSTM',
             color='orange')
    plt.plot(num_of_samples, test_res_dict['Bi-LSTM-augmented'], label =
    'BI-LSTM-augmented',
             color='orange',linestyle='dashed')
    plt.plot(num_of_samples, test_res_dict['GRU'], label = 'GRU',color='green')
    plt.plot(num_of_samples, test_res_dict['GRU-augmented'], label = 'GRU-augmented',
             color='green',linestyle='dashed')
    # for key in test_res_dict:
    #     plt.plot(num_of_samples, test_res_dict[key], label = key)

    plt.xlabel("Number of examples per sign")
    plt.ylabel("Accuracy")
    plt.title("Change of accuracy by number of examples per sign")
    plt.xticks(np.arange(min(num_of_samples), max(num_of_samples), 5.0))
    plt.ylim([30, 100])
    plt.legend()
    plt.show()


def experiment_by_k_fold_cross_validation():
    X_train, X_test, y_train, y_test, signs = \
        DataUtils.extract_data_using_data_utils(90,False)
    benchmark = '1b'
    params = define_params(benchmark)
    if params['benchmark']=='1a' or params['benchmark']=='1b' or params[
        'benchmark']=='2a':
        X_train, X_test, = DataUtils.extract_hands_landmarks(X_train, X_test)
    # elif params['benchmark']=='2a':
    #     X_train, X_test, = DataUtils.extract_key_point_benchmark_2(X_train, X_test)
    print(f"Number of extracted key point is: {X_train.shape[2]/3}")
    start_time = time.time()
    acc_per_fold, loss_per_fold, accs, val_accs, losses, val_losses, model = \
        train_model_using_k_fold(X_train, y_train, X_test, y_test, params['epochs'],
                                 params['batch size'], params['k folds'], signs,
                                 banchmark_method=params['benchmark'])
    print("--- %s seconds ---" % (time.time() - start_time))
    print(val_accs)
    print("Mean acc of validation folds:")
    print(np.mean(np.array(val_accs).mean()))
    compare_train_validation(acc_per_fold, loss_per_fold, accs, val_accs, losses, val_losses, model)
    create_heat_map(model, X_test, y_test,signs)





def sample_experiment():
    num_of_samples = [90]
    test_accuracies_dict = {}
    benchmarks = ['1b']
    benchmarks_method = ['LSTM','BI-LSTM','GRU']
    for idx,benchmark in enumerate(benchmarks):
        test_accuracies = []
        for num in num_of_samples:
            X_train, X_test, y_train, y_test, signs = \
                DataUtils.extract_data_using_data_utils(samples_num = num,
                                                        aug = False)
            params = define_params(benchmark)
            # if params['benchmark'] == '1a' or params['benchmark'] == '1b':
            # X_train, X_test, = DataUtils.extract_key_point_benchmark_1(X_train,
            #                                                                X_test)

            # elif params['benchmark'] == '2':
            X_train, X_test, = DataUtils.extract_hands_and_face_landmarks(X_train,
                                                                          X_test)
            start_time = time.time()
            train_accs, val_accs, train_losses, val_losses, test_acc, test_loss = \
                train_model_using_train_test_split(X_train, y_train, X_test, y_test,
                                       params['epochs'],
                                         params['batch size'], signs,
                                         banchmark_method=params['benchmark'])
            print(f'Results for {num} samples in benchmark {benchmark}')
            print(f'Runtime: {(time.time() - start_time)/60} minutes, train acc:'
                  f' {train_accs}, train loss:'
                  f' {train_losses}, '
                  f'validation acc: {val_accs}, validation loss: {val_losses}, '
                  f'test acc: {test_acc}, test loss: {test_loss}')
            test_accuracies.append(test_acc*100)
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print(f"Test accuracy for benchmark {benchmarks_method[idx]} with"
              f" {num} samples: {test_accuracies}")
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        test_accuracies_dict[benchmarks_method[idx]] = test_accuracies
    # plot_test_accuracies(num_of_samples,test_accuracies_dict)
    print(test_accuracies_dict)


# Demo experiments:

experiment_by_k_fold_cross_validation()

