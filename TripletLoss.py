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
# import tensorflow_datasets as tfds
from scipy.spatial.distance import pdist, squareform
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
import imageio
import Banchmarks

"""
This module implements sign language recognition using triplet loss method.
 - Establish 3 recurrent neural network based architecture: LSTM, Bi-LSTM and 
    GRU.
 - Execute a 10 fold cross validation experiment. Train the models using 
 triplet loss function. 
 - Method evaluation and plots.
"""



DATA_PATH = "/storage/users/.old/.DT/dt-fujitsu-explainability/action_dataset"
sequence_length = 30

def extract_from_zip():
    zip_name = "/storage/users/.old/.DT/dt-fujitsu-explainability/MP_Data30.zip"

    with ZipFile(zip_name, 'r') as zip:
      zip.extractall('/storage/users/.old/.DT/dt-fujitsu-explainability/action_dataset')
      print("Extracted all landmark files into the folder 'action_dataset'")

    zip_name = "/storage/users/.old/.DT/dt-fujitsu-explainability/words.zip"

    with ZipFile(zip_name, 'r') as zip:
      zip.extractall('/storage/users/.old/.DT/dt-fujitsu-explainability/aug_dataset')
      print("Extracted all landmark files into the folder 'aug_dataset'")

def map_actions():
  action_list =[]
  for action in os.scandir(DATA_PATH):
    action_list.append(action.path.split('/')[-1])
  return np.array(action_list)


def upload_landmark(DATA_PATH,actions):
  label_map = {label:num for num, label in enumerate(actions)}
  sequences, labels = [], []
  for action in actions:
      for sequence in np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int):
          res = np.load(os.path.join(DATA_PATH, action, str(sequence), "landmarks.npy"))
          sequences.append(res)
          labels.append(label_map[action])
  return sequences,labels

def upload_aug_landmark(DATA_PATH,actions):
  label_map = {label:num for num, label in enumerate(actions)}
  sequences, labels = [], []
  for action in actions:
    for folder in np.array(os.listdir(os.path.join(DATA_PATH, action))):
      for file in np.array(os.listdir(os.path.join(DATA_PATH, action,folder))):
        if os.path.splitext(file)[1]==".npy":
          res = np.load(os.path.join(DATA_PATH, action, folder, file))
          if res.shape[0]==30:
            sequences.append(res)
            labels.append(label_map[action])
  return sequences,labels

def extract_data():

    DATA_PATH = "/storage/users/.old/.DT/dt-fujitsu-explainability/action_dataset"
    actions = map_actions()
    sequence_length = 30
    sequences, labels = upload_landmark(DATA_PATH,actions)

    DATA_PATH = "/storage/users/.old/.DT/dt-fujitsu-explainability/aug_dataset"
    sequences1, labels1 = upload_aug_landmark(DATA_PATH,actions)
    sequences = sequences+sequences1
    labels = labels + labels1

    X = np.asarray(sequences)
    labelencoder = LabelEncoder()
    y = labelencoder.fit_transform(labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test,actions

def create_model_lstm():
    model = Sequential()
    model.add(LSTM(64, return_sequences=False, activation='relu',
                   input_shape=(30,126)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(50, activation=None))# No activation on final dense layer
    model.add(Lambda(lambda x: tf.math.l2_normalize(x, axis=1))) # L2 normalize embeddings
    return model

def create_model_gru():
    model = Sequential()
    model.add(GRU(128, return_sequences=False, input_shape=(30, 126)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation=None))
    model.add(Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))
    model.summary()
    return model

def create_model_bi_lstm():
    model = Sequential()
    model.add(Bidirectional(LSTM(128, return_sequences=False), input_shape=(30,
                                                                           126)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(50, activation=None))
    model.add(Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))
    model.summary()
    return model

def define_params():
    params = {}
    params['banchmark method'] = 'gru'
    params['k fold'] = 10
    params['epochs'] = 100
    params['validation']= 0.2
    params['batch size'] = 128
    return params

def define_callbacks():
    output_path = "/home/omerhof/sign_language_project/Outputs/checkpoints" \
                  "/triplet_bi_lstm/my_best_model.epoch{epoch:02d}.hdf5"
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=output_path,
        verbose=1,
        save_freq='epoch',
        period = 10)
    return cp_callback

def select_model(banchmark_method):
    if banchmark_method == 'lstm':
        model = create_model_lstm()
    elif banchmark_method == 'gru':
        model = create_model_gru()
    elif banchmark_method == 'bi-lstm':
        model = create_model_bi_lstm()
    model.compile(
        optimizer=Adam(0.0001),
        loss=tfa.losses.TripletSemiHardLoss())
    return model

def train_model_using_k_fold(X_train, y_train, X_test, y_test, num_of_epochs, batch_size,
                             k_folds, signs, banchmark_method):
    X_train, y_train = shuffle(X_train, y_train, random_state=42)
    model = None
    k = k_folds
    kfold = KFold(n_splits=k)
    losses, val_losses, acc_per_fold = [], [], []
    inputs = X_train
    targets = y_train
    callback = define_callbacks()
    for fold, (train_idx, val_idx) in enumerate(kfold.split(inputs, targets)):
        print('Fold {}'.format(fold + 1))
        model = select_model(banchmark_method)
        history = model.fit(inputs[train_idx], targets[train_idx],
                            epochs=num_of_epochs,
                            batch_size=batch_size,
                            validation_data=(inputs[val_idx], targets[val_idx]),
                            callbacks=[callback])
        losses.append(history.history['loss'])
        val_losses.append(history.history['val_loss'])
        # Generate generalization metrics
        predictions = model_predict(X_test,model)
        _,score = find_nearest_neighbors(predictions, y_test, neighbors=3)
        acc_per_fold.append(score)

        # Increase fold number

    print("Last fold train loss: ", history.history['loss'])
    print("Last fold validation loss: ", history.history['val_loss'])
    print ("Folds loss: ", acc_per_fold)
    return acc_per_fold, losses, val_losses, model


def upload_models_and_plot_tsne(models_path, X_test,signs,y_test):
    for idx,model_path in enumerate(os.listdir(models_path)):
        model = k.models.load_model(os.path.join(models_path,model_path))
        if idx % 10==0:
            plot_t_sne(model.predict(X_test), signs, y_test, idx)



def plot_t_sne(results,signs,y_test,iteration):
    plt.clf()
    print(f"tsne image number {iteration}")
    action_for_plot = [signs[action] for action in y_test]
    fig = plt.figure(figsize=(10,15), dpi=300)
    tsne = TSNE()
    X_embedded = TSNE(n_components=2, learning_rate='auto',
                      init='random').fit_transform(results)
    X_embedded.shape

    df = pd.DataFrame()
    df["y"] = y_test
    df["comp-1"] = X_embedded[:, 0]
    df["comp-2"] = X_embedded[:, 1]
    sns.scatterplot(x="comp-1", y="comp-2", hue=action_for_plot,
                    palette=sns.color_palette("hls", 50),
                    data=df).set(title=f"TLS-TL T-SNE Projection Epoch "
                                       f"{iteration}")
    output_path = "/home/omerhof/sign_language_project/Outputs/tsne/tsnee_gru"
    plt.legend(bbox_to_anchor=(0.95, 1), loc=2, borderaxespad=0.)
    # plt.show()
    plt.savefig(f'{output_path}/tsne_{iteration}.png', dpi=fig.dpi)

def train_triplet_loss(X_train, y_train):
    params = define_params()
    model = create_model_gru()
    model.compile(
        optimizer=Adam(0.0001),
        loss=tfa.losses.TripletSemiHardLoss())
    # plot_t_sne(model.predict(X_test))
    callback = define_callbacks()
    history = model.fit(X_train, y_train, validation_split=params['validation'],
                        batch_size=params['batch size'],
                        epochs=params['epochs'],
                        callbacks=[callback])
    return model, history

def model_predict(X_test,model):
    predictions = model.predict(X_test)
    return predictions

def load_model(model_path = None):
    if model_path==None:
        model_path = "/home/omerhof/sign_language_project/Outputs/checkpoints" \
                     "/'my_best_model" \
                     ".epoch200.hdf5"
    model = k.models.load_model(model_path)
    return model

def find_nearest_neighbors(predictions,y_test,neighbors=3):
    distances = pdist(predictions, metric='euclidean')
    dist_matrix = squareform(distances)
    y_pred=[]
    for idx,sign in enumerate(dist_matrix):
        max_idx = np.argpartition(sign, neighbors)
        nearest_neighbors = y_test[max_idx[1:neighbors+1]]
        y_pred.append(np.bincount(nearest_neighbors).argmax())
        # signs[most_frequent_label]
    correct = 0
    test_correct = 0
    for idx, pred in enumerate(y_pred):
        if pred == y_test[idx]:
            correct += 1
            if idx>899:
                test_correct += 1
    score = correct/len(y_pred)
    # score_2 = test_correct/100
    return y_pred,score #score_2


def plot_with_projector(predictions,y_test,signs):
    output_path = '/home/omerhof/sign_language_project/Outputs/projection'
    vecs = "vecs.tsv"
    meta = 'meta.tsv'
    np.savetxt(os.path.join(output_path,vecs), predictions, delimiter='\t')
    labels = []
    for i in range(len(y_test)):
        labels.append(signs[y_test[i]])
    out_m = io.open(os.path.join(output_path,meta), 'w', encoding='utf-8')
    for label in labels:
        out_m.write(label + "\n")
    out_m.close()

def compare_train_validation(losses, val_losses):
    plt.figure()
    np_losses= np.array(losses)
    means_losses = np_losses.mean(axis=0)
    np_val_losses= np.array(val_losses)
    means_val_losses = np_val_losses.mean(axis=0)
    plt.plot(means_losses)
    plt.plot(means_val_losses)
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.suptitle(
        "Train - Validation comparison of the loss metric (mean of all folds)",
        fontsize=14)
    plt.show()

def create_heat_map(model, X_test, y_test,signs,method):
    plt1 = plt.figure(figsize=(10, 10))
    predictions = model_predict(X_test, model)
    y_pred,score = find_nearest_neighbors(predictions, y_test, neighbors=3)
    print(f'model accuracy on test set is: {score}')
    sns.heatmap(confusion_matrix(y_test,y_pred),cmap='Greens',
                xticklabels=signs, yticklabels=signs ,annot=True, fmt='d')
    plt.xlabel('Prediction')
    plt.ylabel('True label')
    plt.title(f"Sign language classification using {method} model on the test set")
    plt.show()

def experiment_k_folds():
    X_train, X_test, y_train, y_test, signs = \
            DataUtils.extract_data_using_data_utils(90,False)
    X_train, X_test, = DataUtils.extract_hands_landmarks(X_train, X_test)
    params = define_params()
    start_time = time.time()
    acc_per_fold, losses, val_losses, model = train_model_using_k_fold(
        X_train,y_train, X_test, y_test, params['epochs'],
        params['batch size'], params['k fold'], signs,
        params['banchmark method'])

    print("--- %s seconds ---" % (time.time() - start_time))
    compare_train_validation(losses, val_losses)
    create_heat_map(model, X_test, y_test, signs,params['banchmark method'])

def simple_train_test_split():
    X_train, X_test, y_train, y_test, signs = \
        DataUtils.extract_data_using_data_utils(90, False)
    X_train, X_test, = DataUtils.extract_hands_landmarks(X_train, X_test)
    # plot_t_sne(X_test, signs, y_test,0)
    model, history = train_triplet_loss(X_train, y_train)
    predictions = model_predict(X_test,model)
    _,score,_ = find_nearest_neighbors(predictions,y_test,signs)

def test_model_with_new_words():
    X_train, X_test, y_train, y_test, signs = \
        DataUtils.extract_data_using_data_utils(90, False,True)
    model = load_model("/home/omerhof/sign_language_project/Outputs"
                       "/checkpoints/triplet_gru/my_best_model.epoch200.hdf5")

    predictions = model_predict(X_test, model)
    # for i in range(1,10):
    #     y_pred,score,score_2 = find_nearest_neighbors(predictions,y_test,i)
    #     print(score,score_2)
    find_nearest_neighbors(predictions, y_test, 3)
    plot_t_sne(predictions, signs, y_test)

def create_gif(input_path,output_path):
    images = []
    onlyfiles = [f for f in os.listdir(input_path) if os.path.isfile(
        os.path.join(input_path,f))]
    for filename in onlyfiles:
        images.append(imageio.imread(os.path.join(input_path,filename)))
    imageio.mimsave(output_path, images, format='GIF', duration=0.5)

def model_for_plot(X_test,y_test,signs):
    model = Sequential()
    model.add(LSTM(64, return_sequences=False, activation='relu',
                   input_shape=(30, 126)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(50, activation=None))
    model.compile()
    preds = model_predict(X_test,model)
    plot_t_sne(preds,signs,y_test,0)
    # No activation on final dense layer

def sample_experiment():
    num_of_samples = [5,10,20,30,40,60,90]
    test_accuracies_dict = {}
    benchmarks = ['gru']
    benchmarks_method = ['gru']
    for idx,benchmark in enumerate(benchmarks):
        test_accuracies = []
        for num in num_of_samples:
            X_train, X_test, y_train, y_test, signs = \
                DataUtils.extract_data_using_data_utils(samples_num = num,
                                                        aug = False)

            X_train, X_test, = DataUtils.extract_hands_landmarks(X_train,
                                                                 X_test)
            start_time = time.time()
            model, history =  train_triplet_loss (X_train,y_train)
            predictions = model_predict(X_test,model)
            _,test_acc = find_nearest_neighbors(predictions,y_test,3)
            print(f'Results for {num} samples in benchmark {benchmark}')
            print(f'Runtime: {(time.time() - start_time)/60} minutes, '
                  f'test acc: {test_acc}')
            test_accuracies.append(test_acc*100)
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print(f"Test accuracy for benchmark {benchmarks_method[idx]} with"
              f" {num} samples: {test_accuracies}")
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        test_accuracies_dict[benchmarks_method[idx]] = test_accuracies
    print(test_accuracies_dict)


# Demo experiment
sample_experiment
test_model_with_new_words()







