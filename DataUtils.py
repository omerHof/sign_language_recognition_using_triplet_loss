import numpy as np
import os
from zipfile import ZipFile
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import pandas as pd
from tensorflow.keras.utils import to_categorical

"""
This module implements data handling:
- Import videos (original and augmented) from directory.
- Data merge.
- Train test split. 
"""


DATA_PATH = "/storage/users/.old/.DT/dt-fujitsu-explainability/action_dataset"
sequence_length = 30

def extract_from_zip():
    """
    Function for extract files from zip. Used for extract the raw data from
    zip to folders.
    :return: Save the data in folders, by the labels name.
    """
    # zip_name = "/storage/users/.old/.DT/dt-fujitsu-explainability/new_words.zip"
    #
    # with ZipFile(zip_name, 'r') as zip:
    #   zip.extractall('/storage/users/.old/.DT/dt-fujitsu-explainability/action_dataset')
    #   print("Extracted all landmark files into the folder 'action_dataset'")
    #
    # zip_name = "/storage/users/.old/.DT/dt-fujitsu-explainability/new_aug.zip"
    #
    # with ZipFile(zip_name, 'r') as zip:
    #   zip.extractall('/storage/users/.old/.DT/dt-fujitsu-explainability/aug_dataset')
    #   print("Extracted all landmark files into the folder 'aug_dataset'")

    zip_name = "/storage/users/.old/.DT/dt-fujitsu-explainability/test_words" \
               ".zip"

    with ZipFile(zip_name, 'r') as zip:
      zip.extractall(
          '/storage/users/.old/.DT/dt-fujitsu-explainability/test_dataset')
      print("Extracted all landmark files into the folder 'test_dataset'")

def map_signs(DATA_PATH):
    """
    Generate a list of sign words.
    :return: a list of sign words
    """
    action_list =[]
    for action in os.scandir(DATA_PATH):
        action_list.append(action.path.split('/')[-1])
    return np.array(action_list)


def upload_landmark(DATA_PATH,actions,num_of_samples,test=False):
    if test:
        label_map = {label:num+50 for num, label in enumerate(actions)}
    else:
        label_map = {label: num for num, label in enumerate(actions)}
    sequences, labels = [], []
    for action in actions:
      for sequence in range(1,num_of_samples+1):
          res = np.load(os.path.join(DATA_PATH, action, str(sequence), "landmarks.npy"))
          sequences.append(res)
          labels.append(label_map[action])
    return sequences,labels

# np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int)

def upload_aug_landmark(DATA_PATH,actions,num_of_samples):
  label_map = {label:num for num, label in enumerate(actions)}
  sequences, labels = [], []
  for action in actions:
    for folder in range(1,num_of_samples+1):
      for file in np.array(os.listdir(os.path.join(DATA_PATH, action,
                                                   str(folder)))):
        if os.path.splitext(file)[1]==".npy":
          res = np.load(os.path.join(DATA_PATH, action, str(folder), file))
          if res.shape[0]==30:
            sequences.append(res)
            labels.append(label_map[action])
  return sequences,labels

#np.array(os.listdir(os.path.join(DATA_PATH, action))):
def extract_original_data(original_video_path,num_of_samples,test=False):

    signs = map_signs(original_video_path)
    original_data, original_labels = upload_landmark(original_video_path,
                                                       signs,num_of_samples,
                                                     test)

    return original_data, original_labels, signs

def extract_aug_data(augmented_data_path,num_of_samples):
    signs = map_signs(augmented_data_path)
    aug_data, aug_labels = upload_aug_landmark(augmented_data_path, signs,num_of_samples)
    return aug_data, aug_labels

def data_processing(data,labels,label_encoder = False):
    X = np.asarray(data)
    if label_encoder:
        labelencoder = LabelEncoder()
        y = labelencoder.fit_transform(labels)
    else:
        y = to_categorical(labels).astype(int)
    return X,y

def split_train_test(X_originals,y_originals,test_ratio):
    X_train, X_test, y_train, y_test = train_test_split(X_originals,
                                                        y_originals,
                                                        test_size=test_ratio,
                                                        random_state=42)
    return X_train, X_test, y_train, y_test

def merge_original_with_aug(X_train,y_train,X_aug,y_aug):
    data = np.concatenate((X_train,X_aug),axis=0)
    labels = np.concatenate((y_train, y_aug), axis=0)
    return data,labels

def plot_data_distrubution(labels,signs):
    df = pd.DataFrame(data=labels, columns=['actions'])
    catagories_overview = df.groupby("actions")['actions'].count()
    plt.figure(figsize=(10, 10), dpi=300)
    plt.pie(catagories_overview,
            explode=(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            labels=signs)
    plt.axis('equal')
    plt.title('Proportion of each observed sign gloss')
    plt.show()

def extract_hands_landmarks(train_data, test_data):
    train_data = train_data[:,:, 1536:]
    test_data = test_data[:,:, 1536:]
    return train_data,test_data

def extract_hands_and_pose_landmarks(train_data, test_data):
    train_poses = train_data[:,:, :132]
    train_hands = train_data[:,:, 1536:]
    train_data = np.concatenate([train_poses,train_hands],axis=2)
    test_poses = test_data[:, :, :132]
    test_hands = test_data[:, :, 1536:]
    test_data = np.concatenate([test_poses, test_hands], axis=2)
    return train_data, test_data

def extract_hands_and_face_landmarks(train_data, test_data):
    train_data = train_data[:, :, 132:]
    test_data = test_data[:, :, 132:]
    return train_data, test_data

def extract_data_using_data_utils(samples_num = 90, aug = True, test = False):
    original_video_path = "/storage/users/.old/.DT/dt-fujitsu-explainability/action_dataset"
    augmented_data_path = "/storage/users/.old/.DT/dt-fujitsu-explainability/aug_dataset"
    test_data_path = "/storage/users/.old/.DT/dt-fujitsu-explainability" \
                     "/test_dataset"

    original_videos, original_labels, signs = extract_original_data(
        original_video_path,samples_num)

    X_originals, y_originals = data_processing(original_videos,
                                               original_labels,False)
    X_train, X_test, y_train, y_test = split_train_test(X_originals,
                                                                  y_originals,
                                                                  0.2)

    if aug:
        aug_videos, aug_labels = extract_aug_data(augmented_data_path,
                                                            samples_num)
        X_aug, y_aug = data_processing(aug_videos, aug_labels,False)
        X_train, y_train = merge_original_with_aug(X_train, y_train, X_aug, y_aug)

    if test:
        test_videos, test_labels,test_signs = extract_original_data(
            test_data_path, 10,True)
        test_data, t_ = data_processing(test_videos, test_labels,
                                                  True)
        test_labels = np.array(test_labels)
        X_test, y_test = merge_original_with_aug(X_test, y_test, test_data,
                                                   test_labels)
        signs = np.concatenate((signs, test_signs), axis=0)

    return X_train, X_test, y_train, y_test, signs

if __name__ == '__main__':

    extract_from_zip()

    original_video_path = "/storage/users/.old/.DT/dt-fujitsu-explainability/action_dataset"
    augmented_data_path = "/storage/users/.old/.DT/dt-fujitsu-explainability/aug_dataset"







