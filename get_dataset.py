# Arda Mavi
import os
import numpy as np
from tensorflow.keras.utils import to_categorical
from scipy.ndimage import imread, zoom
from sklearn.model_selection import train_test_split

def get_img(data_path):
    # Getting image array from path:
    img = imread(data_path)
    img = zoom(img, (150/img.shape[0], 150/img.shape[1], 1), order=3)  # Resize with interpolation
    return img

def save_img(img, path):
    from imageio import imwrite
    imwrite(path, img)
    return

def get_dataset(dataset_path='Data/Train_Data'):
    # Getting all data from data path:
    try:
        X = np.load('Data/npy_train_data/X.npy')
        Y = np.load('Data/npy_train_data/Y.npy')
    except FileNotFoundError:
        labels = os.listdir(dataset_path)  # Get labels
        X = []
        Y = []
        count_categori = -1
        label_dict = {}  # For encoding labels
        for label in labels:
            datas_path = os.path.join(dataset_path, label)
            for data in os.listdir(datas_path):
                img = get_img(os.path.join(datas_path, data))
                X.append(img)
                # For encoding labels:
                if label not in label_dict:
                    count_categori += 1
                    label_dict[label] = count_categori
                Y.append(label_dict[label])
        # Create dataset:
        X = np.array(X).astype('float32') / 255.
        Y = np.array(Y).astype('float32')
        Y = to_categorical(Y, count_categori + 1)
        if not os.path.exists('Data/npy_train_data/'):
            os.makedirs('Data/npy_train_data/')
        np.save('Data/npy_train_data/X.npy', X)
        np.save('Data/npy_train_data/Y.npy', Y)
    X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
    return X, X_test, Y, Y_test

