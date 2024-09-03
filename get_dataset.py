import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

def get_img(data_path):
    img = Image.open(data_path)
    img = img.resize((150, 150))
    img = np.array(img)
    return img

def save_img(img, path):
    img = Image.fromarray(img)
    img.save(path)
    return

def get_dataset(dataset_path='Data/Screenshots'):
    labels = os.listdir(dataset_path) 
    X = []
    Y = []

    for label in labels:
        img_path = os.path.join(dataset_path, label)
        img = get_img(img_path)
        X.append(img)
        # Dummy label - you should implement a way to generate real labels based on your use case
        Y.append(0)

    X = np.array(X).astype('float32') / 255.
    Y = np.array(Y).astype('float32')

    if not os.path.exists('Data/npy_train_data/'):
        os.makedirs('Data/npy_train_data/')
    np.save('Data/npy_train_data/X.npy', X)
    np.save('Data/npy_train_data/Y.npy', Y)

    X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
    return X, X_test, Y, Y_test
