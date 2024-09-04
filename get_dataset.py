import os
import numpy as np
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_img(data_path):
    """Read an image from the given path, resize it to 150x150, and return it as a NumPy array."""
    try:
        img = Image.open(data_path)
        img = img.resize((150, 150))
        img = np.array(img)
        return img
    except (OSError, IOError) as e:
        print(f"Error loading image {data_path}: {e}")
        return None

def save_img(img, path):
    """Save a NumPy array as an image to the given path."""
    img = Image.fromarray(img)
    img.save(path)

def get_dataset(dataset_path='Data/Screenshots'):
    """Load images from the dataset directory, preprocess them, and split into training and testing sets."""
    labels = os.listdir(dataset_path) 
    X = []
    Y = []

    # Allowed image extensions
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')

    for label in labels:
        img_path = os.path.join(dataset_path, label)

        # Check if the file is an image by extension
        if not img_path.lower().endswith(valid_extensions):
            print(f"Skipping non-image file: {img_path}")
            continue

        img = get_img(img_path)
        if img is not None:
            X.append(img)
            # Dummy label - should implement a real labeling method based on your use case
            Y.append(0)

    # Convert lists to NumPy arrays and normalize
    X = np.array(X).astype('float32') / 255.0
    Y = np.array(Y).astype('float32')

    # Create directory for saving .npy files if it doesn't exist
    save_dir = 'Data/npy_train_data/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save the dataset to .npy files
    np.save(os.path.join(save_dir, 'X.npy'), X)
    np.save(os.path.join(save_dir, 'Y.npy'), Y)

    # Split data into training and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
    return X_train, X_test, Y_train, Y_test

# Example of using the get_dataset function
X_train, X_test, Y_train, Y_test = get_dataset()
