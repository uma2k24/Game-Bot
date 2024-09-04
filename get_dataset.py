import os
import numpy as np
from PIL import Image, ImageFile
from sklearn.model_selection import train_test_split
import json

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

def load_events(event_path):
    """Load event data from a JSON file and return it as a list of events."""
    events = []
    try:
        with open(event_path, 'r') as file:
            events = json.load(file)
    except (OSError, IOError) as e:
        print(f"Error loading events from {event_path}: {e}")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {event_path}: {e}")
    return events

def get_dataset(dataset_path='Data/Screenshots', mouse_events_path='mouse_events.json', keyboard_events_path='keyboard_events.json'):
    """Load images and events from the dataset directory, preprocess them, and split into training and testing sets."""
    labels = [f for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f)) and not f.startswith('.')]
    X = []
    Y = []

    # Load events
    mouse_events = load_events(mouse_events_path)
    keyboard_events = load_events(keyboard_events_path)

    for label in labels:
        img_path = os.path.join(dataset_path, label)
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
