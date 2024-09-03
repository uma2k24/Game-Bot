# Arda Mavi
import os
import numpy as np
from time import sleep
from PIL import ImageGrab
from scipy.ndimage import zoom
from game_control import *
from predict import predict
from get_dataset import save_img
from multiprocessing import Process
from keras.models import model_from_json
from pynput.mouse import Listener as mouse_listener
from pynput.keyboard import Listener as key_listener

def get_screenshot():
    img = ImageGrab.grab()
    img = np.array(img)[:, :, :3]  # Get first 3 channels from image as numpy array.
    img = zoom(img, (150 / img.shape[0], 150 / img.shape[1], 1), order=3).astype('float32') / 255.
    return img

def save_event_keyboard(data_path, event, key):
    key_id = get_id(key)
    file_path = os.path.join(data_path, f'-1,-1,{event},{key_id}.png')
    screenshot = get_screenshot()
    save_img(file_path, screenshot)

def save_event_mouse(data_path, x, y):
    file_path = os.path.join(data_path, f'{x},{y},0,0.png')
    screenshot = get_screenshot()
    save_img(file_path, screenshot)

def listen_mouse():
    data_path = 'Data/Train_Data/Mouse'
    os.makedirs(data_path, exist_ok=True)

    def on_click(x, y, button, pressed):
        if pressed:
            save_event_mouse(data_path, x, y)

    with mouse_listener(on_click=on_click) as listener:
        listener.join()

def listen_keyboard():
    data_path = 'Data/Train_Data/Keyboard'
    os.makedirs(data_path, exist_ok=True)

    def on_press(key):
        save_event_keyboard(data_path, 1, key)

    def on_release(key):
        save_event_keyboard(data_path, 2, key)

    with key_listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()

def main():
    dataset_path = 'Data/Train_Data/'
    os.makedirs(dataset_path, exist_ok=True)

    # Start listening to mouse events in a new process
    Process(target=listen_mouse).start()
    listen_keyboard()

if __name__ == '__main__':
    main()
