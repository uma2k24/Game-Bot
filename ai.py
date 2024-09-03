import os
import numpy as np
from PIL import ImageGrab
from keras.models import model_from_json
from game_control import get_key, press, release, click
from predict import predict

def main():
    # Get Model:
    with open('Data/Model/model.json', 'r') as model_file:
        model_json = model_file.read()
    
    model = model_from_json(model_json)
    model.load_weights("Data/Model/weights.h5")

    print('AI start now!')

    while True:
        # Get screenshot:
        screen = ImageGrab.grab()
        # Image to numpy array:
        screen = np.array(screen)

        # Predict action based on the model
        Y = predict(model, screen)

        if Y == [0, 0, 0, 0]:
            # No action needed
            continue
        
        if Y[0] == -1 and Y[1] == -1:
            # Keyboard action only
            key = get_key(Y[3])
            if Y[2] == 1:
                # Press key
                press(key)
            else:
                # Release key
                release(key)
        
        elif Y[2] == 0 and Y[3] == 0:
            # Mouse action only
            click(Y[0], Y[1])
        
        else:
            # Mouse and keyboard action
            # Mouse action
            click(Y[0], Y[1])
            # Keyboard action
            key = get_key(Y[3])
            if Y[2] == 1:
                # Press key
                press(key)
            else:
                # Release key
                release(key)

if __name__ == '__main__':
    main()
