import numpy as np
from keras.models import model_from_json
from PIL import Image

def load_model():
    with open('Data/Model/model.json', 'r') as model_file:
        model_json = model_file.read()
    model = model_from_json(model_json)
    model.load_weights('Data/Model/weights.h5')
    return model

def predict(img_path):
    model = load_model()
    img = Image.open(img_path)
    img = img.resize((150, 150))
    img = np.array(img).astype('float32') / 255.
    img = np.expand_dims(img, axis=0)
    
    prediction = model.predict(img)
    return prediction

if __name__ == '__main__':
    img_path = 'Data/Screenshots/sample.png'  # Update with the path to your image
    prediction = predict(img_path)
    print(prediction)
