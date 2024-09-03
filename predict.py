import numpy as np
from PIL import Image

def predict(model, X):
    # X'yi PIL Image objesine dönüştür
    img = Image.fromarray(X)
    # Resmi 150x150 boyutlarına yeniden boyutlandır
    img = img.resize((150, 150))
    # Resmi numpy array'e dönüştür ve normalize et
    X_resized = np.array(img).astype('float32') / 255.
    # Model tahmini yap
    Y = model.predict(X_resized.reshape(1, 150, 150, 3))
    return Y
