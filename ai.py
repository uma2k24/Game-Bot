import os
import numpy as np
from PIL import ImageGrab
from keras.models import model_from_json
from game_control import get_key, press, release, click
from predict import predict

def load_model():
    """Modeli JSON dosyasından yükleyin ve ağırlıkları yükleyin."""
    try:
        with open('Data/Model/model.json', 'r') as model_file:
            model_json = model_file.read()
        model = model_from_json(model_json)
        model.load_weights("Data/Model/weights.h5")
        print('Model yüklendi ve ağırlıklar yüklendi.')
        return model
    except Exception as e:
        print(f"Model yüklenirken bir hata oluştu: {e}")
        raise

def main():
    # Modeli yükle
    model = load_model()

    print('AI başladı!')

    while True:
        try:
            # Ekran görüntüsü al
            screen = ImageGrab.grab()
            # Resmi numpy array'ye dönüştür
            screen = np.array(screen)

            # Modeli kullanarak aksiyonu tahmin et
            Y = predict(model, screen)

            if Y == [0, 0, 0, 0]:
                # Aksiyon gerekli değil
                continue

            if Y[0] == -1 and Y[1] == -1:
                # Sadece klavye aksiyonu
                key = get_key(Y[3])
                if Y[2] == 1:
                    # Tuş bas
                    press(key)
                else:
                    # Tuşu bırak
                    release(key)
            
            elif Y[2] == 0 and Y[3] == 0:
                # Sadece fare aksiyonu
                click(Y[0], Y[1])
            
            else:
                # Fare ve klavye aksiyonu
                # Fare aksiyonu
                click(Y[0], Y[1])
                # Klavye aksiyonu
                key = get_key(Y[3])
                if Y[2] == 1:
                    # Tuş bas
                    press(key)
                else:
                    # Tuşu bırak
                    release(key)

        except Exception as e:
            print(f"Bir hata oluştu: {e}")

        # CPU kullanımını azaltmak için kısa bir uyku süresi ekleyin
        sleep(0.1)

if __name__ == '__main__':
    main()
