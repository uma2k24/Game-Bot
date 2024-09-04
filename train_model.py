import os
from keras.models import Model
from keras.optimizers import Adadelta
from keras.layers import Input, Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint, TensorBoard
from get_dataset import get_dataset

def save_model(model):
    if not os.path.exists('Data/Model/'):
        os.makedirs('Data/Model/')
    model_json = model.to_json()
    with open("Data/Model/model.json", "w") as model_file:
        model_file.write(model_json)
    model.save_weights("Data/Model/weights.h5")
    print('Model and weights saved')

def get_model():
    inputs = Input(shape=(150, 150, 3))

    conv_1 = Conv2D(32, (3,3), strides=(1,1))(inputs)
    act_1 = Activation('relu')(conv_1)

    conv_2 = Conv2D(64, (3,3), strides=(1,1))(act_1)
    act_2 = Activation('relu')(conv_2)

    conv_3 = Conv2D(64, (3,3), strides=(1,1))(act_2)
    act_3 = Activation('relu')(conv_3)

    pooling_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(act_3)

    conv_4 = Conv2D(128, (3,3), strides=(1,1))(pooling_1)
    act_4 = Activation('relu')(conv_4)

    pooling_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(act_4)

    flat_1 = Flatten()(pooling_2)

    fc = Dense(1280)(flat_1)
    fc = Activation('relu')(fc)
    fc = Dropout(0.5)(fc)
    fc = Dense(4)(fc)

    outputs = Activation('sigmoid')(fc)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return model

def train_model(model, X, X_test, Y, Y_test):
    checkpoints = []
    if not os.path.exists('Data/Checkpoints/'):
        os.makedirs('Data/Checkpoints/')

    checkpoints.append(ModelCheckpoint('Data/Checkpoints/best_weights.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1))
    checkpoints.append(TensorBoard(log_dir='Data/Checkpoints/./logs', histogram_freq=0, write_graph=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None))

    model.fit(X, Y, batch_size=5, epochs=100, validation_data=(X_test, Y_test), shuffle=True, callbacks=checkpoints)
    return model

def main():
    X, X_test, Y, Y_test = get_dataset()
    model = get_model()
    model = train_model(model, X, X_test, Y, Y_test)
    save_model(model)
    return model

if __name__ == '__main__':
    main()
