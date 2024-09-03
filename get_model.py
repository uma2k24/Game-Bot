import os
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.layers import Input, Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout

def save_model(model):
    """Saves the model architecture and weights to disk."""
    model_dir = 'Data/Model/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Save model architecture
    model_json = model.to_json()
    with open(os.path.join(model_dir, "model.json"), "w") as model_file:
        model_file.write(model_json)
    
    # Save model weights
    model.save_weights(os.path.join(model_dir, "weights.h5"))
    print('Model and weights saved')
    return

def get_model():
    """Builds and compiles the CNN model."""
    inputs = Input(shape=(150, 150, 3))

    # Convolutional Layer 1
    conv_1 = Conv2D(32, (3, 3), strides=(1, 1), padding='same')(inputs)
    act_1 = Activation('relu')(conv_1)

    # Convolutional Layer 2
    conv_2 = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(act_1)
    act_2 = Activation('relu')(conv_2)

    # Convolutional Layer 3
    conv_3 = Conv2D(64, (3, 3), strides=(1, 1), padding='same')(act_2)
    act_3 = Activation('relu')(conv_3)

    # Max Pooling Layer 1
    pooling_1 = MaxPooling2D(pool_size=(2, 2))(act_3)

    # Convolutional Layer 4
    conv_4 = Conv2D(128, (3, 3), strides=(1, 1), padding='same')(pooling_1)
    act_4 = Activation('relu')(conv_4)

    # Max Pooling Layer 2
    pooling_2 = MaxPooling2D(pool_size=(2, 2))(act_4)

    # Flatten Layer
    flat_1 = Flatten()(pooling_2)

    # Fully Connected Layers
    fc = Dense(1280, activation='relu')(flat_1)
    fc = Dropout(0.5)(fc)
    fc = Dense(4)(fc)  # Change 4 to the number of classes you have

    # Output Layer
    outputs = Activation('softmax')(fc)  # Use 'softmax' for multi-class classification

    # Compile the model
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer=Adadelta(), metrics=['accuracy'])

    return model

if __name__ == '__main__':
    save_model(get_model())
