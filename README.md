# AI Model Training and Control

## Overview

This project involves training a deep learning model for image classification and automating control actions based on predictions. The model is designed to interact with applications by performing keyboard and mouse actions.

## Project Structure

- **`get_dataset.py`**: Handles data loading, preprocessing, and saving.

- **`get_model.py`**: Defines and manages the deep learning model architecture.

- **`train_model.py`**: Trains the model with the dataset and saves the best model weights.

- **`predict.py`**: Contains functions for making predictions with the trained model.

- **`game_control.py`**: Manages keyboard and mouse actions based on predictions.

- **`database.py`**: Manages SQLite database connections and operations.

## Dependencies

- TensorFlow 2.x

- Keras (integrated within TensorFlow 2.x)

- NumPy

- SciPy

- scikit-learn

- Pillow

- pynput

- sqlite3

Install the dependencies using pip:

```sh

pip install tensorflow numpy scipy scikit-learn pillow pynput

```

## Usage

### Data Preparation

1\. **Collect Data**:

   - Use `game_control.py` to record keyboard and mouse actions.

   - The recorded data will be saved in `Data/Train_Data`.

2\. **Create Dataset**:

   - Run `get_dataset.py` to preprocess and save the dataset as numpy arrays.

   ```sh

   python3 get_dataset.py

   ```

### Model Training

1\. **Define and Train the Model**:

   - Run `train_model.py` to train the model with the prepared dataset.

   ```sh

   python3 train_model.py

   ```

2\. **Model Checkpoints**:

   - Checkpoints and TensorBoard logs will be saved in `Data/Checkpoints/`.

### Predictions

1\. **Load and Predict**:

   - Use `predict.py` to make predictions based on new input images.

### Database Management

1\. **SQLite Database**:

   - `database.py` contains functions to interact with an SQLite database.

## File Descriptions

- **`get_dataset.py`**: 

   - Preprocesses images from `Data/Train_Data` and splits them into training and testing datasets.

   - Saves datasets as numpy arrays in `Data/npy_train_data/`.

- **`get_model.py`**:

   - Defines the Convolutional Neural Network (CNN) architecture.

   - Saves and loads model configurations and weights.

- **`train_model.py`**:

   - Trains the model using the dataset and saves the best performing model weights.

- **`predict.py`**:

   - Contains the `predict` function to perform inference on new data.

- **`game_control.py`**:

   - Manages the interaction with the system using keyboard and mouse actions based on predictions.

- **`database.py`**:

   - Provides functions for interacting with an SQLite database, including creating tables and managing data.

## Notes

- Ensure that TensorFlow 2.x is installed as it integrates Keras and provides the required functionalities.

- Adjust paths and configurations according to your project setup.

- Refer to the TensorFlow and Keras documentation for further details on model training and callbacks.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
