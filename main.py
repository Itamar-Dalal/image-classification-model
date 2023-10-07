import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import Tuple


class ImageGenerator:
    def __init__(self):
        pass

    def main(self):
        pass

    def load_dataset(
        self,
    ) -> Tuple[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray],
    ]:
        """
        Loading the CIFAR-10 dataset.

        x_train: This variable will hold the training images. These are the images used to train a machine learning model.
        y_train: This variable will hold the corresponding labels or class indices for the training images.
        x_test: This variable will hold the testing images. These are used to evaluate the model's performance.
        y_test: This variable will hold the corresponding labels or class indices for the testing images.
        """
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

        """
        Normalize pixel values to range [0, 1].
        """
        x_train, x_test = x_train / 255.0, x_test / 255.0

        """
        Split data into training (80%) and validation (20%) sets.
        """
        split_ratio: float = 0.2
        num_samples: int = len(x_train)
        num_validation_samples: int = int(num_samples * split_ratio)
        indices: np.ndarray = np.arange(num_samples)
        np.random.shuffle(indices)
        x_train, y_train = x_train[indices], y_train[indices]
        x_validation, y_validation = (
            x_train[:num_validation_samples],
            y_train[:num_validation_samples],
        )
        x_train, y_train = (
            x_train[num_validation_samples:],
            y_train[num_validation_samples:],
        )

        """
        Organize and return the data sets.
        """
        train_data: Tuple[np.ndarray, np.ndarray] = (x_train, y_train)
        validation_data: Tuple[np.ndarray, np.ndarray] = (x_validation, y_validation)
        test_data: Tuple[np.ndarray, np.ndarray] = (x_test, y_test)
        return train_data, validation_data, test_data


if __name__ == "__main__":
    img_ai: ImageGenerator = ImageGenerator()
    img_ai.load_dataset()
