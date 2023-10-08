import tensorflow as tf

print("TensorFlow version:", tf.__version__)

from tensorflow import keras
from keras.layers import Dense, Flatten, Conv2D
from keras import Model
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt


class ImageGenerator(Model):
    def __init__(self):
        super().__init__()
        self.x_train, self.y_train = None, None
        self.x_test, self.y_test = None, None

    def main(self):
        self.load_dataset()
        self.create_model()

    def load_dataset(
        self, batch_size: int = 32
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Load and prepare the CIFAR-10 dataset.

        Args:
        batch_size: The size of each batch for training, validation, and testing data.

        Returns:
        train_dataset: A TensorFlow Dataset containing training data batches.
        validation_dataset: A TensorFlow Dataset containing validation data batches.
        test_dataset: A TensorFlow Dataset containing test data batches.
        """
        (self.x_train, self.y_train), (self.x_test, self.y_test) = keras.datasets.cifar10.load_data()

        # Normalize pixel values to range [0, 1].
        self.x_train, self.x_test = self.x_train / 255.0, self.x_test / 255.0

        # Split data into training (80%) and validation (20%) sets.
        split_ratio: float = 0.2
        num_samples: int = len(self.x_train)
        num_validation_samples: int = int(num_samples * split_ratio)
        indices: np.ndarray = np.arange(num_samples)
        np.random.shuffle(indices)
        self.x_train, self.y_train = self.x_train[indices], self.y_train[indices]
        x_validation, y_validation = (
            self.x_train[:num_validation_samples],
            self.y_train[:num_validation_samples],
        )
        self.x_train, self.y_train = (
            self.x_train[num_validation_samples:],
            self.y_train[num_validation_samples:],
        )

        # Create TensorFlow Datasets and batch the data.
        train_dataset = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train)).batch(
            batch_size
        )
        validation_dataset = tf.data.Dataset.from_tensor_slices(
            (x_validation, y_validation)
        ).batch(batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test)).batch(
            batch_size
        )

        return train_dataset, validation_dataset, test_dataset

    def create_model(self):
        model = keras.Sequential(
            [
                keras.layers.Conv2D(
                    32,
                    (3, 3),
                    activation="relu",
                    padding="same",
                    input_shape=(32, 32, 3),
                ),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Flatten(),
                keras.layers.Dense(128, activation="relu"),
                keras.layers.Dense(10),
            ]
        )
        


if __name__ == "__main__":
    img_ai: ImageGenerator = ImageGenerator()
    #train_data, validation_data, test_data = img_ai.load_dataset(batch_size=32)
    img_ai.main()
