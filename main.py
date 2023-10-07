import tensorflow as tf
from tensorflow import keras
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt


class ImageGenerator:
    def __init__(self):
        pass

    def main(self):
        pass

    def load_dataset(
        self, batch_size: int = 32
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Loading the CIFAR-10 dataset and batching the data.

        batch_size: The size of each batch for training, validation, and testing data.

        Returns:
        train_dataset: A TensorFlow Dataset containing training data batches.
        validation_dataset: A TensorFlow Dataset containing validation data batches.
        test_dataset: A TensorFlow Dataset containing test data batches.
        """
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

        # Normalize pixel values to range [0, 1].
        x_train, x_test = x_train / 255.0, x_test / 255.0

        # Split data into training (80%) and validation (20%) sets.
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

        # Create TensorFlow Datasets and batch the data.
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(
            batch_size
        )
        validation_dataset = tf.data.Dataset.from_tensor_slices(
            (x_validation, y_validation)
        ).batch(batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(
            batch_size
        )
        
        '''
        # Visualize some examples
        plt.figure(figsize=(10, 10))
        for i in range(25):
            plt.subplot(5, 5, i + 1)
            plt.imshow(x_train[i])
            plt.axis('off')
        plt.show()
        '''
        return train_dataset, validation_dataset, test_dataset


if __name__ == "__main__":
    img_ai: ImageGenerator = ImageGenerator()
    train_data, validation_data, test_data = img_ai.load_dataset(batch_size=32)
