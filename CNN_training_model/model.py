import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


class CIFAR10Trainer(keras.Model):
    BATCH_SIZE = 32
    IMAGE_HEIGHT = 32
    IMAGE_WIDTH = 32
    NUM_CLASSES = 10

    def __init__(self, num_epochs):
        """
        Initialize the CIFAR10Trainer class.

        This class serves as a container for training and evaluating a convolutional neural network (CNN)
        on the CIFAR-10 dataset, a commonly used dataset for image classification tasks.
        """
        super().__init__()
        self.x_train, self.y_train = None, None
        self.x_test, self.y_test = None, None
        (
            self.train_dataset,
            self.validation_dataset,
            self.test_dataset,
        ) = self.load_dataset(batch_size = self.BATCH_SIZE)
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam()
        self.num_epochs = num_epochs

    def main(self):
        """
        Main training and evaluation function.

        This function orchestrates the entire training and evaluation process:
        1. Loads the CIFAR-10 dataset.
        2. Creates and compiles a CNN model.
        3. Trains the model.
        4. Evaluates the model on the test dataset.
        5. Plots training and validation accuracy.
        """
        self.load_dataset()
        self.create_model()
        history = self.train_model()
        self.evaluate_model()
        self.plot_accuracy(history)

    def load_dataset(
        self, batch_size: int = BATCH_SIZE
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Load and prepare the CIFAR-10 dataset.

        Args:
            batch_size (int): Batch size for training.

        Returns:
            Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: Training, validation, and test datasets.

        This function loads the CIFAR-10 dataset, normalizes pixel values, and splits it into training,
        validation, and test sets. It also creates TensorFlow Datasets and batches the data for training.
        """
        # Load and prepare the CIFAR-10 dataset.
        (self.x_train, self.y_train), (
            self.x_test,
            self.y_test,
        ) = keras.datasets.cifar10.load_data()

        # Normalize pixel values to the range [0, 1].
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
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (self.x_train, self.y_train)
        ).batch(batch_size)
        validation_dataset = tf.data.Dataset.from_tensor_slices(
            (x_validation, y_validation)
        ).batch(batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices(
            (self.x_test, self.y_test)
        ).batch(batch_size)

        return train_dataset, validation_dataset, test_dataset

    def create_model(self):
        """
        Create and compile the model.

        This function defines and compiles a convolutional neural network (CNN) model for image classification.
        The model architecture consists of convolutional layers, max-pooling layers, and dense layers.
        """
        model = keras.Sequential(
            [
                keras.layers.Conv2D(
                    32,
                    (3, 3),
                    activation="relu",
                    padding="same",
                    input_shape=(self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 3),
                ),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
                keras.layers.MaxPooling2D((2, 2)),
                keras.layers.Flatten(),
                keras.layers.Dense(128, activation="relu"),
                keras.layers.Dense(self.NUM_CLASSES),
            ]
        )

        model.compile(
            optimizer=self.optimizer,
            loss=self.loss_fn,
            metrics=["accuracy"],
        )

        self.model = model

    def train_model(self):
        """
        Train the model.

        Args:
            num_epochs (int): Number of training epochs.

        Returns:
            dict: Training history.

        This function trains the CNN model on the training dataset for a specified number of epochs.
        It also records and returns the training history, including training and validation accuracy over epochs.
        """
        history = {"accuracy": [], "val_accuracy": []}
        for epoch in range(self.num_epochs):
            for batch_data, batch_labels in self.train_dataset:
                with tf.GradientTape() as tape:
                    logits = self.model(batch_data)
                    loss_value = self.loss_fn(batch_labels, logits)

                gradients = tape.gradient(loss_value, self.model.trainable_variables)
                self.optimizer.apply_gradients(
                    zip(gradients, self.model.trainable_variables)
                )

            validation_accuracy = self.calculate_validation_accuracy()
            history["accuracy"].append(validation_accuracy)
            history["val_accuracy"].append(validation_accuracy)
            print(
                f"Epoch {epoch+1}/{self.num_epochs}, Validation Accuracy: {validation_accuracy:.4f}"
            )

        return history

    def evaluate_model(self):
        """
        Evaluate the model on the test dataset.

        This function evaluates the trained model on the test dataset and prints the test accuracy.
        """
        _, test_accuracy = self.model.evaluate(self.test_dataset)
        print(f"Test Accuracy: {test_accuracy:.4f}")

    def calculate_validation_accuracy(self):
        """
        Calculate validation accuracy.

        Returns:
            float: Validation accuracy.

        This function calculates the validation accuracy of the model on the validation dataset.
        """
        correct_predictions = 0
        total_predictions = 0

        for batch_data, batch_labels in self.validation_dataset:
            logits = self.model(batch_data)
            predicted_labels = np.argmax(logits, axis=1)
            correct_predictions += np.sum(predicted_labels == batch_labels)
            total_predictions += batch_labels.shape[0]

        return correct_predictions / total_predictions

    def plot_accuracy(self, history):
        """
        Plot training and validation accuracy.

        This function plots the training and validation accuracy over epochs.
        """
        plt.figure(figsize=(8, 6))
        plt.plot(history["accuracy"], label="Training Accuracy")
        plt.plot(history["val_accuracy"], label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training and Validation Accuracy")
        plt.legend()
        plt.show()
