import tensorflow as tf

print(f"Tenosrflow version: {tf.__version__}")
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


class CIFAR10Trainer(tf.keras.Model):
    """
    CIFAR-10 Trainer using Convolutional Neural Networks (CNN) for image classification.

    This class loads the CIFAR-10 dataset, creates a CNN model, trains the model, and evaluates its performance.

    Args:
        num_epochs (int): The number of training epochs.

    Attributes:
        BATCH_SIZE (int): Batch size for training and validation.
        IMAGE_HEIGHT (int): Height of input images.
        IMAGE_WIDTH (int): Width of input images.
        NUM_CLASSES (int): Number of classes in the CIFAR-10 dataset.

    Methods:
        main(): Main entry point to load data, create model, train, evaluate, and plot accuracy.
        load_dataset(batch_size: int = BATCH_SIZE) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
            Load and prepare the CIFAR-10 dataset and return TensorFlow Datasets.
        create_model(): Create a CNN model using Keras Sequential API.
        train_model() -> dict: Train the model for a specified number of epochs.
        evaluate_model(): Evaluate the model on the test dataset and print the test accuracy.
        calculate_validation_accuracy() -> float: Calculate the validation accuracy during training.
        calculate_training_accuracy() -> float: Calculate the training accuracy during training.
        plot_accuracy(history: dict): Plot training and validation accuracy over epochs.
    """

    BATCH_SIZE = 32
    IMAGE_HEIGHT = 32
    IMAGE_WIDTH = 32
    NUM_CLASSES = 10

    def __init__(self, num_epochs):
        super().__init__()
        self.x_train, self.y_train = None, None
        self.x_test, self.y_test = None, None
        (
            self.train_dataset,
            self.validation_dataset,
            self.test_dataset,
        ) = self.load_dataset(batch_size=self.BATCH_SIZE)
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = tf.keras.optimizers.Adam()
        self.num_epochs = num_epochs

    def main(self):
        """
        Main entry point for training the CIFAR-10 classification model.
        Loads data, creates the model, trains, evaluates, and plots accuracy.
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
            batch_size (int): Batch size for training, validation, and testing.

        Returns:
            Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: Datasets for training, validation, and testing.
        """
        # Load and preprocess CIFAR-10 dataset.
        (self.x_train, self.y_train), (
            self.x_test,
            self.y_test,
        ) = tf.keras.datasets.cifar10.load_data()

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
        Create a Convolutional Neural Network (CNN) model using Keras Sequential API.
        """
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Conv2D(
                    32,
                    (3, 3),
                    activation="relu",
                    padding="same",
                    input_shape=(self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 3),
                ),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
                tf.keras.layers.MaxPooling2D((2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(self.NUM_CLASSES),
            ]
        )

        model.compile(
            optimizer=self.optimizer,
            loss=self.loss_fn,
            metrics=["accuracy"],
        )

        self.model = model

    def train_model(self) -> dict:
        """
        Train the CNN model for a specified number of epochs.

        Returns:
            dict: Training history containing accuracy and validation accuracy for each epoch.
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
            history["val_accuracy"].append(validation_accuracy)

            # Calculate and append training accuracy
            training_accuracy = self.calculate_training_accuracy()
            history["accuracy"].append(training_accuracy)

            print(
                f"Epoch {epoch+1}/{self.num_epochs}, Training Accuracy: {training_accuracy:.4f}, Validation Accuracy: {validation_accuracy:.4f}"
            )

        return history

    def evaluate_model(self):
        """
        Evaluate the trained model on the test dataset and print the test accuracy.
        """
        _, test_accuracy = self.model.evaluate(self.test_dataset)
        print(f"Test Accuracy: {test_accuracy:.4f}")

    def calculate_validation_accuracy(self) -> float:
        """
        Calculate the validation accuracy during training.

        Returns:
            float: Validation accuracy.
        """
        correct_predictions = 0
        total_predictions = 0

        for batch_data, batch_labels in self.validation_dataset:
            logits = self.model(batch_data)
            predicted_labels = np.argmax(logits, axis=1)
            correct_predictions += np.sum(predicted_labels == batch_labels)
            total_predictions += batch_labels.shape[0]

        return correct_predictions / total_predictions

    def calculate_training_accuracy(self) -> float:
        """
        Calculate the training accuracy during training.

        Returns:
            float: Training accuracy.
        """
        correct_predictions = 0
        total_predictions = 0

        for batch_data, batch_labels in self.train_dataset:
            logits = self.model(batch_data)
            predicted_labels = np.argmax(logits, axis=1)
            correct_predictions += np.sum(predicted_labels == batch_labels)
            total_predictions += batch_labels.shape[0]

        return correct_predictions / total_predictions

    def plot_accuracy(self, history: dict):
        """
        Plot training and validation accuracy over epochs.

        Args:
            history (dict): Training history containing accuracy and validation accuracy for each epoch.
        """
        plt.figure(figsize=(8, 6))
        plt.plot(history["val_accuracy"], label="Validation Accuracy")
        plt.plot(history["accuracy"], linestyle="--", label="Training Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training and Validation Accuracy")
        plt.legend()
        plt.show()
