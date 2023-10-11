import tensorflow as tf

print(f"Tensorflow version: {tf.__version__}")

from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


class CIFAR10Trainer(keras.Model):
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
        create_model() -> None: Create a CNN model using Keras Sequential API.
        train_model() -> dict: Train the model for a specified number of epochs.
        evaluate_model() -> None: Evaluate the model on the test dataset and print the test accuracy.
        calculate_validation_accuracy() -> float: Calculate the validation accuracy during training.
        calculate_training_accuracy() -> float: Calculate the training accuracy during training.
        plot_accuracy(history: dict) -> None: Plot training and validation accuracy over epochs.
    """

    BATCH_SIZE: int = 32
    IMAGE_HEIGHT: int = 32
    IMAGE_WIDTH: int = 32
    NUM_CLASSES: int = 10

    def __init__(self, num_epochs: int):
        try:
            super().__init__()
            if not (1 <= num_epochs <= 20):
                raise ValueError("num_epochs must be between 1 and 20")
            self.num_epochs: int = num_epochs
            self.x_train: np.ndarray = None
            self.y_train: np.ndarray = None
            self.x_test: np.ndarray = None
            self.y_test: np.ndarray = None
            (
                self.train_dataset,
                self.validation_dataset,
                self.test_dataset,
            ) = self.load_dataset(batch_size=self.BATCH_SIZE)
            self.loss_fn: keras.losses.Loss = (
                tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            )
            self.optimizer: keras.optimizers.Optimizer = tf.keras.optimizers.Adam()
        except Exception as e:
            print(f"An error occurred during initialization: {str(e)}")

    def main(self) -> None:
        try:
            """
            Main entry point for training the CIFAR-10 classification model.
            Loads data, creates the model, trains, evaluates, and plots accuracy.
            """
            self.load_dataset()
            self.create_model()
            self.train_model()
            self.evaluate_model()
            self.plot_accuracy()
        except Exception as e:
            print(
                f"An error occurred in the main function of CIFAR10Trainer class: {str(e)}"
            )

    @staticmethod
    def load_dataset(
        batch_size: int = BATCH_SIZE,
    ) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        try:
            """
            Load and prepare the CIFAR-10 dataset.

            Args:
                batch_size (int): Batch size for training, validation, and testing.

            Returns:
                Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]: Datasets for training, validation, and testing.
            """
            # Load and preprocess CIFAR-10 dataset.
            (x_train, y_train), (
                x_test,
                y_test,
            ) = keras.datasets.cifar10.load_data()

            # Normalize pixel values to the range [0, 1].
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
            train_dataset = tf.data.Dataset.from_tensor_slices(
                (x_train, y_train)
            ).batch(batch_size)
            validation_dataset = tf.data.Dataset.from_tensor_slices(
                (x_validation, y_validation)
            ).batch(batch_size)
            test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(
                batch_size
            )

            return train_dataset, validation_dataset, test_dataset
        except Exception as e:
            print(f"An error occurred in the load_dataset function: {str(e)}")

    def create_model(self) -> None:
        try:
            """
            Create a Convolutional Neural Network (CNN) model using Keras Sequential API.
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
        except Exception as e:
            print(f"An error occurred in the create_model function: {str(e)}")

    def train_model(self) -> dict:
        try:
            """
            Train the CNN model for a specified number of epochs.

            Returns:
                dict: Training history containing accuracy and validation accuracy for each epoch.
            """
            self.history = {"accuracy": [], "val_accuracy": []}
            for epoch in range(self.num_epochs):
                for batch_data, batch_labels in self.train_dataset:
                    with tf.GradientTape() as tape:
                        logits = self.model(batch_data)
                        loss_value = self.loss_fn(batch_labels, logits)

                    gradients = tape.gradient(
                        loss_value, self.model.trainable_variables
                    )
                    self.optimizer.apply_gradients(
                        zip(gradients, self.model.trainable_variables)
                    )

                # Calculate and append validation accuracy
                validation_accuracy = self.calculate_validation_accuracy()
                self.history["val_accuracy"].append(validation_accuracy)

                # Calculate and append training accuracy
                training_accuracy = self.calculate_training_accuracy()
                self.history["accuracy"].append(training_accuracy)

                print(
                    f"Epoch {epoch + 1}/{self.num_epochs}, Training Accuracy: {training_accuracy:.4f}, Validation Accuracy: {validation_accuracy:.4f}"
                )

        except Exception as e:
            print(f"An error occurred in the train_model function: {str(e)}")

    def evaluate_model(self) -> None:
        try:
            """
            Evaluate the trained model on the test dataset and print the test accuracy.
            """
            _, test_accuracy = self.model.evaluate(self.test_dataset)
            print(f"Test Accuracy: {test_accuracy:.4f}")
        except Exception as e:
            print(f"An error occurred in the evaluate_model function: {str(e)}")

    def calculate_validation_accuracy(self) -> float:
        try:
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
        except Exception as e:
            print(
                f"An error occurred in the calculate_validation_accuracy function: {str(e)}"
            )

    def calculate_training_accuracy(self) -> float:
        try:
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
        except Exception as e:
            print(
                f"An error occurred in the calculate_training_accuracy function: {str(e)}"
            )

    def plot_accuracy(self) -> None:
        try:
            """
            Plot training and validation accuracy over epochs.

            Args:
                history (dict): Training history containing accuracy and validation accuracy for each epoch.
            """
            plt.figure(figsize=(8, 6))
            x_values = range(1, len(self.history["val_accuracy"]) + 1)
            plt.plot(
                x_values, self.history["val_accuracy"], label="Validation Accuracy"
            )
            plt.plot(
                x_values,
                self.history["accuracy"],
                linestyle="--",
                label="Training Accuracy",
            )
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title("Model Training and Validation Accuracy")
            plt.grid(True)
            plt.legend()
            plt.show()
            plt.savefig("/app/plot.png")
        except Exception as e:
            print(f"An error occurred in the plot_accuracy function: {str(e)}")


if __name__ == "__main__":
    try:
        num_epochs: int = input("Enter number of epoches: ")
        while not (1 <= num_epochs <= 20):
            print("Value Error: number of epoches must be between 1 and 20.")
            num_epochs = input("Enter number of epoches: ")

        model = CIFAR10Trainer(num_epochs)
        model.main()

    except ValueError as ve:
        print(f"Error: {str(ve)}")
    except Exception as e:
        print(f"An error occurred in the main function of main.py: {str(e)}")
