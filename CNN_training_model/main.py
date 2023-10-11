from model import CIFAR10Trainer
import os


def main() -> None:
    try:
        num_epochs = int(os.environ.get("NUM_EPOCHS", 10))
        if not (1 <= num_epochs <= 20):
            raise ValueError("Number of epochs must be between 1 and 20.")

        model = CIFAR10Trainer(num_epochs)
        model.main()

    except ValueError as ve:
        print(f"Error: {str(ve)}")
    except Exception as e:
        print(f"An error occurred in the main function of main.py: {str(e)}")


if __name__ == "__main__":
    main()
