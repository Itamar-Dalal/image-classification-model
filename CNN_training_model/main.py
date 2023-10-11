from model import CIFAR10Trainer


def main() -> None:
    try:
        num_epochs: int = 2
        model: CIFAR10Trainer = CIFAR10Trainer(num_epochs)
        model.main()
    except Exception as e:
        print(f"An error occurred in the main function of main.py: {str(e)}")


if __name__ == "__main__":
    main()
