from model import CIFAR10Trainer

if __name__ == "__main__":
    num_epochs = 5
    model: CIFAR10Trainer = CIFAR10Trainer(num_epochs)
    model.main()