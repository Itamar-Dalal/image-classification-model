# Image Classification Model

This Python 3.11 program is designed for image classification using TensorFlow. It utilizes the CIFAR-10 dataset, containing 60,000 images across ten classes. The program leverages Convolutional Neural Networks (CNNs) and the Adam optimizer for efficient model training. Users can customize the number of training epochs. The program also provides a straightforward evaluation process to measure the model's test accuracy. Furthermore, it includes a tool for visualizing training and validation accuracy over epochs, aiding in monitoring the model's performance.

## Requirements

To run this project, make sure you have the following prerequisites installed on your system:

- **Docker**: You need Docker installed to containerize and run this application. You can download and install Docker from [Docker's official website](https://www.docker.com/get-started).

## Usage
### Running with Docker container 

To run this project using Docker, follow these steps:

1. **Clone the Repository**: Clone this repository to your local machine.

   ```shell
   git clone https://github.com/Itamar-Dalal/image-classification.git

2. **Build the Docker Image**: Navigate to the project directory and build the Docker image using the provided Dockerfile.

   ```shell
    cd image-classification
    docker build -t image-classification-app .

3. **Run the Docker Container**: Start a Docker container from the image, specifying the number of training epochs as an environment variable. <br/>
***example***:
   ```shell
   docker run -p 80:80 -v /path/to/your/local/project/directory:/app -e NUM_EPOCHS=2 image-classification-app
Replace /path/to/your/local/project/directory with the actual path to the project directory on your machine.
Adjust the NUM_EPOCHS environment variable to set the number of training epochs (2 to 20).

4. **Access the results**: Once the program is finished, you can access the results by navigating to your project directory and opening the `plot.png`.

### Running Directly with Python

If you prefer not to use Docker, you can run the project directly using Python. Follow these steps:

1. **Clone the Repository**: Clone this repository to your local machine.

   ```shell
   git clone https://github.com/Itamar-Dalal/image-classification.git
2. **Navigate to the Project Directory**: Change your working directory to the project directory.

    ```shell
    cd (project directory)
3. **Run the Model**: Run the image classification model directly using Python.

    ```shell
    python model.py
You can specify the number of training epochs interactively when prompted.

## Acknowledgments

I would like to acknowledge the following libraries, tools, and resources that were instrumental in the development of this project:

- [TensorFlow](https://www.tensorflow.org/): TensorFlow is an open-source machine learning framework that forms the backbone of our image classification model. It provides the tools and infrastructure for training and deploying machine learning models efficiently.

- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html): The CIFAR-10 dataset, consisting of 60,000 images across ten classes, serves as the foundation for our image classification model. This dataset is widely used for benchmarking image classification algorithms.

- [Docker](https://www.docker.com/): Docker provides containerization capabilities, enabling us to package our application and its dependencies into a standardized environment. This greatly simplifies deployment and ensures consistency across different systems.

- [Keras](https://keras.io/): Keras is a high-level neural networks API that works seamlessly with TensorFlow. We use Keras to build and compile our Convolutional Neural Network (CNN) model efficiently.

- [Matplotlib](https://matplotlib.org/): Matplotlib is a powerful library for creating visualizations in Python. It is utilized to generate informative plots that help us monitor the training and validation accuracy of our model.



The model uses CNN (Convolutional Neural Network):
```python
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
```
These are the layers used in CNN:
<image src="https://miro.medium.com/max/2055/1*uAeANQIOQPqWZnnuH-VEyw.jpeg"/>
The convolution operation:
<image src="https://www.askpython.com/wp-content/uploads/2023/02/conv.webp"/>
