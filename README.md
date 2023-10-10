# image_generator
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
