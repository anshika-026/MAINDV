import tensorflow as tf

from tensorflow.keras import layers, models


# Create CNN model

model = models.Sequential([

    # Input layer
    layers.Input(shape=(48, 48, 1)),


    # First CNN block
    layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),


    # Second CNN block
    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),


    # Third CNN block
    layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),


    # Convert features into one vector
    layers.Flatten(),


    # Dense layer
    layers.Dense(
        128,
        activation="relu"
    ),


    # Prevent overfitting
    layers.Dropout(
        0.5
    ),


    # Output layer
    layers.Dense(
        7,
        activation="softmax"
    )

])


# Print model structure

model.summary()