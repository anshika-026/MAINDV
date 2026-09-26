import tensorflow as tf
import numpy as np

from tensorflow.keras import layers, models
from sklearn.utils.class_weight import compute_class_weight


# ==========================================
# 1. DATASET PATHS
# ==========================================

train_path = "dataset/train"
val_path = "dataset/val"


# ==========================================
# 2. SETTINGS
# ==========================================

image_size = (48, 48)

batch_size = 32

epochs = 30


# ==========================================
# 3. LOAD DATASETS
# ==========================================

train_dataset = tf.keras.utils.image_dataset_from_directory(

    train_path,

    image_size=image_size,

    batch_size=batch_size,

    color_mode="grayscale",

    shuffle=True
)


val_dataset = tf.keras.utils.image_dataset_from_directory(

    val_path,

    image_size=image_size,

    batch_size=batch_size,

    color_mode="grayscale",

    shuffle=False
)


# Get class names

class_names = train_dataset.class_names

print("\nClass Names:")

print(class_names)


# ==========================================
# 4. NORMALIZE IMAGES
# ==========================================

normalization_layer = layers.Rescaling(1.0 / 255)


train_dataset = train_dataset.map(

    lambda images, labels: (
        normalization_layer(images),
        labels
    )
)


val_dataset = val_dataset.map(

    lambda images, labels: (
        normalization_layer(images),
        labels
    )
)


# ==========================================
# 5. DATA AUGMENTATION
# ==========================================

data_augmentation = tf.keras.Sequential([

    layers.RandomRotation(0.05),

    layers.RandomZoom(0.1),

    layers.RandomTranslation(
        height_factor=0.05,
        width_factor=0.05
    )

])


# ==========================================
# 6. CALCULATE CLASS WEIGHTS
# ==========================================

train_labels = []


for images, labels in train_dataset:

    train_labels.extend(
        labels.numpy()
    )


train_labels = np.array(
    train_labels
)


classes = np.unique(
    train_labels
)


weights = compute_class_weight(

    class_weight="balanced",

    classes=classes,

    y=train_labels
)


class_weights = dict(

    zip(
        classes,
        weights
    )
)


print("\nClass Weights:")

for class_index, weight in class_weights.items():

    print(

        class_names[class_index],

        ":",

        round(weight, 2)

    )


# ==========================================
# 7. CREATE CNN MODEL
# ==========================================

model = models.Sequential([


    # Input
    layers.Input(
        shape=(48, 48, 1)
    ),


    # Data Augmentation
    data_augmentation,


    # First CNN Block
    layers.Conv2D(

        32,

        (3, 3),

        activation="relu"

    ),

    layers.MaxPooling2D(
        (2, 2)
    ),


    # Second CNN Block
    layers.Conv2D(

        64,

        (3, 3),

        activation="relu"

    ),

    layers.MaxPooling2D(
        (2, 2)
    ),


    # Third CNN Block
    layers.Conv2D(

        128,

        (3, 3),

        activation="relu"

    ),

    layers.MaxPooling2D(
        (2, 2)
    ),


    # Convert to vector
    layers.Flatten(),


    # Dense Layer
    layers.Dense(

        128,

        activation="relu"

    ),


    # Prevent Overfitting
    layers.Dropout(
        0.5
    ),


    # Output Layer
    layers.Dense(

        7,

        activation="softmax"

    )

])


# ==========================================
# 8. COMPILE MODEL
# ==========================================

model.compile(

    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)


# ==========================================
# 9. MODEL SUMMARY
# ==========================================

model.summary()


# ==========================================
# 10. CALLBACKS
# ==========================================

early_stopping = tf.keras.callbacks.EarlyStopping(

    monitor="val_loss",

    patience=5,

    restore_best_weights=True

)


# ==========================================
# 11. TRAIN MODEL
# ==========================================

history = model.fit(

    train_dataset,

    validation_data=val_dataset,

    epochs=epochs,

    class_weight=class_weights,

    callbacks=[

        early_stopping

    ]

)


# ==========================================
# 12. SAVE MODEL
# ==========================================

model.save(

    "models/emotion_model.keras"

)


print(

    "\nModel saved successfully!"

)