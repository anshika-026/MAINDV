import tensorflow as tf

from tensorflow.keras import layers, models


# ==========================================
# 1. SETTINGS
# ==========================================

train_path = "dataset/train"
val_path = "dataset/val"

image_size = (48, 48)
batch_size = 32
epochs = 30


# ==========================================
# 2. LOAD DATASETS
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


class_names = train_dataset.class_names

print("\nClass Names:")
print(class_names)


# ==========================================
# 3. NORMALIZE DATA
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
# 4. PREFETCH
# ==========================================

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(AUTOTUNE)

val_dataset = val_dataset.prefetch(AUTOTUNE)


# ==========================================
# 5. DATA AUGMENTATION
# ==========================================

data_augmentation = models.Sequential([

    layers.RandomRotation(0.08),

    layers.RandomZoom(0.1),

    layers.RandomTranslation(
        height_factor=0.05,
        width_factor=0.05
    ),

    layers.RandomFlip("horizontal")

])


# ==========================================
# 6. MODEL V3
# ==========================================

model = models.Sequential([

    layers.Input(shape=(48, 48, 1)),


    # Data augmentation
    data_augmentation,


    # ======================================
    # CNN BLOCK 1
    # ======================================

    layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.BatchNormalization(),

    layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.MaxPooling2D((2, 2)),

    layers.Dropout(0.25),


    # ======================================
    # CNN BLOCK 2
    # ======================================

    layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.BatchNormalization(),

    layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.MaxPooling2D((2, 2)),

    layers.Dropout(0.25),


    # ======================================
    # CNN BLOCK 3
    # ======================================

    layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.BatchNormalization(),

    layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.MaxPooling2D((2, 2)),

    layers.Dropout(0.25),


    # Feature reduction
    layers.GlobalAveragePooling2D(),


    # Dense layer
    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(0.5),


    # Output
    layers.Dense(
        7,
        activation="softmax"
    )

])


# ==========================================
# 7. COMPILE MODEL
# ==========================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]

)


# ==========================================
# 8. CALLBACKS
# ==========================================

early_stopping = tf.keras.callbacks.EarlyStopping(

    monitor="val_loss",

    patience=7,

    restore_best_weights=True
)


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=3,

    min_lr=0.00001
)


# ==========================================
# 9. CUSTOM CLASS WEIGHTS
# ==========================================

class_weights = {

    0: 1.0,   # Angry
    1: 2.5,   # Disgust
    2: 1.0,   # Fear
    3: 1.0,   # Happy
    4: 1.0,   # Neutral
    5: 1.0,   # Sad
    6: 1.0    # Surprise

}


print("\nClass Weights:")
print(class_weights)


# ==========================================
# 10. TRAIN MODEL
# ==========================================

history = model.fit(

    train_dataset,

    validation_data=val_dataset,

    epochs=epochs,

    class_weight=class_weights,

    callbacks=[
        early_stopping,
        reduce_lr
    ]

)


# ==========================================
# 11. SAVE MODEL
# ==========================================

model.save(
    "models/emotion_model_v3.keras"
)


print("\nModel V3 saved successfully!")