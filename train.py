import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from models.classifier import build_model
from config import Config

def create_data_generators(data_dir, image_size, batch_size):
    # Data augmentation for training data
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.3,
        height_shift_range=0.3,
        shear_range=0.3,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2  # Added validation split here for train/validation split
    )

    # Train and validation generators
    train_generator = train_datagen.flow_from_directory(
        directory=data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        subset='training'
    )

    val_generator = train_datagen.flow_from_directory(
        directory=data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False,
        subset='validation'
    )

    return train_generator, val_generator

def train():
    # Set random seed for reproducibility
    np.random.seed(Config.RANDOM_SEED)
    tf.random.set_seed(Config.RANDOM_SEED)

    # Create data generators
    train_generator, val_generator = create_data_generators(Config.DATA_DIR, Config.IMAGE_SIZE, Config.BATCH_SIZE)

    # Build model
    model = build_model(num_classes=Config.NUM_CLASSES, input_shape=Config.IMAGE_SIZE + (3,))

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Callbacks for early stopping, model checkpoint, and learning rate reduction
    early_stopping = EarlyStopping(patience=Config.EARLY_STOPPING_PATIENCE, restore_best_weights=True, monitor='val_loss')
    model_checkpoint = ModelCheckpoint(Config.MODEL_PATH, save_best_only=True, monitor='val_loss')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

    # Train the model
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=Config.EPOCHS,
        callbacks=[early_stopping, model_checkpoint, reduce_lr],
        verbose=1
    )

    return history, model

if __name__ == "__main__":
    history, model = train()
