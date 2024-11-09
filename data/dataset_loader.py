#data_loader.py
# data/dataset_loader.py
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class DataLoader:
    def __init__(self, config):
        self.config = config
        
    def get_data_generators(self):
        # Enhanced data augmentation
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=0.2,
            brightness_range=[0.8, 1.2]
        )
        
        valid_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2
        )
        
        # Training data generator
        train_generator = train_datagen.flow_from_directory(
            self.config.DATA_DIR,
            target_size=(self.config.IMAGE_HEIGHT, self.config.IMAGE_WIDTH),
            batch_size=self.config.BATCH_SIZE,
            class_mode='categorical',
            subset='training',  # Training subset
            shuffle=True
        )
        
        # Validation data generator
        val_generator = valid_datagen.flow_from_directory(
            self.config.DATA_DIR,
            target_size=(self.config.IMAGE_HEIGHT, self.config.IMAGE_WIDTH),
            batch_size=self.config.BATCH_SIZE,
            class_mode='categorical',
            subset='validation',  # Validation subset
            shuffle=False  # Shuffle is typically set to False for validation data
        )
        
        return train_generator, val_generator
