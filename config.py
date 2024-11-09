# config.py
import os
class Config:
    NUM_CLASSES = 39  # Number of plant disease classes
    IMAGE_SIZE = (224, 224)  # Image size to resize images to
    IMAGE_HEIGHT = IMAGE_WIDTH = 224
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 1e-4
    EARLY_STOPPING_PATIENCE = 5
    MODEL_PATH = 'best_model.h5'  # Where to save the best model
    DATA_DIR = 'C:/Users/YAXITA/plant-disease-detection/data/Plant_leave_diseases_dataset_without_augmentation'  # Path to data
    RANDOM_SEED = 42  # Seed for reproducibility
