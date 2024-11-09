# models/classifier.py
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model

def build_model(num_classes, input_shape=(224, 224, 3)):
    # Use MobileNetV2 for better feature extraction
    base_model = MobileNetV2(
        weights='imagenet',  # Pre-trained weights from ImageNet
        include_top=False,   # Exclude the top layers
        input_shape=input_shape
    )

    # Fine-tuning: Unfreeze some layers for better adaptation
    for layer in base_model.layers[-40:]:  # Increased number of trainable layers
        layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # Adding dense layers after global average pooling
    x = Dense(1024, kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = tf.nn.relu(x)
    x = Dropout(0.5)(x)

    x = Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = BatchNormalization()(x)
    x = tf.nn.relu(x)
    x = Dropout(0.3)(x)

    # Output layer
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model
