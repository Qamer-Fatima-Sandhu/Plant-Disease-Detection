"""
PlantDiseaseNet - Custom CNN Model for Plant Disease Classification

This module implements a custom CNN architecture designed for classifying
plant diseases from leaf images. The model uses a hierarchical feature
extraction approach with progressive filter increase.

Architecture:
- 4 Convolutional blocks (32→64→128→256 filters)
- Batch Normalization for training stability
- Dropout for regularization
- Dense classification head
- 38-class output (PlantVillage dataset)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models


def build_plantdiseasenet(input_shape=(224, 224, 3), num_classes=38, config=None):
    """
    Build the PlantDiseaseNet CNN model.

    Args:
        input_shape (tuple): Shape of input images (height, width, channels)
        num_classes (int): Number of output classes
        config (dict): Optional configuration dictionary

    Returns:
        keras.Model: Compiled Keras model
    """

    # Input layer
    inputs = layers.Input(shape=input_shape, name='input_layer')
    x = inputs

    # ============================================================
    # Block 1: Low-level features (edges, colors, textures)
    # ============================================================
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu', name='block1_conv1')(x)
    x = layers.BatchNormalization(name='block1_bn1')(x)
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu', name='block1_conv2')(x)
    x = layers.BatchNormalization(name='block1_bn2')(x)
    x = layers.MaxPooling2D((2, 2), name='block1_pool')(x)
    x = layers.Dropout(0.25, name='block1_dropout')(x)

    # ============================================================
    # Block 2: Mid-level features (patterns, shapes)
    # ============================================================
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='block2_conv1')(x)
    x = layers.BatchNormalization(name='block2_bn1')(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='block2_conv2')(x)
    x = layers.BatchNormalization(name='block2_bn2')(x)
    x = layers.MaxPooling2D((2, 2), name='block2_pool')(x)
    x = layers.Dropout(0.25, name='block2_dropout')(x)

    # ============================================================
    # Block 3: High-level features (complex patterns)
    # ============================================================
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='block3_conv1')(x)
    x = layers.BatchNormalization(name='block3_bn1')(x)
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='block3_conv2')(x)
    x = layers.BatchNormalization(name='block3_bn2')(x)
    x = layers.MaxPooling2D((2, 2), name='block3_pool')(x)
    x = layers.Dropout(0.3, name='block3_dropout')(x)

    # ============================================================
    # Block 4: Deep features (disease-specific patterns)
    # ============================================================
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='block4_conv1')(x)
    x = layers.BatchNormalization(name='block4_bn1')(x)
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu', name='block4_conv2')(x)
    x = layers.BatchNormalization(name='block4_bn2')(x)
    x = layers.MaxPooling2D((2, 2), name='block4_pool')(x)
    x = layers.Dropout(0.3, name='block4_dropout')(x)

    # ============================================================
    # Classification Head
    # ============================================================
    x = layers.Flatten(name='flatten')(x)

    # First dense layer
    x = layers.Dense(512, activation='relu', name='fc1')(x)
    x = layers.BatchNormalization(name='fc1_bn')(x)
    x = layers.Dropout(0.5, name='fc1_dropout')(x)

    # Second dense layer
    x = layers.Dense(256, activation='relu', name='fc2')(x)
    x = layers.BatchNormalization(name='fc2_bn')(x)
    x = layers.Dropout(0.5, name='fc2_dropout')(x)

    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)

    # Create model
    model = models.Model(inputs=inputs, outputs=outputs, name='PlantDiseaseNet')

    return model


def get_model_summary(model):
    """
    Print a detailed summary of the model architecture.

    Args:
        model (keras.Model): The model to summarize
    """
    print("=" * 80)
    print(f"Model: {model.name}")
    print("=" * 80)
    model.summary()
    print("=" * 80)

    # Count parameters
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = sum([tf.size(w).numpy() for w in model.non_trainable_weights])
    total_params = trainable_params + non_trainable_params

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    print("=" * 80)


if __name__ == "__main__":
    # Test model creation
    print("Building PlantDiseaseNet...")
    model = build_plantdiseasenet(input_shape=(224, 224, 3), num_classes=38)

    # Print summary
    get_model_summary(model)

    # Test forward pass
    import numpy as np
    print("\nTesting forward pass...")
    dummy_input = np.random.rand(1, 224, 224, 3).astype('float32')
    output = model.predict(dummy_input, verbose=0)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output sum (should be ~1.0): {output.sum():.6f}")
    print("\nModel test successful!")
