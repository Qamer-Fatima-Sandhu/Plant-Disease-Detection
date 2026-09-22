"""
Training Script for PlantDiseaseNet

This script handles the complete training pipeline:
- Load configuration
- Create data generators
- Build model
- Configure callbacks
- Train model
- Save training history and results
"""

import os
import sys
import json
import yaml
import datetime
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers, metrics
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
    CSVLogger,
    TensorBoard
)

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.cnn_model import build_plantdiseasenet, get_model_summary
from data.data_loader import PlantDiseaseDataLoader


def setup_gpu():
    """Configure GPU settings for optimal performance on Mac M4."""
    print("=" * 80)
    print("GPU Configuration")
    print("=" * 80)

    # Check for GPU availability
    physical_devices = tf.config.list_physical_devices('GPU')

    if physical_devices:
        print(f"✓ GPU detected: {len(physical_devices)} device(s)")
        for device in physical_devices:
            print(f"  - {device}")

        # Enable memory growth to prevent OOM errors
        try:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            print("✓ Memory growth enabled")
        except RuntimeError as e:
            print(f"⚠ Could not enable memory growth: {e}")
    else:
        print("⚠ No GPU detected. Training will use CPU (slower).")
        print("  For Mac M4, ensure tensorflow-metal is installed:")
        print("  pip install tensorflow-macos tensorflow-metal")

    print("=" * 80)


def create_callbacks(config):
    """
    Create training callbacks based on configuration.

    Args:
        config (dict): Configuration dictionary

    Returns:
        list: List of Keras callbacks
    """
    callback_config = config['training']['callbacks']
    paths = config['paths']

    # Create directories
    os.makedirs(paths['models'], exist_ok=True)
    os.makedirs(paths['logs'], exist_ok=True)
    os.makedirs(paths['checkpoints'], exist_ok=True)

    callbacks = []

    # ModelCheckpoint - Save best model
    checkpoint_path = callback_config['model_checkpoint']['filepath']
    checkpoint = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor=callback_config['model_checkpoint']['monitor'],
        mode=callback_config['model_checkpoint']['mode'],
        save_best_only=callback_config['model_checkpoint']['save_best_only'],
        verbose=callback_config['model_checkpoint']['verbose']
    )
    callbacks.append(checkpoint)
    print(f"✓ ModelCheckpoint: Saving best model to {checkpoint_path}")

    # EarlyStopping - Stop training when no improvement
    early_stop = EarlyStopping(
        monitor=callback_config['early_stopping']['monitor'],
        patience=callback_config['early_stopping']['patience'],
        restore_best_weights=callback_config['early_stopping']['restore_best_weights'],
        verbose=callback_config['early_stopping']['verbose']
    )
    callbacks.append(early_stop)
    print(f"✓ EarlyStopping: Patience = {callback_config['early_stopping']['patience']} epochs")

    # ReduceLROnPlateau - Reduce learning rate when plateaued
    reduce_lr = ReduceLROnPlateau(
        monitor=callback_config['reduce_lr']['monitor'],
        factor=callback_config['reduce_lr']['factor'],
        patience=callback_config['reduce_lr']['patience'],
        min_lr=callback_config['reduce_lr']['min_lr'],
        verbose=callback_config['reduce_lr']['verbose']
    )
    callbacks.append(reduce_lr)
    print(f"✓ ReduceLROnPlateau: Factor = {callback_config['reduce_lr']['factor']}")

    # CSVLogger - Log training metrics to CSV
    csv_path = callback_config['csv_logger']['filename']
    csv_logger = CSVLogger(
        filename=csv_path,
        append=callback_config['csv_logger']['append']
    )
    callbacks.append(csv_logger)
    print(f"✓ CSVLogger: Logging to {csv_path}")

    # TensorBoard - Visualization
    log_dir = os.path.join(
        callback_config['tensorboard']['log_dir'],
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    tensorboard = TensorBoard(
        log_dir=log_dir,
        histogram_freq=callback_config['tensorboard']['histogram_freq'],
        write_graph=callback_config['tensorboard']['write_graph']
    )
    callbacks.append(tensorboard)
    print(f"✓ TensorBoard: Logging to {log_dir}")

    return callbacks


def compile_model(model, config):
    """
    Compile the model with optimizer, loss, and metrics.

    Args:
        model (keras.Model): The model to compile
        config (dict): Configuration dictionary

    Returns:
        keras.Model: Compiled model
    """
    training_config = config['training']

    # Create optimizer
    optimizer = optimizers.Adam(
        learning_rate=training_config['initial_lr'],
        beta_1=training_config['optimizer_params']['beta_1'],
        beta_2=training_config['optimizer_params']['beta_2'],
        epsilon=training_config['optimizer_params']['epsilon']
    )

    # Define metrics
    model_metrics = [
        'accuracy',
        metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')
    ]

    # Compile model
    model.compile(
        optimizer=optimizer,
        loss=training_config['loss'],
        metrics=model_metrics
    )

    print("=" * 80)
    print("Model Compilation")
    print("=" * 80)
    print(f"✓ Optimizer: {training_config['optimizer']}")
    print(f"✓ Learning Rate: {training_config['initial_lr']}")
    print(f"✓ Loss: {training_config['loss']}")
    print(f"✓ Metrics: {[m if isinstance(m, str) else m.name for m in model_metrics]}")
    print("=" * 80)

    return model


def train_model(model, train_generator, val_generator, callbacks, config):
    """
    Train the model.

    Args:
        model (keras.Model): Compiled model
        train_generator: Training data generator
        val_generator: Validation data generator
        callbacks (list): List of callbacks
        config (dict): Configuration dictionary

    Returns:
        keras.callbacks.History: Training history
    """
    training_config = config['training']

    print("=" * 80)
    print("Starting Training")
    print("=" * 80)
    print(f"Epochs: {training_config['epochs']}")
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {val_generator.samples}")
    print(f"Steps per epoch: {train_generator.samples // train_generator.batch_size}")
    print("=" * 80)

    # Train model
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=training_config['epochs'],
        callbacks=callbacks,
        verbose=1
    )

    return history


def save_training_history(history, config):
    """
    Save training history to JSON file.

    Args:
        history (keras.callbacks.History): Training history
        config (dict): Configuration dictionary
    """
    history_dict = {
        'loss': [float(x) for x in history.history['loss']],
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']]
    }

    # Add top-3 accuracy if available
    if 'top_3_accuracy' in history.history:
        history_dict['top_3_accuracy'] = [float(x) for x in history.history['top_3_accuracy']]
        history_dict['val_top_3_accuracy'] = [float(x) for x in history.history['val_top_3_accuracy']]

    history_path = 'models/saved_models/training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)

    print(f"✓ Training history saved to: {history_path}")


def print_training_summary(history):
    """
    Print summary of training results.

    Args:
        history (keras.callbacks.History): Training history
    """
    print("=" * 80)
    print("Training Complete!")
    print("=" * 80)

    final_epoch = len(history.history['loss'])
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]

    best_val_acc = max(history.history['val_accuracy'])
    best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1

    print(f"Total epochs trained: {final_epoch}")
    print(f"\nFinal Epoch Results:")
    print(f"  Training Accuracy:   {final_train_acc:.4f}")
    print(f"  Validation Accuracy: {final_val_acc:.4f}")
    print(f"  Training Loss:       {final_train_loss:.4f}")
    print(f"  Validation Loss:     {final_val_loss:.4f}")
    print(f"\nBest Results:")
    print(f"  Best Val Accuracy: {best_val_acc:.4f} (Epoch {best_epoch})")

    if 'top_3_accuracy' in history.history:
        final_top3 = history.history['val_top_3_accuracy'][-1]
        print(f"  Final Top-3 Accuracy: {final_top3:.4f}")

    print("=" * 80)


def main():
    """Main training pipeline."""
    # Load configuration
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    print("=" * 80)
    print("PLANT DISEASE DETECTION - TRAINING PIPELINE")
    print("=" * 80)

    # Setup GPU
    setup_gpu()

    # Create data loader
    print("\n" + "=" * 80)
    print("Loading Data")
    print("=" * 80)
    data_loader = PlantDiseaseDataLoader('config.yaml')

    # Create data generators
    train_gen, val_gen, test_gen, class_mapping = data_loader.get_data_generators()

    # Build model
    print("\n" + "=" * 80)
    print("Building Model")
    print("=" * 80)
    model = build_plantdiseasenet(
        input_shape=tuple(config['model']['input_shape']),
        num_classes=config['model']['num_classes']
    )

    # Print model summary
    get_model_summary(model)

    # Compile model
    model = compile_model(model, config)

    # Create callbacks
    print("\n" + "=" * 80)
    print("Configuring Callbacks")
    print("=" * 80)
    callbacks = create_callbacks(config)

    # Train model
    print("\n")
    history = train_model(model, train_gen, val_gen, callbacks, config)

    # Save training history
    save_training_history(history, config)

    # Print summary
    print("\n")
    print_training_summary(history)

    print("\n✓ Model saved to: models/saved_models/best_model.keras")
    print("✓ Run 'tensorboard --logdir=logs/fit' to visualize training")
    print("\nNext steps:")
    print("  1. Evaluate model on test set")
    print("  2. Build Streamlit web application")
    print("  3. Test with sample images")


if __name__ == "__main__":
    main()
