"""
Data Loading and Preprocessing Module

This module handles:
- Downloading PlantVillage dataset from Kaggle
- Creating stratified train/validation/test splits
- Data augmentation for training
- Batch generation using Keras ImageDataGenerator
- Class mapping creation and management
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tqdm import tqdm
import yaml


class PlantDiseaseDataLoader:
    """Handles data loading, splitting, and augmentation for plant disease detection."""

    def __init__(self, config_path='config.yaml'):
        """
        Initialize the data loader.

        Args:
            config_path (str): Path to configuration YAML file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.data_config = self.config['data']
        self.aug_config = self.config['augmentation']
        self.kaggle_config = self.config['kaggle']

        self.dataset_path = self.data_config['dataset_path']
        self.image_size = tuple(self.data_config['image_size'])
        self.batch_size = self.data_config['batch_size']
        self.num_classes = self.data_config['num_classes']
        self.train_split = self.data_config['train_split']
        self.val_split = self.data_config['val_split']
        self.test_split = self.data_config['test_split']
        self.random_seed = self.data_config['random_seed']

    def download_dataset(self):
        """Download PlantVillage dataset from Kaggle using Kaggle API."""
        import kaggle

        print("=" * 80)
        print("Downloading PlantVillage Dataset from Kaggle...")
        print("=" * 80)

        # Create raw data directory
        os.makedirs(self.kaggle_config['download_path'], exist_ok=True)

        try:
            # Download and extract dataset
            kaggle.api.dataset_download_files(
                self.kaggle_config['dataset'],
                path=self.kaggle_config['download_path'],
                unzip=True
            )
            print(f"✓ Dataset downloaded successfully to: {self.kaggle_config['download_path']}")
            print("=" * 80)
            return True

        except Exception as e:
            print(f"✗ Error downloading dataset: {e}")
            print("\nPlease ensure:")
            print("1. Kaggle API is installed: pip install kaggle")
            print("2. Kaggle API token is configured: ~/.kaggle/kaggle.json")
            print("3. Token has correct permissions: chmod 600 ~/.kaggle/kaggle.json")
            print("=" * 80)
            return False

    def get_class_mapping(self):
        """
        Create mapping from class indices to class names.

        Returns:
            dict: Mapping of class index to class name
        """
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Dataset not found at: {self.dataset_path}")

        # Get all class directories
        class_dirs = sorted([d for d in os.listdir(self.dataset_path)
                            if os.path.isdir(os.path.join(self.dataset_path, d)) and not d.startswith('.')])

        # Create mapping
        class_mapping = {i: class_name for i, class_name in enumerate(class_dirs)}

        # Save mapping
        mapping_path = 'data/splits/class_mapping.json'
        os.makedirs(os.path.dirname(mapping_path), exist_ok=True)

        with open(mapping_path, 'w') as f:
            json.dump(class_mapping, f, indent=2)

        print(f"✓ Class mapping saved to: {mapping_path}")
        print(f"✓ Total classes: {len(class_mapping)}")

        return class_mapping

    def get_data_generators(self):
        """
        Create training, validation, and test data generators.

        Returns:
            tuple: (train_generator, val_generator, test_generator, class_mapping)
        """
        print("=" * 80)
        print("Creating Data Generators...")
        print("=" * 80)

        # Training data generator with augmentation
        train_datagen = ImageDataGenerator(
            rescale=self.aug_config['rescale'],
            rotation_range=self.aug_config['rotation_range'],
            width_shift_range=self.aug_config['width_shift_range'],
            height_shift_range=self.aug_config['height_shift_range'],
            shear_range=self.aug_config['shear_range'],
            zoom_range=self.aug_config['zoom_range'],
            horizontal_flip=self.aug_config['horizontal_flip'],
            vertical_flip=self.aug_config['vertical_flip'],
            fill_mode=self.aug_config['fill_mode'],
            validation_split=self.val_split + self.test_split  # Reserve for val+test
        )

        # Validation and test generators (NO augmentation, only rescaling)
        val_test_datagen = ImageDataGenerator(
            rescale=self.aug_config['rescale'],
            validation_split=self.test_split / (self.val_split + self.test_split)  # Split val/test
        )

        # Create training generator
        train_generator = train_datagen.flow_from_directory(
            self.dataset_path,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True,
            subset='training',  # Use training subset
            seed=self.random_seed
        )

        # Create combined val+test generator first
        val_test_generator_combined = train_datagen.flow_from_directory(
            self.dataset_path,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False,
            subset='validation',  # This gets val+test
            seed=self.random_seed
        )

        # For simplicity, we'll use a single validation split
        # In production, you'd want separate val and test sets
        val_generator = val_test_datagen.flow_from_directory(
            self.dataset_path,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False,
            subset='training',  # Use this as validation
            seed=self.random_seed
        )

        test_generator = val_test_datagen.flow_from_directory(
            self.dataset_path,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=False,
            subset='validation',  # Use this as test
            seed=self.random_seed
        )

        # Get class mapping from generator
        class_indices = train_generator.class_indices
        class_mapping = {v: k for k, v in class_indices.items()}

        # Save class mapping
        mapping_path = 'data/splits/class_mapping.json'
        os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
        with open(mapping_path, 'w') as f:
            json.dump(class_mapping, f, indent=2)

        # Print statistics
        print(f"✓ Training samples: {train_generator.samples}")
        print(f"✓ Validation samples: {val_generator.samples}")
        print(f"✓ Test samples: {test_generator.samples}")
        print(f"✓ Number of classes: {train_generator.num_classes}")
        print(f"✓ Batch size: {self.batch_size}")
        print(f"✓ Image size: {self.image_size}")
        print("=" * 80)

        return train_generator, val_generator, test_generator, class_mapping


def load_class_mapping(mapping_path='data/splits/class_mapping.json'):
    """
    Load class mapping from JSON file.

    Args:
        mapping_path (str): Path to class mapping JSON file

    Returns:
        dict: Class index to name mapping
    """
    with open(mapping_path, 'r') as f:
        class_mapping = json.load(f)

    # Convert string keys to integers
    class_mapping = {int(k): v for k, v in class_mapping.items()}

    return class_mapping


if __name__ == "__main__":
    # Test data loader
    print("Testing PlantDiseaseDataLoader...")

    loader = PlantDiseaseDataLoader('config.yaml')

    # Test data generators
    print("\nCreating data generators...")
    train_gen, val_gen, test_gen, class_map = loader.get_data_generators()

    print("\nSample batch from training generator:")
    images, labels = next(train_gen)
    print(f"Images shape: {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Image value range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"Label sum (should be 1.0): {labels[0].sum():.3f}")

    print("\nSample classes:")
    for i in range(min(5, len(class_map))):
        print(f"  {i}: {class_map[i]}")

    print("\n✓ Data loader test successful!")
