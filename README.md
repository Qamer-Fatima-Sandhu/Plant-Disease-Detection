# Plant Disease Detection System

A deep learning-based system for detecting plant diseases from leaf images using a custom Convolutional Neural Network (CNN).

## Project Overview

This project implements an end-to-end plant disease detection system that can identify 38 different plant diseases across various crops (tomato, potato, pepper, corn, apple, grape, and more) from leaf images. The system uses a custom CNN architecture called **PlantDiseaseNet** trained on the PlantVillage dataset.

### Key Features

- Custom CNN Architecture - Built from scratch without transfer learning
- 38 Disease Classes - Covers multiple crops and disease types
- High Accuracy - Achieves 99.47% validation accuracy
- Web Interface - User-friendly Streamlit app for easy predictions
- Real-time Predictions - Fast inference (~1-2 seconds per image)
- Confidence Scores - Shows prediction confidence and top-5 alternatives
- Jupyter Notebook - Complete demonstration notebook for presentations

### Model Performance

- **Validation Accuracy:** 99.47%
- **Top-3 Accuracy:** 99.98%
- **Training Accuracy:** 98.81%
- **Total Parameters:** 27 Million
- **Model Size:** 309 MB
- **Inference Time:** 1-2 seconds per image

---

## Table of Contents

1. [Installation](#installation)
2. [Dataset Setup](#dataset-setup)
3. [Training the Model](#training-the-model)
4. [Running the Web Application](#running-the-web-application)
5. [Using the Jupyter Notebook](#using-the-jupyter-notebook)
6. [Google Colab Setup](#google-colab-setup)
7. [Visualization](#visualization)
8. [Project Structure](#project-structure)
9. [Model Architecture](#model-architecture)
10. [Troubleshooting](#troubleshooting)

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- Virtual environment (recommended)
- 4GB RAM minimum (24GB recommended for training)
- GPU optional but recommended for training

### Step 1: Clone or Download Project

```bash
cd /path/to/leaf-disease-detection
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Mac/Linux
# or
venv\Scripts\activate  # On Windows
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt
```

**Note for Mac with Apple Silicon (M1/M2/M3/M4):**
```bash
pip install tensorflow-macos==2.17.0 tensorflow-metal==1.1.0
```

### Verify Installation

```bash
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
python -c "import streamlit; print('Streamlit installed')"
python -c "import jupyter; print('Jupyter installed')"
```

---

## Dataset Setup

### Option 1: Download via Kaggle API (Recommended)

#### Step 1: Setup Kaggle API

1. Go to https://www.kaggle.com/settings
2. Scroll to "API" section
3. Click "Create New API Token"
4. This downloads `kaggle.json`

#### Step 2: Configure Kaggle Credentials

```bash
# Create .kaggle directory
mkdir -p ~/.kaggle

# Move the token file
mv ~/Downloads/kaggle.json ~/.kaggle/

# Set permissions
chmod 600 ~/.kaggle/kaggle.json
```

#### Step 3: Download Dataset

```bash
# Activate virtual environment
source venv/bin/activate

# Download using Python
python -c "from src.data.data_loader import PlantDiseaseDataLoader; loader = PlantDiseaseDataLoader(); loader.download_dataset()"
```

### Option 2: Manual Download

1. Visit: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
2. Download the dataset (1.2 GB)
3. Extract to: `data/raw/PlantVillage/`

### Verify Dataset

```bash
ls data/raw/PlantVillage/
# Should show directories like: Apple___Apple_scab, Tomato___Early_blight, etc.
```

---

## Training the Model

### Quick Start Training

```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Start training
python src/training/train.py
```

### Training Configuration

The model trains with settings defined in `config.yaml`:

- **Epochs:** 40 (with early stopping)
- **Batch Size:** 32
- **Learning Rate:** 0.001 (adaptive reduction)
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Data Split:** 70% train, 15% validation, 15% test

### Training Time Estimates

- **GPU (Apple Silicon/NVIDIA/AMD):** 30-60 minutes
- **CPU only:** 2-4 hours

**Note:** Early stopping typically triggers around epoch 30-35.

### Monitor Training with TensorBoard

Open a new terminal and run:

```bash
tensorboard --logdir=logs/fit
```

Then open http://localhost:6006 in your browser to see:
- Training and validation accuracy curves
- Loss curves
- Model graph
- Histograms of weights and biases

### Training Output

After training completes, you'll find:
- `models/saved_models/best_model.keras` - Trained model (309 MB)
- `models/saved_models/training_history.json` - Training metrics
- `models/saved_models/training_history.csv` - Training log
- `logs/fit/` - TensorBoard logs

### Customize Training

Edit `config.yaml` to modify:
- Batch size
- Number of epochs
- Learning rate
- Model architecture parameters
- Data augmentation settings

---

## Running the Web Application

### Start Streamlit App

```bash
# Activate virtual environment
source venv/bin/activate

# Run the app
streamlit run app/streamlit_app.py
```

The app will automatically open in your browser at http://localhost:8501

### Using the Web Interface

1. **Upload Image**
   - Click "Browse files" button
   - Select a leaf image (JPG, JPEG, or PNG)
   - Image preview will appear

2. **Analyze Image**
   - Click "Analyze Image" button
   - Wait 1-2 seconds for prediction

3. **View Results**
   - Primary prediction with confidence score
   - Top-5 alternative predictions
   - Confidence visualization (bar chart)
   - Color-coded confidence levels:
     - Green: High confidence (>90%)
     - Orange: Medium confidence (70-90%)
     - Red: Low confidence (<70%)

### Stopping the App

Press `Ctrl+C` in the terminal to stop the Streamlit server.

---

## Using the Jupyter Notebook

The project includes a comprehensive Jupyter notebook for demonstrations and presentations.

### Run Locally

```bash
# Activate virtual environment
source venv/bin/activate

# Start Jupyter
jupyter notebook notebooks/Plant_Disease_Detection_Complete.ipynb
```

### Notebook Features

The notebook includes:
- **Step 1-2:** Environment setup and dependency installation
- **Step 3:** Project structure verification
- **Step 4:** Load and inspect the trained model
- **Step 5:** Visualize training history with plots
- **Step 6:** Define prediction functions
- **Step 7:** Upload and test images with predictions
- **Step 8:** View all 38 disease classes
- **Step 9:** Detailed model architecture breakdown
- **Step 10:** Project summary and results
- **Step 11:** Optional training directly in notebook
- **Step 12:** Run Streamlit app with public URL (Colab)

### Run Cells Sequentially

Execute cells in order from top to bottom using:
- Click the "Run" button
- Press `Shift + Enter`
- Menu: Cell → Run All

---

## Google Colab Setup

Perfect for sharing with professors or running without local setup.

### Step 1: Prepare Project for Colab

```bash
# Create zip file with all necessary files
./prepare_for_colab.sh
```

This creates `leaf-disease-detection.zip` (~310 MB) containing:
- Trained model
- Source code
- Configuration files
- Notebook
- Requirements file

### Step 2: Upload to Google Colab

#### Option A: Via Google Drive (Recommended)

1. Upload `leaf-disease-detection.zip` to your Google Drive
2. Go to https://colab.research.google.com/
3. Upload `notebooks/Plant_Disease_Detection_Complete.ipynb`
4. In the notebook, run the "Option A" cell in Step 1
5. Mount Google Drive when prompted
6. Update the path to your zip file location

#### Option B: Direct Upload

1. Go to https://colab.research.google.com/
2. Upload `notebooks/Plant_Disease_Detection_Complete.ipynb`
3. In the notebook, run the "Option B" cell in Step 1
4. Upload `leaf-disease-detection.zip` when prompted (takes 2-5 minutes)

### Step 3: Run the Notebook

1. Execute cells sequentially from top to bottom
2. The notebook auto-detects Colab environment
3. Dependencies install automatically
4. All features work the same as local

### Step 4: Demonstrate Streamlit in Colab

The notebook includes a special section (Step 12) that:
1. Installs ngrok for public URL generation
2. Starts Streamlit server in Colab
3. Creates a public URL like: `https://xxxx.ngrok.io`
4. Share this URL with your professor
5. Anyone with the URL can access and interact with the app

### Sharing with Professor

**Method 1: Share Colab Link**
- In Colab, click "Share" button (top right)
- Set to "Anyone with the link can view"
- Send link to professor

**Method 2: Share Notebook File**
- Download notebook: File → Download → .ipynb
- Email the .ipynb file
- Professor uploads to their Colab

**Method 3: Live Demo**
- Run Step 12 in notebook
- Share the ngrok public URL
- Professor can interact with app directly

---

## Visualization

### Training History Plots

The notebook automatically generates:

1. **Accuracy Plot**
   - Training accuracy over epochs
   - Validation accuracy over epochs
   - Shows model learning progress

2. **Loss Plot**
   - Training loss over epochs
   - Validation loss over epochs
   - Indicates convergence

3. **Top-3 Accuracy Plot**
   - Shows how often correct answer is in top-3 predictions
   - Useful for understanding model confidence

### Prediction Visualizations

When testing images, you'll see:

1. **Input Image Display**
   - Original uploaded image
   - Resized preview

2. **Primary Prediction Box**
   - Disease name
   - Confidence percentage
   - Color-coded by confidence level

3. **Top-5 Predictions Chart**
   - Horizontal bar chart
   - All top-5 predictions with confidence scores
   - Easy comparison of alternatives

4. **Confidence Progress Bar**
   - Visual representation of prediction confidence
   - Helps assess reliability

### TensorBoard Visualizations

Access advanced visualizations:

```bash
tensorboard --logdir=logs/fit
```

View at http://localhost:6006:
- Scalars: Accuracy and loss curves
- Graphs: Model architecture visualization
- Distributions: Weight and bias distributions
- Histograms: Layer activation patterns

---

## Project Structure

```
leaf-disease-detection/
├── app/
│   └── streamlit_app.py          # Web application
├── config.yaml                    # Configuration file
├── data/
│   ├── raw/
│   │   └── PlantVillage/         # Dataset (gitignored)
│   └── splits/
│       └── class_mapping.json    # Class names
├── models/
│   └── saved_models/
│       ├── best_model.keras      # Trained model (309 MB)
│       ├── training_history.json # Training metrics
│       └── training_history.csv  # Training log
├── notebooks/
│   └── Plant_Disease_Detection_Complete.ipynb  # Demo notebook
├── src/
│   ├── data/
│   │   └── data_loader.py        # Data loading and augmentation
│   ├── models/
│   │   └── cnn_model.py          # Model architecture
│   ├── training/
│   │   └── train.py              # Training pipeline
│   └── utils/
├── logs/                          # TensorBoard logs (gitignored)
├── venv/                          # Virtual environment (gitignored)
├── .gitignore                     # Git ignore rules
├── requirements.txt               # Python dependencies (local)
├── requirements-colab.txt         # Python dependencies (Colab)
├── prepare_for_colab.sh          # Create Colab zip
├── test_notebook_locally.sh      # Validate notebook
├── run_app.sh                    # Quick start Streamlit
├── start_training.sh             # Quick start training
└── README.md                      # This file
```

---

## Model Architecture

### PlantDiseaseNet CNN

```
Input Layer (224×224×3 RGB images)
    ↓
Block 1: Low-level features (edges, colors, textures)
    Conv2D(32, 3×3) + BatchNorm + ReLU
    Conv2D(32, 3×3) + BatchNorm + ReLU
    MaxPooling2D(2×2)
    Dropout(0.25)
    ↓
Block 2: Mid-level features (patterns, shapes)
    Conv2D(64, 3×3) + BatchNorm + ReLU
    Conv2D(64, 3×3) + BatchNorm + ReLU
    MaxPooling2D(2×2)
    Dropout(0.25)
    ↓
Block 3: High-level features (complex patterns)
    Conv2D(128, 3×3) + BatchNorm + ReLU
    Conv2D(128, 3×3) + BatchNorm + ReLU
    MaxPooling2D(2×2)
    Dropout(0.3)
    ↓
Block 4: Deep features (disease-specific patterns)
    Conv2D(256, 3×3) + BatchNorm + ReLU
    Conv2D(256, 3×3) + BatchNorm + ReLU
    MaxPooling2D(2×2)
    Dropout(0.3)
    ↓
Classification Head
    Flatten
    Dense(512) + BatchNorm + ReLU + Dropout(0.5)
    Dense(256) + BatchNorm + ReLU + Dropout(0.5)
    Dense(38, softmax)
```

### Design Principles

- **Progressive Filter Increase:** 32→64→128→256 for hierarchical feature learning
- **VGG-Style Architecture:** Double convolutions per block for complex feature extraction
- **Batch Normalization:** After each convolutional layer for training stability
- **Strategic Dropout:** 0.25-0.5 to prevent overfitting
- **No Transfer Learning:** Built from scratch for domain specialization

### Data Augmentation

Training images undergo real-time augmentation:
- Rotation: ±40 degrees
- Width/Height shift: ±20%
- Shear transformation: 20%
- Zoom: ±20%
- Horizontal and vertical flips
- Normalization: [0, 1]

### Training Callbacks

- **ModelCheckpoint:** Saves best model based on validation accuracy
- **EarlyStopping:** Stops training if no improvement for 10 epochs
- **ReduceLROnPlateau:** Reduces learning rate when plateaued (factor=0.5, patience=3)
- **CSVLogger:** Logs metrics to CSV file
- **TensorBoard:** Real-time visualization

---

## Troubleshooting

### Installation Issues

**Problem:** `pip install` fails
```bash
# Solution: Upgrade pip
pip install --upgrade pip
python -m pip install --upgrade pip
```

**Problem:** TensorFlow installation fails on Mac
```bash
# Solution: Use Mac-specific TensorFlow
pip install tensorflow-macos==2.17.0 tensorflow-metal==1.1.0
```

**Problem:** Module not found errors
```bash
# Solution: Ensure virtual environment is activated
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### Dataset Issues

**Problem:** Kaggle API not configured
```bash
# Solution: Setup Kaggle credentials
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

**Problem:** Dataset not found during training
```bash
# Solution: Verify dataset location
ls data/raw/PlantVillage/
# Should show disease class directories
```

### Training Issues

**Problem:** Out of memory during training
```bash
# Solution: Reduce batch size in config.yaml
# Change batch_size from 32 to 16 or 8
```

**Problem:** Training is very slow
```bash
# Solution: Check GPU availability
python -c "import tensorflow as tf; print('GPU available:', len(tf.config.list_physical_devices('GPU')) > 0)"

# For Mac: Install tensorflow-metal
pip install tensorflow-metal
```

**Problem:** Model not saving
```bash
# Solution: Check directory exists
mkdir -p models/saved_models
```

### Streamlit Issues

**Problem:** Streamlit not starting
```bash
# Solution: Reinstall Streamlit
pip uninstall streamlit
pip install streamlit>=1.29.0
```

**Problem:** Model not found in Streamlit app
```bash
# Solution: Verify model file exists
ls -lh models/saved_models/best_model.keras
# Should show ~309 MB file
```

**Problem:** Port 8501 already in use
```bash
# Solution: Use different port
streamlit run app/streamlit_app.py --server.port 8502
```

### Notebook Issues

**Problem:** Jupyter not starting
```bash
# Solution: Install Jupyter
pip install jupyter ipykernel
```

**Problem:** Kernel dies during execution
```bash
# Solution: Increase available memory or reduce batch size
# Restart kernel and run cells again
```

### Google Colab Issues

**Problem:** File upload times out
```bash
# Solution: Use Google Drive option instead
# Upload zip to Drive first, then mount in Colab
```

**Problem:** ngrok URL not working
```bash
# Solution: Wait 10-15 seconds after starting Streamlit
# Restart the cell if needed
```

**Problem:** Session disconnected
```bash
# Solution: Keep browser tab active
# Reconnect and re-run cells from checkpoint
```

### General Issues

**Problem:** Import errors
```bash
# Solution: Ensure you're in project root
cd /path/to/leaf-disease-detection
source venv/bin/activate
```

**Problem:** Permission denied errors
```bash
# Solution: Check file permissions
chmod +x *.sh  # Make scripts executable
```

---

## Quick Commands Reference

### Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Training
```bash
python src/training/train.py
tensorboard --logdir=logs/fit
```

### Run App
```bash
streamlit run app/streamlit_app.py
```

### Notebook
```bash
jupyter notebook notebooks/Plant_Disease_Detection_Complete.ipynb
```

### Colab Prep
```bash
./prepare_for_colab.sh
```

### Test
```bash
./test_notebook_locally.sh
```

---

## Dataset Information

### PlantVillage Dataset

- **Total Images:** 54,303
- **Classes:** 38 disease/healthy combinations
- **Image Size:** 224×224 pixels (resized)
- **Format:** JPG
- **Source:** Kaggle

### Supported Crops (14 total)

- Apple
- Blueberry
- Cherry
- Corn
- Grape
- Orange
- Peach
- Pepper (Bell)
- Potato
- Raspberry
- Soybean
- Squash
- Strawberry
- Tomato

### Disease Examples

- Tomato: Early blight, Late blight, Leaf mold, Septoria leaf spot, etc.
- Potato: Early blight, Late blight
- Corn: Common rust, Gray leaf spot, Northern leaf blight
- Apple: Apple scab, Black rot, Cedar apple rust
- Grape: Black rot, Esca, Leaf blight

---

## Performance Metrics

### Achieved Results

| Metric | Value |
|--------|-------|
| Validation Accuracy | 99.47% |
| Top-3 Accuracy | 99.98% |
| Training Accuracy | 98.81% |
| Validation Loss | 0.0212 |
| Total Parameters | 27 Million |
| Model Size | 309 MB |
| Inference Time | 1-2 seconds |

### Performance Breakdown

- Out of 27,162 validation images, only 144 misclassifications
- Correctly identifies disease in top-3 predictions 99.98% of the time
- Consistent performance across all 38 disease classes
- Fast inference suitable for real-time applications

---

## Requirements Files

### requirements.txt (Local Development)

Used for local development. Includes:
- TensorFlow 2.17.0
- Streamlit
- Jupyter and ipykernel
- Data processing libraries (numpy, pandas, scikit-learn)
- Visualization libraries (matplotlib, seaborn, plotly)
- Utilities (pyyaml, tqdm)
- Kaggle API

### requirements-colab.txt (Google Colab)

Used automatically in Colab notebook. Excludes:
- Jupyter packages (already in Colab)

Includes:
- ml_dtypes fix for JAX compatibility
- All other dependencies same as requirements.txt

---

## For Presentations

### Key Points to Highlight

1. **Custom Architecture**
   - Built from scratch without transfer learning
   - Specialized for plant disease detection
   - 27 million parameters optimized for this task

2. **High Accuracy**
   - 99.47% validation accuracy
   - 99.98% top-3 accuracy
   - Reliable predictions with confidence scores

3. **Real-world Application**
   - Helps farmers detect diseases early
   - Fast inference (1-2 seconds)
   - User-friendly web interface

4. **Complete System**
   - Training pipeline
   - Web application
   - Jupyter notebook for demonstrations
   - TensorBoard visualization

5. **Production Ready**
   - Optimized for CPU and GPU
   - Easy deployment
   - Comprehensive documentation

### Demo Workflow

1. **Show Project Structure**
   - Organized codebase
   - Clear separation of concerns

2. **Explain Architecture**
   - CNN design principles
   - Progressive feature extraction

3. **Display Training Results**
   - Show training curves
   - Highlight accuracy metrics

4. **Live Demo**
   - Run Streamlit app (or use notebook Step 12 for public URL)
   - Upload test images
   - Show predictions and confidence scores

5. **Code Walkthrough**
   - Data loading and augmentation
   - Model architecture
   - Training pipeline

---

## Author

**Abdul Rafay**
- Project: Plant Disease Detection using Deep Learning
- Date: January 2026
- Purpose: Educational Project - Semester Project

---

## License

This project is for educational purposes as part of a semester project.

---

## Important Note

This system is designed for educational purposes and should not be used as the sole diagnostic tool for agricultural decisions. Always consult with agricultural experts for critical plant health assessments.

---

## Additional Resources

- **PlantVillage Dataset:** https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset
- **TensorFlow Documentation:** https://www.tensorflow.org/
- **Streamlit Documentation:** https://docs.streamlit.io/
- **Keras Documentation:** https://keras.io/

---

**Last Updated:** January 2026
