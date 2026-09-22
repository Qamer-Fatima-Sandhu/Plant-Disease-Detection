"""
Streamlit Web Application for Plant Disease Detection

This app allows users to upload plant leaf images and get real-time disease predictions
using the trained PlantDiseaseNet CNN model.
"""

import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import os
import sys
import plotly.graph_objects as go
import plotly.express as px

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Page configuration
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2e7d32;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .high-confidence {
        background-color: #c8e6c9;
        border-left: 5px solid #2e7d32;
    }
    .medium-confidence {
        background-color: #fff3e0;
        border-left: 5px solid #f57c00;
    }
    .low-confidence {
        background-color: #ffcdd2;
        border-left: 5px solid #c62828;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the trained model (cached to avoid reloading)."""
    model_path = 'models/saved_models/best_model.keras'

    if not os.path.exists(model_path):
        st.error(f"Model not found at: {model_path}")
        st.info("Please train the model first by running: python src/training/train.py")
        return None

    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_data
def load_class_mapping():
    """Load class index to name mapping."""
    mapping_path = 'data/splits/class_mapping.json'

    if not os.path.exists(mapping_path):
        # Return default mapping if file doesn't exist
        return {i: f"Class_{i}" for i in range(38)}

    with open(mapping_path, 'r') as f:
        class_mapping = json.load(f)

    # Convert string keys to integers
    class_mapping = {int(k): v for k, v in class_mapping.items()}

    return class_mapping


def preprocess_image(image):
    """
    Preprocess uploaded image for model prediction.

    Args:
        image (PIL.Image): Uploaded image

    Returns:
        np.ndarray: Preprocessed image array
    """
    # Resize to model input size
    img = image.resize((224, 224))

    # Convert to array
    img_array = np.array(img)

    # Handle grayscale images
    if len(img_array.shape) == 2:
        img_array = np.stack([img_array] * 3, axis=-1)

    # Handle RGBA images
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    # Normalize to [0, 1]
    img_array = img_array.astype('float32') / 255.0

    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


def make_prediction(model, image, class_names):
    """
    Make prediction on uploaded image.

    Args:
        model: Loaded Keras model
        image (PIL.Image): Uploaded image
        class_names (dict): Class mapping

    Returns:
        dict: Prediction results
    """
    # Preprocess image
    processed_image = preprocess_image(image)

    # Make prediction
    with st.spinner('🔍 Analyzing image...'):
        predictions = model.predict(processed_image, verbose=0)

    # Get top prediction
    top_idx = np.argmax(predictions[0])
    top_class = class_names[top_idx]
    top_confidence = float(predictions[0][top_idx])

    # Get top 5 predictions
    top_5_idx = np.argsort(predictions[0])[-5:][::-1]
    top_5 = [
        (class_names[idx], float(predictions[0][idx]))
        for idx in top_5_idx
    ]

    return {
        'class': top_class,
        'confidence': top_confidence,
        'top_5': top_5,
        'all_probabilities': predictions[0]
    }


def render_header():
    """Render application header."""
    st.markdown('<p class="main-header">🌿 Plant Disease Detection System</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Upload a plant leaf image to detect diseases using deep learning</p>',
        unsafe_allow_html=True
    )
    st.markdown("---")


def render_sidebar(class_names):
    """Render sidebar with information."""
    with st.sidebar:
        st.header("📋 About")
        st.info("""
        This application uses a custom **Convolutional Neural Network (PlantDiseaseNet)**
        trained on the **PlantVillage dataset** to identify plant diseases from leaf images.
        """)

        st.header("🔍 How to Use")
        st.markdown("""
        1. **Upload** a clear image of a plant leaf
        2. **Wait** for the model to process (~1-2 seconds)
        3. **View** the prediction and confidence score
        4. **Check** the top 5 possible diseases
        """)

        st.header("📊 Model Statistics")
        st.metric("Classes", len(class_names))
        st.metric("Architecture", "Custom CNN")
        st.metric("Parameters", "~8M")

        st.header("🌱 Supported Crops")
        crops = set()
        for class_name in class_names.values():
            crop = class_name.split('___')[0] if '___' in class_name else class_name.split('_')[0]
            crops.add(crop)

        for crop in sorted(crops):
            st.text(f"• {crop}")

        with st.expander("📚 All Disease Classes"):
            for i, class_name in sorted(class_names.items()):
                # Format class name for better readability
                formatted_name = class_name.replace('___', ' - ').replace('_', ' ')
                st.text(f"{i+1}. {formatted_name}")


def render_prediction_results(results):
    """Render prediction results with visualizations."""
    confidence = results['confidence']
    predicted_class = results['class']

    # Format class name
    formatted_class = predicted_class.replace('___', ' - ').replace('_', ' ')

    # Determine confidence level
    if confidence >= 0.9:
        conf_class = "high-confidence"
        emoji = "✅"
        message = "High Confidence"
    elif confidence >= 0.7:
        conf_class = "medium-confidence"
        emoji = "⚠️"
        message = "Medium Confidence"
    else:
        conf_class = "low-confidence"
        emoji = "❓"
        message = "Low Confidence"

    # Display main prediction
    st.markdown(f"""
    <div class="prediction-box {conf_class}">
        <h2>{emoji} {formatted_class}</h2>
        <h3>Confidence: {confidence*100:.2f}%</h3>
        <p>{message}</p>
    </div>
    """, unsafe_allow_html=True)

    # Confidence progress bar
    st.progress(confidence)

    # Display top 5 predictions
    st.subheader("📊 Top 5 Predictions")

    col1, col2 = st.columns(2)

    with col1:
        # Table view
        top_5_data = []
        for i, (cls, conf) in enumerate(results['top_5'], 1):
            formatted_cls = cls.replace('___', ' - ').replace('_', ' ')
            top_5_data.append({
                'Rank': i,
                'Disease': formatted_cls,
                'Confidence': f"{conf*100:.2f}%"
            })

        st.table(top_5_data)

    with col2:
        # Bar chart
        top_5_classes = [cls.replace('___', '\n').replace('_', ' ') for cls, _ in results['top_5']]
        top_5_confidences = [conf * 100 for _, conf in results['top_5']]

        fig = go.Figure(data=[
            go.Bar(
                y=top_5_classes,
                x=top_5_confidences,
                orientation='h',
                marker=dict(
                    color=top_5_confidences,
                    colorscale='Viridis',
                    showscale=False
                ),
                text=[f"{c:.1f}%" for c in top_5_confidences],
                textposition='outside'
            )
        ])

        fig.update_layout(
            title="Confidence Scores",
            xaxis_title="Confidence (%)",
            yaxis_title="",
            height=300,
            margin=dict(l=200, r=50, t=50, b=50)
        )

        st.plotly_chart(fig, use_container_width=True)

    # Interpretation guide
    if confidence < 0.7:
        st.warning("""
        ⚠️ **Low Confidence Warning**

        The model is not very confident about this prediction. Possible reasons:
        - Image quality is poor or blurry
        - Leaf is not clearly visible
        - Disease symptoms are not prominent
        - The image may be from a plant not in the training dataset

        **Recommendation:** Try uploading a clearer, well-lit image of the diseased leaf.
        """)


def main():
    """Main application."""
    # Load model and class mapping
    model = load_model()
    class_names = load_class_mapping()

    if model is None:
        st.stop()

    # Render header and sidebar
    render_header()
    render_sidebar(class_names)

    # Main content area
    st.header("📤 Upload Leaf Image")

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image of a plant leaf (JPG, JPEG, or PNG format)"
    )

    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📷 Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)

        with col2:
            st.subheader("📋 Image Information")
            st.write(f"**Filename:** {uploaded_file.name}")
            st.write(f"**Size:** {image.size[0]} x {image.size[1]} pixels")
            st.write(f"**Format:** {image.format}")
            st.write(f"**Mode:** {image.mode}")

        # Predict button
        st.markdown("---")

        if st.button("🔬 Analyze Image", type="primary", use_container_width=True):
            # Make prediction
            results = make_prediction(model, image, class_names)

            # Display results
            st.markdown("---")
            st.header("🎯 Prediction Results")
            render_prediction_results(results)

    else:
        # Show instructions
        st.info("""
        👆 **Get Started**

        1. Click the "Browse files" button above
        2. Select a clear image of a plant leaf
        3. Click "Analyze Image" to get the prediction

        **Tips for best results:**
        - Use well-lit images
        - Ensure the leaf is clearly visible
        - Avoid blurry or dark images
        - Include disease symptoms if present
        """)

        # Show example
        st.subheader("📸 Example Images")
        st.write("Upload images of leaves from crops like:")

        example_cols = st.columns(4)
        examples = ["Tomato 🍅", "Potato 🥔", "Pepper 🌶️", "Corn 🌽"]

        for col, example in zip(example_cols, examples):
            with col:
                st.info(example)


if __name__ == "__main__":
    main()
