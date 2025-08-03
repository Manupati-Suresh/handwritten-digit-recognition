
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import logging
from PIL import Image, ImageOps
from typing import Optional
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="✍️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        color: #1f77b4;
    }
    .prediction-box {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #ffe6e6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ff4444;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained SVM model with error handling."""
    try:
        model_path = "svm_digit_model.pkl"
        if not os.path.exists(model_path):
            st.error(f"Model file '{model_path}' not found. Please ensure the model is trained and saved.")
            st.stop()
        
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)
        
        # Handle both old and new model formats
        if isinstance(model_data, dict) and 'model' in model_data:
            model = model_data['model']
            metadata = model_data.get('metadata', {})
            logger.info(f"Model loaded successfully with metadata: {metadata}")
        else:
            model = model_data
            logger.info("Model loaded successfully (legacy format)")
        
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        st.error(f"Failed to load model: {str(e)}")
        st.stop()

def validate_image(image: Image.Image) -> bool:
    """Validate uploaded image format and size."""
    try:
        # Check if image is valid
        if image is None:
            return False
        
        # Check image mode
        if image.mode not in ['L', 'RGB', 'RGBA']:
            return False
            
        return True
    except Exception as e:
        logger.error(f"Image validation error: {str(e)}")
        return False

def preprocess_image(image: Image.Image) -> Optional[np.ndarray]:
    """Preprocess the uploaded image for prediction."""
    try:
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        # Resize to 28x28
        image = image.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array and normalize
        img_array = np.array(image)
        
        # Check if image needs inversion (white background to black)
        if np.mean(img_array) > 127:
            img_array = 255 - img_array
        
        # Normalize to [0, 1]
        img_array = img_array.astype(np.float32) / 255.0
        
        # Reshape for model input
        img_array = img_array.reshape(1, -1)
        
        return img_array
    except Exception as e:
        logger.error(f"Image preprocessing error: {str(e)}")
        return None

def main():
    """Main application function."""
    # Header
    st.markdown('<h1 class="main-header">✍️ Handwritten Digit Recognition using SVM</h1>', unsafe_allow_html=True)
    
    # Load model
    model = load_model()
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        This app uses a Support Vector Machine (SVM) to recognize handwritten digits (0-9).
        
        **How to use:**
        1. Upload a clear image of a handwritten digit
        2. The image will be automatically resized to 28x28 pixels
        3. Click 'Predict Digit' to get the result
        
        **Tips for best results:**
        - Use clear, dark digits on light background
        - Ensure the digit fills most of the image
        - Avoid cluttered backgrounds
        """)
        
        st.header("📊 Model Info")
        st.write("""
        - **Algorithm:** Support Vector Machine
        - **Training Data:** MNIST Dataset
        - **Input Size:** 28x28 pixels
        - **Classes:** 0-9 digits
        """)
    
    # Main content
    st.write("Upload an image of a handwritten digit for recognition:")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["png", "jpg", "jpeg", "bmp", "tiff"],
        help="Upload a clear image of a handwritten digit (0-9)"
    )
    
    if uploaded_file is not None:
        try:
            # Display uploaded image
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📤 Uploaded Image")
                image = Image.open(uploaded_file)
                st.image(image, caption="Original Image", use_column_width=True)
            
            # Validate image
            if not validate_image(image):
                st.error("Invalid image format. Please upload a valid image file.")
                return
            
            # Preprocess image
            processed_image = preprocess_image(image)
            
            if processed_image is None:
                st.error("Failed to process the image. Please try with a different image.")
                return
            
            with col2:
                st.subheader("🔄 Processed Image")
                # Show processed 28x28 image
                processed_display = (processed_image.reshape(28, 28) * 255).astype(np.uint8)
                st.image(processed_display, caption="Processed (28x28)", use_column_width=True)
            
            # Prediction section
            st.markdown("---")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🔮 Predict Digit", type="primary", use_container_width=True):
                    with st.spinner("Analyzing digit..."):
                        try:
                            # Make prediction
                            prediction = model.predict(processed_image)[0]
                            confidence_scores = model.decision_function(processed_image)[0]
                            
                            # Display result
                            st.markdown(f"""
                            <div class="prediction-box">
                                <h2 style="text-align: center; margin: 0;">
                                    🎯 Predicted Digit: <span style="color: #1f77b4; font-size: 2em;">{prediction}</span>
                                </h2>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Show confidence scores
                            st.subheader("📊 Confidence Scores")
                            scores_df = pd.DataFrame({
                                'Digit': range(10),
                                'Score': confidence_scores
                            })
                            scores_df = scores_df.sort_values('Score', ascending=False)
                            
                            # Create bar chart
                            st.bar_chart(scores_df.set_index('Digit')['Score'])
                            
                            # Show top 3 predictions
                            st.write("**Top 3 Predictions:**")
                            for i, (_, row) in enumerate(scores_df.head(3).iterrows()):
                                st.write(f"{i+1}. Digit {int(row['Digit'])}: {row['Score']:.3f}")
                                
                        except Exception as e:
                            logger.error(f"Prediction error: {str(e)}")
                            st.markdown(f"""
                            <div class="error-box">
                                <h3>❌ Prediction Failed</h3>
                                <p>Error: {str(e)}</p>
                                <p>Please try with a different image.</p>
                            </div>
                            """, unsafe_allow_html=True)
            
        except Exception as e:
            logger.error(f"Application error: {str(e)}")
            st.error(f"An error occurred while processing your request: {str(e)}")
    
    else:
        st.info("👆 Please upload an image to get started!")
        
        # Show example
        st.subheader("📝 Example")
        st.write("Here's what a good input image looks like:")
        
        # Create a sample digit image for demonstration
        sample_digit = np.zeros((28, 28))
        # Draw a simple "7"
        sample_digit[5:8, 8:20] = 1.0  # top horizontal line
        sample_digit[8:20, 17:20] = 1.0  # diagonal line
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            st.image(sample_digit, caption="Example: Clear digit on dark background", use_column_width=True)

if __name__ == "__main__":
    main()
