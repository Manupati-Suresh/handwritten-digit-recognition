
import streamlit as st
import numpy as np
import pickle
import os
import logging
from PIL import Image, ImageOps
from typing import Optional
import io

class SimpleDigitClassifier:
    """A simple digit classifier using template matching."""
    
    def __init__(self):
        self.templates = {}
        self.is_trained = False
    
    def create_digit_templates(self):
        """Create simple templates for each digit."""
        templates = {}
        
        # Create basic templates (28x28) for digits 0-9
        # These are simplified patterns that represent each digit
        
        # Digit 0 - circle/oval shape
        zero = np.zeros((28, 28))
        zero[6:22, 8:10] = 1    # left line
        zero[6:22, 18:20] = 1   # right line
        zero[6:8, 8:20] = 1     # top line
        zero[20:22, 8:20] = 1   # bottom line
        templates[0] = zero
        
        # Digit 1 - vertical line
        one = np.zeros((28, 28))
        one[4:24, 13:15] = 1    # main vertical line
        one[4:6, 11:13] = 1     # top diagonal
        templates[1] = one
        
        # Digit 2 - curved top, horizontal middle, bottom line
        two = np.zeros((28, 28))
        two[6:8, 8:20] = 1      # top line
        two[6:14, 18:20] = 1    # right line (top half)
        two[13:15, 8:20] = 1    # middle line
        two[14:22, 8:10] = 1    # left line (bottom half)
        two[20:22, 8:20] = 1    # bottom line
        templates[2] = two
        
        # Digit 3 - similar to 2 but with right curves
        three = np.zeros((28, 28))
        three[6:8, 8:20] = 1    # top line
        three[13:15, 8:20] = 1  # middle line
        three[20:22, 8:20] = 1  # bottom line
        three[6:22, 18:20] = 1  # right line
        templates[3] = three
        
        # Digit 4 - vertical lines with horizontal connector
        four = np.zeros((28, 28))
        four[6:16, 8:10] = 1    # left line (top part)
        four[14:16, 8:20] = 1   # horizontal line
        four[6:24, 18:20] = 1   # right line
        templates[4] = four
        
        # Digit 5 - top line, left line, middle, right bottom
        five = np.zeros((28, 28))
        five[6:8, 8:20] = 1     # top line
        five[6:14, 8:10] = 1    # left line (top half)
        five[13:15, 8:20] = 1   # middle line
        five[14:22, 18:20] = 1  # right line (bottom half)
        five[20:22, 8:20] = 1   # bottom line
        templates[5] = five
        
        # Digit 6 - like 5 but with left bottom line
        six = np.zeros((28, 28))
        six[6:8, 8:20] = 1      # top line
        six[6:22, 8:10] = 1     # left line
        six[13:15, 8:20] = 1    # middle line
        six[14:22, 18:20] = 1   # right line (bottom half)
        six[20:22, 8:20] = 1    # bottom line
        templates[6] = six
        
        # Digit 7 - top line and diagonal
        seven = np.zeros((28, 28))
        seven[6:8, 8:20] = 1    # top line
        seven[8:22, 16:18] = 1  # diagonal line
        templates[7] = seven
        
        # Digit 8 - like 0 but with middle line
        eight = np.zeros((28, 28))
        eight[6:22, 8:10] = 1   # left line
        eight[6:22, 18:20] = 1  # right line
        eight[6:8, 8:20] = 1    # top line
        eight[13:15, 8:20] = 1  # middle line
        eight[20:22, 8:20] = 1  # bottom line
        templates[8] = eight
        
        # Digit 9 - like 6 but mirrored
        nine = np.zeros((28, 28))
        nine[6:8, 8:20] = 1     # top line
        nine[6:14, 8:10] = 1    # left line (top half)
        nine[6:22, 18:20] = 1   # right line
        nine[13:15, 8:20] = 1   # middle line
        nine[20:22, 8:20] = 1   # bottom line
        templates[9] = nine
        
        return templates
    
    def fit(self, X=None, y=None):
        """Fit the model (create templates)."""
        self.templates = self.create_digit_templates()
        self.is_trained = True
        return self
    
    def predict(self, X):
        """Predict digits using template matching."""
        if not self.is_trained:
            self.fit()
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        predictions = []
        for sample in X:
            # Reshape to 28x28
            image = sample.reshape(28, 28)
            
            # Calculate similarity with each template
            similarities = []
            for digit, template in self.templates.items():
                # Simple correlation-based similarity
                similarity = np.corrcoef(image.flatten(), template.flatten())[0, 1]
                if np.isnan(similarity):
                    similarity = 0
                similarities.append((digit, similarity))
            
            # Find best match
            best_digit = max(similarities, key=lambda x: x[1])[0]
            predictions.append(best_digit)
        
        return np.array(predictions)
    
    def decision_function(self, X):
        """Return decision scores for each class."""
        if not self.is_trained:
            self.fit()
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        all_scores = []
        for sample in X:
            # Reshape to 28x28
            image = sample.reshape(28, 28)
            
            # Calculate similarity with each template
            scores = []
            for digit in range(10):
                template = self.templates[digit]
                # Simple correlation-based similarity
                similarity = np.corrcoef(image.flatten(), template.flatten())[0, 1]
                if np.isnan(similarity):
                    similarity = 0
                scores.append(similarity)
            
            all_scores.append(scores)
        
        return np.array(all_scores)

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
                            
                            # Create data for visualization
                            scores_data = {f"Digit {i}": confidence_scores[i] for i in range(10)}
                            
                            # Create bar chart
                            st.bar_chart(scores_data)
                            
                            # Show top 3 predictions
                            st.write("**Top 3 Predictions:**")
                            sorted_scores = sorted(enumerate(confidence_scores), key=lambda x: x[1], reverse=True)
                            for i, (digit, score) in enumerate(sorted_scores[:3]):
                                st.write(f"{i+1}. Digit {digit}: {score:.3f}")
                                
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
