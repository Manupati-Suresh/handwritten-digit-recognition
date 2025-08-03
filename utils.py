"""
Utility functions for the digit recognition app.
"""

import numpy as np
import pandas as pd
from PIL import Image
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

def validate_model_file(model_path: str) -> bool:
    """Validate that the model file exists and is readable."""
    import os
    import pickle
    
    try:
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return False
        
        # Try to load the model to ensure it's valid
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Check if it's the new format with metadata
        if isinstance(model_data, dict) and 'model' in model_data:
            model = model_data['model']
        else:
            model = model_data
        
        # Basic validation - check if it has predict method
        if not hasattr(model, 'predict'):
            logger.error("Invalid model: missing predict method")
            return False
        
        logger.info("Model validation successful")
        return True
        
    except Exception as e:
        logger.error(f"Model validation failed: {str(e)}")
        return False

def create_sample_images() -> dict:
    """Create sample digit images for demonstration."""
    samples = {}
    
    # Create simple digit patterns
    # Digit 0
    zero = np.zeros((28, 28))
    zero[8:20, 10:12] = 255  # left line
    zero[8:20, 16:18] = 255  # right line
    zero[8:10, 10:18] = 255  # top line
    zero[18:20, 10:18] = 255  # bottom line
    samples[0] = zero
    
    # Digit 1
    one = np.zeros((28, 28))
    one[6:22, 13:15] = 255  # vertical line
    one[6:8, 11:13] = 255   # top diagonal
    samples[1] = one
    
    # Digit 7
    seven = np.zeros((28, 28))
    seven[6:8, 8:20] = 255   # top horizontal line
    seven[8:20, 17:19] = 255  # diagonal line
    samples[7] = seven
    
    return samples

def get_image_stats(image: np.ndarray) -> dict:
    """Get basic statistics about an image array."""
    return {
        'shape': image.shape,
        'dtype': image.dtype,
        'min': float(np.min(image)),
        'max': float(np.max(image)),
        'mean': float(np.mean(image)),
        'std': float(np.std(image))
    }

def normalize_image(image: np.ndarray) -> np.ndarray:
    """Normalize image to [0, 1] range."""
    if image.max() > 1.0:
        return image.astype(np.float32) / 255.0
    return image.astype(np.float32)

def resize_image_with_padding(image: Image.Image, target_size: Tuple[int, int] = (28, 28)) -> Image.Image:
    """Resize image while maintaining aspect ratio and adding padding if needed."""
    # Calculate aspect ratio
    aspect_ratio = image.width / image.height
    target_aspect = target_size[0] / target_size[1]
    
    if aspect_ratio > target_aspect:
        # Image is wider, fit to width
        new_width = target_size[0]
        new_height = int(target_size[0] / aspect_ratio)
    else:
        # Image is taller, fit to height
        new_height = target_size[1]
        new_width = int(target_size[1] * aspect_ratio)
    
    # Resize image
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Create new image with target size and paste resized image in center
    result = Image.new('L', target_size, color=0)  # Black background
    
    # Calculate position to center the image
    x_offset = (target_size[0] - new_width) // 2
    y_offset = (target_size[1] - new_height) // 2
    
    result.paste(resized, (x_offset, y_offset))
    
    return result

def check_data_files() -> dict:
    """Check if training/test data files exist."""
    import os
    
    files = {
        'mnist_train.csv': os.path.exists('mnist_train.csv'),
        'mnist_test.csv': os.path.exists('mnist_test.csv'),
        'svm_digit_model.pkl': os.path.exists('svm_digit_model.pkl')
    }
    
    return files

def get_model_info(model_path: str = "svm_digit_model.pkl") -> Optional[dict]:
    """Get information about the trained model."""
    import pickle
    import os
    
    try:
        if not os.path.exists(model_path):
            return None
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        # Check if it's the new format with metadata
        if isinstance(model_data, dict) and 'metadata' in model_data:
            return model_data['metadata']
        else:
            # Old format, return basic info
            return {
                'model_type': 'SVM',
                'format': 'legacy',
                'created_at': 'unknown'
            }
    
    except Exception as e:
        logger.error(f"Error reading model info: {str(e)}")
        return None