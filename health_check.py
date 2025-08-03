"""
Health check script to verify app dependencies and model availability.
Run this before deploying to ensure everything is working correctly.
"""

import sys
import os
import importlib
import logging
from utils import validate_model_file, check_data_files, get_model_info

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    logger.info(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error("Python 3.8+ is required")
        return False
    
    logger.info("✅ Python version is compatible")
    return True

def check_dependencies():
    """Check if all required packages are installed."""
    required_packages = [
        'streamlit',
        'numpy',
        'pandas',
        'sklearn',
        'PIL',
        'pickle'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'PIL':
                importlib.import_module('PIL')
            elif package == 'sklearn':
                importlib.import_module('sklearn')
            else:
                importlib.import_module(package)
            logger.info(f"✅ {package} is available")
        except ImportError:
            logger.error(f"❌ {package} is not installed")
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing packages: {missing_packages}")
        logger.error("Run: pip install -r requirements.txt")
        return False
    
    logger.info("✅ All dependencies are installed")
    return True

def check_model():
    """Check if the trained model is available and valid."""
    model_path = "svm_digit_model.pkl"
    
    if not validate_model_file(model_path):
        logger.error("❌ Model validation failed")
        logger.error("Run: python train_model.py to train the model")
        return False
    
    # Get model info
    model_info = get_model_info(model_path)
    if model_info:
        logger.info("✅ Model is valid")
        logger.info(f"Model info: {model_info}")
    else:
        logger.warning("⚠️ Could not read model metadata")
    
    return True

def check_data_files_status():
    """Check status of data files."""
    files = check_data_files()
    
    for filename, exists in files.items():
        if exists:
            logger.info(f"✅ {filename} found")
        else:
            if filename.endswith('.pkl'):
                logger.error(f"❌ {filename} not found - required for app")
            else:
                logger.warning(f"⚠️ {filename} not found - needed for training")
    
    return files['svm_digit_model.pkl']

def check_streamlit_config():
    """Check Streamlit configuration."""
    config_path = ".streamlit/config.toml"
    
    if os.path.exists(config_path):
        logger.info("✅ Streamlit config found")
        return True
    else:
        logger.warning("⚠️ Streamlit config not found (optional)")
        return True

def run_basic_app_test():
    """Run a basic test of the app components."""
    try:
        # Test imports
        import app
        logger.info("✅ App imports successfully")
        
        # Test model loading function
        model = app.load_model()
        if model is not None:
            logger.info("✅ Model loads successfully")
        else:
            logger.error("❌ Model loading failed")
            return False
        
        # Test image preprocessing
        from PIL import Image
        import numpy as np
        
        # Create a test image
        test_image = Image.new('L', (50, 50), color=128)
        processed = app.preprocess_image(test_image)
        
        if processed is not None and processed.shape == (1, 784):
            logger.info("✅ Image preprocessing works")
        else:
            logger.error("❌ Image preprocessing failed")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ App test failed: {str(e)}")
        return False

def main():
    """Run all health checks."""
    logger.info("🔍 Running health checks...")
    
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", check_dependencies),
        ("Data Files", check_data_files_status),
        ("Model", check_model),
        ("Streamlit Config", check_streamlit_config),
        ("App Test", run_basic_app_test)
    ]
    
    passed = 0
    total = len(checks)
    
    for check_name, check_func in checks:
        logger.info(f"\n--- {check_name} ---")
        if check_func():
            passed += 1
        else:
            logger.error(f"{check_name} check failed")
    
    logger.info(f"\n📊 Health Check Results: {passed}/{total} checks passed")
    
    if passed == total:
        logger.info("🎉 All checks passed! App is ready for deployment.")
        return True
    else:
        logger.error("❌ Some checks failed. Please fix the issues before deploying.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)