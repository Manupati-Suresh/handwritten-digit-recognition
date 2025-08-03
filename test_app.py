"""
Simple test script to verify app functionality.
"""

import numpy as np
from PIL import Image
import pickle
import os
import sys
from simple_model import SimpleDigitClassifier

def test_model_loading():
    """Test if model can be loaded correctly."""
    print("Testing model loading...")
    
    if not os.path.exists("svm_digit_model.pkl"):
        print("❌ Model file not found")
        return False
    
    try:
        with open("svm_digit_model.pkl", "rb") as f:
            model_data = pickle.load(f)
        
        # Handle both formats
        if isinstance(model_data, dict) and 'model' in model_data:
            model = model_data['model']
            print(f"✅ Model loaded with metadata: {model_data.get('metadata', {})}")
        else:
            model = model_data
            print("✅ Model loaded (legacy format)")
        
        # Test prediction
        test_input = np.random.rand(1, 784)
        prediction = model.predict(test_input)
        print(f"✅ Test prediction: {prediction[0]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def test_image_processing():
    """Test image preprocessing functionality."""
    print("\nTesting image processing...")
    
    try:
        # Create a test image
        test_image = Image.new('L', (50, 50), color=128)
        
        # Test preprocessing (simplified version)
        if test_image.mode != 'L':
            test_image = test_image.convert('L')
        
        test_image = test_image.resize((28, 28), Image.Resampling.LANCZOS)
        img_array = np.array(test_image)
        
        if np.mean(img_array) > 127:
            img_array = 255 - img_array
        
        img_array = img_array.astype(np.float32) / 255.0
        img_array = img_array.reshape(1, -1)
        
        print(f"✅ Image processed successfully, shape: {img_array.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Image processing failed: {e}")
        return False

def test_dependencies():
    """Test if all required dependencies are available."""
    print("\nTesting dependencies...")
    
    required_modules = [
        'streamlit',
        'numpy',
        'pandas',
        'sklearn',
        'PIL'
    ]
    
    for module in required_modules:
        try:
            if module == 'PIL':
                import PIL
            elif module == 'sklearn':
                import sklearn
            else:
                __import__(module)
            print(f"✅ {module} available")
        except ImportError:
            print(f"❌ {module} not available")
            return False
    
    return True

def main():
    """Run all tests."""
    print("🧪 Running app tests...\n")
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Model Loading", test_model_loading),
        ("Image Processing", test_image_processing)
    ]
    
    passed = 0
    for test_name, test_func in tests:
        if test_func():
            passed += 1
        else:
            print(f"❌ {test_name} test failed")
    
    print(f"\n📊 Test Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! App should work correctly.")
        return True
    else:
        print("❌ Some tests failed. Please fix the issues.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)