# ✍️ Handwritten Digit Recognition App

A robust Streamlit application that uses Support Vector Machine (SVM) to recognize handwritten digits (0-9) from uploaded images.

## 🚀 Features

- **Image Upload**: Support for multiple image formats (PNG, JPG, JPEG, BMP, TIFF)
- **Automatic Preprocessing**: Images are automatically resized and normalized
- **Real-time Prediction**: Instant digit recognition with confidence scores
- **Interactive UI**: Clean, responsive interface with visual feedback
- **Error Handling**: Comprehensive error handling and user feedback
- **Model Caching**: Efficient model loading with Streamlit caching

## 🛠️ Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

### Optional: Retrain the Model
If you want to retrain the model with your own data:
```bash
# Download MNIST data (optional - only needed for retraining)
python download_data.py

# Retrain the model
python train_model.py
```

## 📊 Model Information

- **Algorithm**: Support Vector Machine (SVM) with RBF kernel
- **Training Data**: MNIST Dataset
- **Input Size**: 28x28 pixels grayscale
- **Classes**: 10 digits (0-9)
- **Performance**: ~95%+ accuracy on test data

## 🌐 Deployment

This app is optimized for deployment on Streamlit Cloud:

1. Fork this repository
2. Connect your GitHub account to Streamlit Cloud
3. Deploy directly from the repository
4. The app will automatically install dependencies and start

## 📁 Project Structure

```
├── app.py                 # Main Streamlit application
├── train_model.py         # Model training script
├── retrain_model.py       # Optimized model training for deployment
├── download_data.py       # Script to download MNIST data (optional)
├── requirements.txt       # Python dependencies
├── svm_digit_model.pkl   # Trained model file (optimized for deployment)
├── utils.py              # Utility functions
├── health_check.py       # Health check and validation
├── .streamlit/
│   └── config.toml       # Streamlit configuration
└── README.md             # This file
```

## 🎯 Usage Tips

For best results when uploading images:
- Use clear, dark digits on light background
- Ensure the digit fills most of the image
- Avoid cluttered backgrounds
- Images will be automatically resized to 28x28 pixels

## 🔧 Technical Details

- **Framework**: Streamlit
- **ML Library**: scikit-learn
- **Image Processing**: PIL (Pillow)
- **Data Handling**: pandas, numpy
- **Model Format**: Pickle serialization

## 📈 Performance Optimization

- Model caching with `@st.cache_resource`
- Efficient image preprocessing
- Optimized UI rendering
- Error handling and validation
- Memory-efficient data loading

## 🤝 Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## 📄 License

This project is open source and available under the MIT License.