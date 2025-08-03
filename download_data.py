"""
Script to download MNIST data for training (optional).
This is not needed for the deployed app, only for retraining.
"""

import os
import urllib.request
import pandas as pd
from sklearn.datasets import fetch_openml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_mnist_data():
    """Download MNIST data and save as CSV files."""
    try:
        logger.info("Downloading MNIST data from OpenML...")
        
        # Download MNIST data
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X, y = mnist.data, mnist.target.astype(int)
        
        logger.info(f"Downloaded {len(X)} samples")
        
        # Create train/test split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Save as CSV
        logger.info("Saving training data...")
        train_df = pd.DataFrame(X_train)
        train_df.insert(0, 'label', y_train)
        train_df.to_csv('mnist_train.csv', index=False)
        
        logger.info("Saving test data...")
        test_df = pd.DataFrame(X_test)
        test_df.insert(0, 'label', y_test)
        test_df.to_csv('mnist_test.csv', index=False)
        
        logger.info("✅ MNIST data downloaded and saved successfully!")
        logger.info("You can now run train_model.py to retrain the model")
        
    except Exception as e:
        logger.error(f"Failed to download data: {str(e)}")
        logger.error("You may need to install additional dependencies:")
        logger.error("pip install scikit-learn pandas")

if __name__ == "__main__":
    if not os.path.exists('mnist_train.csv'):
        download_mnist_data()
    else:
        logger.info("MNIST data already exists")