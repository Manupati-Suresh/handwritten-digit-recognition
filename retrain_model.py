"""
Quick model retraining script for deployment.
This creates a smaller, deployment-optimized model.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle
import os
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_optimized_model():
    """Create a smaller, deployment-optimized model."""
    
    if not os.path.exists("mnist_train.csv"):
        logger.error("Training data not found. Please ensure mnist_train.csv is available.")
        return False
    
    try:
        logger.info("Loading training data...")
        df = pd.read_csv("mnist_train.csv")
        
        # Use a subset for faster training and smaller model
        subset_size = min(10000, len(df))
        df_subset = df.sample(n=subset_size, random_state=42)
        
        X = df_subset.drop("label", axis=1).values.astype(np.float32) / 255.0
        y = df_subset["label"].values
        
        logger.info(f"Using {len(X)} samples for training")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train optimized model
        logger.info("Training optimized SVM model...")
        model = svm.SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc = accuracy_score(y_test, model.predict(X_test))
        
        logger.info(f"Training accuracy: {train_acc:.4f}")
        logger.info(f"Test accuracy: {test_acc:.4f}")
        
        # Save model with metadata
        model_data = {
            'model': model,
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'model_type': 'SVM (Optimized)',
                'training_samples': len(X_train),
                'test_accuracy': test_acc,
                'features': X_train.shape[1],
                'classes': len(np.unique(y)),
                'optimization': 'deployment_ready'
            }
        }
        
        with open("svm_digit_model.pkl", "wb") as f:
            pickle.dump(model_data, f)
        
        # Check file size
        file_size = os.path.getsize("svm_digit_model.pkl") / (1024 * 1024)
        logger.info(f"Model saved successfully ({file_size:.1f} MB)")
        
        return True
        
    except Exception as e:
        logger.error(f"Model training failed: {str(e)}")
        return False

def main():
    """Main function."""
    logger.info("Creating deployment-optimized model...")
    
    if create_optimized_model():
        logger.info("✅ Optimized model created successfully!")
        logger.info("The model is now ready for deployment on Streamlit Cloud.")
    else:
        logger.error("❌ Failed to create optimized model.")
        return False
    
    return True

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)