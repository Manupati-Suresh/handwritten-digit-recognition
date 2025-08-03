
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_data(train_file="mnist_train.csv", test_file="mnist_test.csv"):
    """Load and preprocess MNIST data with error handling."""
    try:
        logger.info("Loading training data...")
        if not os.path.exists(train_file):
            raise FileNotFoundError(f"Training file '{train_file}' not found")
        
        # Load training data
        df_train = pd.read_csv(train_file)
        logger.info(f"Loaded {len(df_train)} training samples")
        
        # Separate features and labels
        X_train_full = df_train.drop("label", axis=1).values.astype(np.float32) / 255.0
        y_train_full = df_train["label"].values
        
        # Load test data if available
        X_test, y_test = None, None
        if os.path.exists(test_file):
            logger.info("Loading test data...")
            df_test = pd.read_csv(test_file)
            X_test = df_test.drop("label", axis=1).values.astype(np.float32) / 255.0
            y_test = df_test["label"].values
            logger.info(f"Loaded {len(df_test)} test samples")
        
        return X_train_full, y_train_full, X_test, y_test
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def train_model(X_train, y_train, X_val, y_val, model_params=None):
    """Train SVM model with validation."""
    try:
        if model_params is None:
            model_params = {
                'kernel': 'rbf',  # RBF kernel often performs better than linear
                'C': 1.0,
                'gamma': 'scale',
                'random_state': 42
            }
        
        logger.info(f"Training SVM with parameters: {model_params}")
        
        # Create and train model
        model = svm.SVC(**model_params)
        
        # Use a subset for faster training if dataset is large
        if len(X_train) > 10000:
            logger.info("Using subset of data for faster training...")
            subset_size = 10000
            indices = np.random.choice(len(X_train), subset_size, replace=False)
            X_train_subset = X_train[indices]
            y_train_subset = y_train[indices]
        else:
            X_train_subset = X_train
            y_train_subset = y_train
        
        model.fit(X_train_subset, y_train_subset)
        logger.info("Model training completed")
        
        # Validate model
        val_predictions = model.predict(X_val)
        val_accuracy = accuracy_score(y_val, val_predictions)
        logger.info(f"Validation accuracy: {val_accuracy:.4f}")
        
        return model, val_accuracy
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    try:
        logger.info("Evaluating model...")
        
        # Make predictions
        predictions = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        
        logger.info(f"Test accuracy: {accuracy:.4f}")
        
        # Print detailed classification report
        print("\nClassification Report:")
        print(classification_report(y_test, predictions))
        
        # Print confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, predictions)
        print(cm)
        
        return accuracy, predictions
        
    except Exception as e:
        logger.error(f"Error evaluating model: {str(e)}")
        raise

def save_model(model, filename="svm_digit_model.pkl", metadata=None):
    """Save model with metadata."""
    try:
        # Create model data with metadata
        model_data = {
            'model': model,
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'model_type': 'SVM',
                'sklearn_version': '1.3.0',  # Update as needed
                **(metadata or {})
            }
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filename}")
        
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise

def main():
    """Main training pipeline."""
    try:
        logger.info("Starting model training pipeline...")
        
        # Load data
        X_train_full, y_train_full, X_test, y_test = load_data()
        
        # Split training data into train/validation if no separate test set
        if X_test is None:
            logger.info("No separate test set found, splitting training data...")
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full, y_train_full, 
                test_size=0.2, 
                random_state=42,
                stratify=y_train_full
            )
            X_test, y_test = X_val, y_val
        else:
            # Use all training data for training
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_full, y_train_full,
                test_size=0.1,
                random_state=42,
                stratify=y_train_full
            )
        
        logger.info(f"Training set size: {len(X_train)}")
        logger.info(f"Validation set size: {len(X_val)}")
        logger.info(f"Test set size: {len(X_test)}")
        
        # Train model
        model, val_accuracy = train_model(X_train, y_train, X_val, y_val)
        
        # Evaluate on test set
        test_accuracy, _ = evaluate_model(model, X_test, y_test)
        
        # Save model with metadata
        metadata = {
            'validation_accuracy': val_accuracy,
            'test_accuracy': test_accuracy,
            'training_samples': len(X_train),
            'features': X_train.shape[1]
        }
        
        save_model(model, metadata=metadata)
        
        logger.info("Training pipeline completed successfully!")
        print(f"\nFinal Results:")
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
