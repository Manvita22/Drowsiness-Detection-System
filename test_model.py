"""
Test script to validate trained model performance on test dataset
"""

import os
import numpy as np
import cv2
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns

IMG_SIZE = 24
MODEL_PATH = 'models/cnnCat2.h5'

def load_test_images():
    """Load test dataset"""
    dataset_path = "dataset_new/test"
    
    images = []
    labels = []
    
    # Load Closed eyes
    closed_path = os.path.join(dataset_path, "Closed")
    print(f"Loading test images from {closed_path}...")
    for filename in os.listdir(closed_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(closed_path, filename)
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    img = img / 255.0
                    images.append(img)
                    labels.append(0)
            except Exception as e:
                print(f"Error: {e}")
    
    # Load Open eyes
    open_path = os.path.join(dataset_path, "Open")
    print(f"Loading test images from {open_path}...")
    for filename in os.listdir(open_path):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(open_path, filename)
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    img = img / 255.0
                    images.append(img)
                    labels.append(1)
            except Exception as e:
                print(f"Error: {e}")
    
    X_test = np.array(images)
    y_test = np.array(labels)
    
    # Expand dimensions
    X_test = np.expand_dims(X_test, axis=-1)
    
    print(f"Loaded {len(X_test)} test images")
    print(f"Label distribution: {np.bincount(y_test)}")
    
    return X_test, y_test

def test_model():
    """Test trained model"""
    print("=" * 60)
    print("MODEL TESTING")
    print("=" * 60)
    
    # Load test data
    X_test, y_test = load_test_images()
    
    # Load model
    try:
        model = load_model(MODEL_PATH)
        print(f"Model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Make predictions
    print("\nGenerating predictions...")
    predictions = model.predict(X_test, verbose=0)
    y_pred = np.argmax(predictions, axis=1)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n{'=' * 60}")
    print(f"RESULTS")
    print(f"{'=' * 60}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Closed', 'Open']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Closed', 'Open'], 
                yticklabels=['Closed', 'Open'])
    plt.title(f'Confusion Matrix - Accuracy: {accuracy * 100:.2f}%')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('models/confusion_matrix.png')
    print("\nConfusion matrix saved to models/confusion_matrix.png")
    plt.close()
    
    # Plot prediction confidence
    closed_predictions = predictions[y_test == 0]
    open_predictions = predictions[y_test == 1]
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(closed_predictions[:, 0], bins=30, alpha=0.7, label='Pred Closed')
    plt.hist(closed_predictions[:, 1], bins=30, alpha=0.7, label='Pred Open')
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.title('Predictions for Actual Closed Eyes')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(open_predictions[:, 0], bins=30, alpha=0.7, label='Pred Closed')
    plt.hist(open_predictions[:, 1], bins=30, alpha=0.7, label='Pred Open')
    plt.xlabel('Confidence')
    plt.ylabel('Frequency')
    plt.title('Predictions for Actual Open Eyes')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('models/prediction_confidence.png')
    print("Prediction confidence plot saved to models/prediction_confidence.png")
    plt.close()

if __name__ == "__main__":
    test_model()
