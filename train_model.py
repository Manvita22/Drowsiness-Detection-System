import os
import numpy as np
import cv2
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Configuration
IMG_SIZE = 24
BATCH_SIZE = 32
EPOCHS = 10  # Reduced from 15 for faster training
LEARNING_RATE = 0.001

def load_images_from_folder(folder, label):
    """Load images from folder and assign label"""
    images = []
    labels = []
    
    print(f"Loading images from {folder}...")
    for filename in os.listdir(folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder, filename)
            try:
                # Read image in grayscale
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # Resize to IMG_SIZE x IMG_SIZE
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    # Normalize pixel values to 0-1
                    img = img / 255.0
                    images.append(img)
                    labels.append(label)
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
    
    print(f"Loaded {len(images)} images from {folder}")
    return np.array(images), np.array(labels)

def prepare_data():
    """Prepare training and testing data"""
    dataset_path = "dataset_new"
    
    # Load Closed eyes (label: 0)
    closed_train, closed_train_labels = load_images_from_folder(
        os.path.join(dataset_path, "train", "Closed"), 0
    )
    closed_test, closed_test_labels = load_images_from_folder(
        os.path.join(dataset_path, "test", "Closed"), 0
    )
    
    # Load Open eyes (label: 1)
    open_train, open_train_labels = load_images_from_folder(
        os.path.join(dataset_path, "train", "Open"), 1
    )
    open_test, open_test_labels = load_images_from_folder(
        os.path.join(dataset_path, "test", "Open"), 1
    )
    
    # Concatenate all data
    X_train = np.concatenate([closed_train, open_train], axis=0)
    y_train = np.concatenate([closed_train_labels, open_train_labels], axis=0)
    
    X_test = np.concatenate([closed_test, open_test], axis=0)
    y_test = np.concatenate([closed_test_labels, open_test_labels], axis=0)
    
    # Expand dimensions to add channel (24, 24) -> (24, 24, 1)
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    
    # Shuffle the data
    shuffle_idx_train = np.random.permutation(len(X_train))
    X_train = X_train[shuffle_idx_train]
    y_train = y_train[shuffle_idx_train]
    
    shuffle_idx_test = np.random.permutation(len(X_test))
    X_test = X_test[shuffle_idx_test]
    y_test = y_test[shuffle_idx_test]
    
    print(f"\nData Summary:")
    print(f"Training set size: {X_train.shape}")
    print(f"Testing set size: {X_test.shape}")
    print(f"Training labels distribution: {np.bincount(y_train)}")
    print(f"Testing labels distribution: {np.bincount(y_test)}")
    
    return X_train, y_train, X_test, y_test

def build_model():
    """Build CNN model"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        MaxPooling2D((2, 2)),
        
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        
        Dense(2, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel Architecture:")
    model.summary()
    
    return model

def train_model():
    """Train the model"""
    print("=" * 60)
    print("DROWSINESS DETECTION - CNN MODEL TRAINING")
    print("=" * 60)
    
    # Prepare data
    X_train, y_train, X_test, y_test = prepare_data()
    
    # Build model
    model = build_model()
    
    # Data augmentation for better generalization
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2
    )
    
    print("\nTraining model...")
    history = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        epochs=EPOCHS,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Save model
    model_path = "models/cnnCat2.h5"
    os.makedirs("models", exist_ok=True)
    model.save(model_path)
    print(f"\nModel saved to {model_path}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('models/training_history.png')
    print("Training history saved to models/training_history.png")
    plt.close()

if __name__ == "__main__":
    train_model()
