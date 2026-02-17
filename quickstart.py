"""
Quick start script - Guides you through the process
"""

import os
import sys
import subprocess

def print_header(title):
    print("\n" + "=" * 60)
    print(title.center(60))
    print("=" * 60)

def main():
    print_header("DROWSINESS DETECTION SYSTEM - QUICK START")
    
    while True:
        print("\nWhat would you like to do?")
        print("\n1. Setup (Install dependencies)")
        print("2. Train Model (Train CNN on your dataset)")
        print("3. Test Model (Evaluate on test data)")
        print("4. Run Detection (Real-time drowsiness detection with webcam)")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            print_header("INSTALLATION")
            print("\nInstalling dependencies from requirements.txt...")
            result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            if result.returncode == 0:
                os.makedirs("models", exist_ok=True)
                print("\n✓ Setup complete!")
            else:
                print("\n✗ Setup failed!")
        
        elif choice == "2":
            print_header("MODEL TRAINING")
            print("\nTraining CNN model on your dataset...")
            print("This may take 5-10 minutes on CPU")
            result = subprocess.run([sys.executable, "train_model.py"])
            if result.returncode == 0:
                print("\n✓ Training complete! Model saved to models/cnnCat2.h5")
            else:
                print("\n✗ Training failed!")
        
        elif choice == "3":
            print_header("MODEL TESTING")
            if not os.path.exists("models/cnnCat2.h5"):
                print("\n✗ Model not found! Please train the model first (option 2)")
            else:
                print("\nTesting model on test dataset...")
                result = subprocess.run([sys.executable, "test_model.py"])
                if result.returncode == 0:
                    print("\n✓ Testing complete!")
                else:
                    print("\n✗ Testing failed!")
        
        elif choice == "4":
            print_header("DROWSINESS DETECTION")
            if not os.path.exists("models/cnnCat2.h5"):
                print("\n✗ Model not found! Please train the model first (option 2)")
            else:
                print("\nStarting real-time drowsiness detection...")
                print("Make sure your webcam is connected")
                print("Press 'q' to stop the detection system")
                result = subprocess.run([sys.executable, "drowsiness_detection.py"])
        
        elif choice == "5":
            print("\nThank you for using Drowsiness Detection System!")
            break
        
        else:
            print("\n✗ Invalid choice! Please enter 1-5")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
