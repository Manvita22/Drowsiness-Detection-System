#!/bin/bash
# Drowsiness Detection System - Setup Script for Linux/Mac

echo ""
echo "========================================"
echo "Drowsiness Detection System Setup"
echo "========================================"
echo ""

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    echo "Please install Python 3.8+ from https://www.python.org/"
    exit 1
fi

echo "[1/4] Python found:"
python3 --version

# Install requirements
echo ""
echo "[2/4] Installing required packages..."
pip3 install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error: Failed to install packages"
    exit 1
fi

# Create models directory
echo ""
echo "[3/4] Creating directories..."
mkdir -p models
echo "Models directory created"

# Display next steps
echo ""
echo "[4/4] Setup Complete!"
echo ""
echo "========================================"
echo "Next Steps:"
echo "========================================"
echo ""
echo "Option 1: Train the model (first time)"
echo "   python3 train_model.py"
echo ""
echo "Option 2: Test on test dataset"
echo "   python3 test_model.py"
echo ""
echo "Option 3: Run detection system (requires webcam)"
echo "   python3 drowsiness_detection.py"
echo ""
echo "Notes:"
echo "- Training takes 5-10 minutes on CPU"
echo "- For faster training, use a GPU system"
echo "- Place alarm.wav in project root for audio alerts"
echo "- Press 'q' to exit detection system"
echo ""
