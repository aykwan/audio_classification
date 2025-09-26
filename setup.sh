#!/bin/bash

# UrbanSound8K Audio Classifier Setup Script
echo "🎵 Setting up UrbanSound8K Audio Classifier..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8+ first."
    exit 1
fi

echo "✅ Python found: $(python3 --version)"

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "🔧 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing dependencies..."
pip install -r requirements.txt

echo "🎉 Setup complete!"
echo ""
echo "🚀 To run the application:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Run the app: streamlit run urban_sound_classifier_app.py"
echo ""
echo "📂 Optional: Download UrbanSound8K dataset to ./data/UrbanSound8K/"
echo "🤖 Optional: Train models and save to ./models/ directory"
