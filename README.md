# UrbanSound8K Audio Classifier

A comprehensive Streamlit web application for audio classification using the UrbanSound8K dataset. This app demonstrates multiple machine learning approaches including SVM, Random Forest, Gradient Boosting, and Convolutional Neural Networks (CNN) for urban sound classification.

## 🎯 Features

- **Multi-Model Classification**: Compare predictions from 4 different ML models
- **Interactive Audio Player**: Upload and play audio files directly in the browser
- **Real-time Feature Extraction**: MFCC visualization and feature analysis
- **Model Performance Comparison**: View accuracy metrics and prediction confidence
- **Responsive Design**: Clean, professional interface built with Streamlit

## 🎵 Supported Audio Classes

The application can classify 10 types of urban sounds:
1. Air Conditioner
2. Car Horn
3. Children Playing
4. Dog Bark
5. Drilling
6. Engine Idling
7. Gun Shot
8. Jackhammer
9. Siren
10. Street Music

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/urbansound8k-classifier.git
   cd urbansound8k-classifier
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run urban_sound_classifier_app.py
   ```

4. **Open your browser**
   - The app will automatically open at `http://localhost:8501`
   - Upload an audio file (.wav, .mp3, .flac, .ogg) to start classification

## 📊 Model Performance

Based on 10-fold cross-validation on the UrbanSound8K dataset:

| Model | Average Accuracy |
|-------|------------------|
| SVM | 47.95% |
| Random Forest | 50.08% |
| Gradient Boosting | 52.54% |
| **CNN** | **91.55%** |

## 🏗️ Architecture

### Feature Extraction
- **Traditional Models (SVM, RF, GB)**: 26 features (13 MFCC + 13 Delta MFCC coefficients)
- **CNN Model**: 40 MFCC coefficients with 174 time frames

### Model Details
- **SVM**: Support Vector Machine with RBF kernel and hyperparameter tuning
- **Random Forest**: 200 trees with optimized parameters
- **Gradient Boosting**: Histogram-based gradient boosting classifier
- **CNN**: 2 Conv2D layers + Dense layers with dropout regularization

## 📁 Project Structure

```
urbansound8k-classifier/
├── urban_sound_classifier_app.py    # Main Streamlit application
├── requirements.txt                 # Python dependencies
├── README.md                        # Project documentation
├── models/                          # Trained model files (to be added)
│   ├── svm_model.pkl
│   ├── rf_model.pkl
│   ├── gb_model.pkl
│   └── cnn_model.h5
└── data/                           # Dataset directory (to be added)
    └── UrbanSound8K/
        ├── fold1/
        ├── fold2/
        └── ...
```

## 🔧 Advanced Setup (Optional)

For full functionality with actual trained models:

### 1. Download the UrbanSound8K Dataset
```bash
# Download from: https://urbansounddataset.weebly.com/urbansound8k.html
# Extract to ./data/UrbanSound8K/
```

### 2. Train the Models
```python
# Use the provided practicum code to train models
# Save trained models to ./models/ directory
```

### 3. Update Model Loading
```python
# Modify the app to load actual trained models instead of using simulations
# Update simulate_predictions() function with real model inference
```

## 💡 Usage Tips

1. **Best Audio Quality**: Use clear, single-class audio samples for best results
2. **File Formats**: WAV files typically provide the most consistent results
3. **Audio Length**: The models work best with audio clips of 2-4 seconds
4. **Background Noise**: Minimize background noise for more accurate predictions

## 🛠️ Customization

### Adding New Models
1. Implement feature extraction in `extract_features()`
2. Add model training code
3. Update the prediction function
4. Add new tab in the Streamlit interface

### Modifying Feature Extraction
- Adjust MFCC parameters in `extract_mfcc_features()`
- Add new audio features (spectral, temporal, etc.)
- Experiment with different preprocessing techniques

## 📈 Performance Optimization

- **Feature Caching**: Cache extracted features for repeated analyses
- **Model Loading**: Load models once at startup for faster predictions
- **Batch Processing**: Process multiple files simultaneously
- **GPU Acceleration**: Use TensorFlow GPU for CNN inference

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **UrbanSound8K Dataset**: J. Salamon, C. Jacoby and J.P. Bello. "A Dataset and Taxonomy for Urban Sound Research", 22nd ACM International Conference on Multimedia, Orlando USA, Nov. 2014.
- **Librosa**: Audio analysis library for Python
- **Streamlit**: Framework for building ML web applications
- **TensorFlow/Keras**: Deep learning framework

## 📞 Support

- Create an issue on GitHub for bug reports
- Star the repository if you find it helpful
- Share your improvements and suggestions

---

**Built with ❤️ for the audio ML community**
