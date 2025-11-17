# UrbanSound8K Audio Classifier

A comprehensive Streamlit web application for audio classification using the UrbanSound8K dataset. This app demonstrates multiple machine learning approaches including SVM, Random Forest, Gradient Boosting, and Convolutional Neural Networks (CNN) for urban sound classification.

## 🎯 Features

- **Multi-Model Classification**: Compare predictions from 4 different ML models
- **Real-time Feature Extraction**: MFCC visualization and feature analysis
- **Model Performance Comparison**: View accuracy metrics and prediction confidence

## 🎵 Audio Classes

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

## 🏗️ Architecture

### Feature Extraction
- **Traditional Models (SVM, RF, GB)**: 26 features (13 MFCC + 13 Delta MFCC coefficients)
- **CNN Model**: 40 MFCC coefficients with 174 time frames

### Model Details
- **SVM**: Support Vector Machine with RBF kernel and hyperparameter tuning
- **Random Forest**: 200 trees with optimized parameters
- **Gradient Boosting**: Histogram-based gradient boosting classifier
- **CNN**: 2 Conv2D layers + Dense layers with dropout regularization

## 🙏 Acknowledgments

- **UrbanSound8K Dataset**: J. Salamon, C. Jacoby and J.P. Bello. "A Dataset and Taxonomy for Urban Sound Research", 22nd ACM International Conference on Multimedia, Orlando USA, Nov. 2014.
- **Librosa**: Audio analysis library for Python
- **Streamlit**: Framework for building ML web applications
- **TensorFlow/Keras**: Deep learning framework
