
import streamlit as st
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import os
import random
import warnings
import io
from typing import Tuple, Optional, Dict, Any
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
random_seed = 42
np.random.seed(random_seed)
tf.random.set_seed(random_seed)
random.seed(random_seed)

# Configure Streamlit page
st.set_page_config(
    page_title="UrbanSound8K Audio Classifier", 
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define class names for UrbanSound8K
CLASS_NAMES = {
    0: 'air_conditioner',
    1: 'car_horn', 
    2: 'children_playing',
    3: 'dog_bark',
    4: 'drilling',
    5: 'engine_idling',
    6: 'gun_shot',
    7: 'jackhammer',
    8: 'siren',
    9: 'street_music'
}

class AudioClassifier:
    """Main audio classification class handling all models and predictions"""

    def __init__(self):
        self.models = {}
        self.model_loaded = {
            'SVM': False,
            'Random Forest': False, 
            'Gradient Boosting': False,
            'CNN': False
        }
        self.load_models()

    def load_models(self):
        """Load pre-trained models if available"""
        model_files = {
            'SVM': 'models/svm_model.pkl',
            'Random Forest': 'models/rf_model.pkl',
            'Gradient Boosting': 'models/gb_model.pkl',
            'CNN': 'models/cnn_model.h5'
        }

        for model_name, file_path in model_files.items():
            try:
                if os.path.exists(file_path):
                    if model_name == 'CNN':
                        self.models[model_name] = load_model(file_path)
                    else:
                        with open(file_path, 'rb') as f:
                            self.models[model_name] = pickle.load(f)
                    self.model_loaded[model_name] = True
                    st.sidebar.success(f"✅ {model_name} model loaded")
                else:
                    st.sidebar.info(f"ℹ️ {model_name} model not found - using demo mode")
            except Exception as e:
                st.sidebar.error(f"❌ Error loading {model_name}: {str(e)}")

    def extract_mfcc_features(self, audio_file, n_mfcc: int = 40, max_len: int = 174) -> Optional[np.ndarray]:
        """Extract MFCC features for CNN model"""
        try:
            if hasattr(audio_file, 'read'):
                audio_data = audio_file.read()
                y, sr = librosa.load(io.BytesIO(audio_data), sr=None, mono=True)
            else:
                y, sr = librosa.load(audio_file, sr=None, mono=True)

            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

            # Pad/truncate to fixed length
            if mfcc.shape[1] < max_len:
                pad_width = max_len - mfcc.shape[1]
                mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
            else:
                mfcc = mfcc[:, :max_len]

            return mfcc
        except Exception as e:
            st.error(f"Error extracting MFCC features: {e}")
            return None

    def extract_traditional_features(self, audio_file) -> Optional[np.ndarray]:
        """Extract traditional features for SVM/RF/GB models"""
        try:
            if hasattr(audio_file, 'read'):
                audio_data = audio_file.read()
                y, sr = librosa.load(io.BytesIO(audio_data), sr=None, mono=True)
            else:
                y, sr = librosa.load(audio_file, sr=None, mono=True)

            # Extract MFCCs and delta MFCCs (as per practicum)
            mfcc = librosa.feature.mfcc(
                y=y, sr=sr, n_mfcc=13, n_fft=2048, 
                n_mels=40, fmin=20, fmax=sr//2
            )

            width = min(9, mfcc.shape[1])
            if width < 3:
                width = 3
            if width % 2 == 0:
                width -= 1

            mfcc_delta = librosa.feature.delta(mfcc, width=width)

            # Combine MFCC and delta MFCC
            combined = np.concatenate([mfcc, mfcc_delta], axis=0)
            combined_mean = np.mean(combined, axis=1)

            return combined_mean
        except Exception as e:
            st.error(f"Error extracting traditional features: {e}")
            return None

    def predict(self, features: np.ndarray, model_name: str) -> Tuple[int, float, np.ndarray]:
        """Make prediction using specified model"""
        if self.model_loaded[model_name]:
            # Use actual trained model
            model = self.models[model_name]

            if model_name == 'CNN':
                # Prepare features for CNN
                features_reshaped = features.reshape(1, 40, 174, 1)
                probabilities = model.predict(features_reshaped, verbose=0)[0]
                prediction = np.argmax(probabilities)
                confidence = probabilities[prediction]
            else:
                # Traditional ML models
                features_reshaped = features.reshape(1, -1)
                probabilities = model.predict_proba(features_reshaped)[0]
                prediction = np.argmax(probabilities)
                confidence = probabilities[prediction]
        else:
            # Demo mode with simulated predictions
            np.random.seed(hash(str(features.sum())) % 2**32)  # Deterministic based on features

            if model_name == 'SVM':
                probabilities = np.random.dirichlet(np.ones(10) * 2)
            elif model_name == 'Random Forest':
                probabilities = np.random.dirichlet(np.ones(10) * 3)
            elif model_name == 'Gradient Boosting':
                probabilities = np.random.dirichlet(np.ones(10) * 2.5)
            else:  # CNN
                probabilities = np.random.dirichlet(np.ones(10) * 4)

            prediction = np.argmax(probabilities)
            confidence = probabilities[prediction]

        return prediction, confidence, probabilities

    @staticmethod
    def build_cnn_model():
        """Build the CNN model architecture from the practicum"""
        model = Sequential([
            Input(shape=(40, 174, 1)),
            Conv2D(32, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.3),
            Conv2D(64, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Dropout(0.3),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(10, activation='softmax')
        ])

        model.compile(optimizer='adam', 
                      loss='sparse_categorical_crossentropy', 
                      metrics=['accuracy'])
        return model

class AudioVisualizer:
    """Handles all visualization tasks"""

    @staticmethod
    def plot_mfcc_features(mfcc: np.ndarray) -> plt.Figure:
        """Plot MFCC features visualization"""
        fig, ax = plt.subplots(figsize=(14, 8))
        librosa.display.specshow(
            mfcc, 
            x_axis='time', 
            ax=ax, 
            cmap='viridis',
            hop_length=512
        )
        im = ax.imshow(mfcc, aspect='auto', origin='lower', cmap='viridis')
        cbar = plt.colorbar(im, ax=ax, format='%+2.0f dB')
        plt.title('MFCC Features', fontsize=16, fontweight='bold')
        plt.xlabel('Time (s)', fontsize=12)
        plt.ylabel('MFCC Coefficients', fontsize=12)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_waveform(y: np.ndarray, sr: int) -> plt.Figure:
        """Plot audio waveform"""
        fig, ax = plt.subplots(figsize=(14, 4))
        time = np.linspace(0, len(y)/sr, len(y))
        ax.plot(time, y, alpha=0.7, color='blue')
        ax.set_xlabel('Time (s)', fontsize=12)
        ax.set_ylabel('Amplitude', fontsize=12)
        ax.set_title('Audio Waveform', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_prediction_probabilities(probabilities: np.ndarray, model_name: str) -> plt.Figure:
        """Plot prediction probabilities as bar chart"""
        fig, ax = plt.subplots(figsize=(14, 8))

        classes = [CLASS_NAMES[i].replace('_', ' ').title() for i in range(10)]
        colors = plt.cm.Set3(np.linspace(0, 1, 10))

        bars = ax.bar(classes, probabilities, color=colors, alpha=0.8, edgecolor='black', linewidth=1)

        # Highlight the predicted class
        max_idx = np.argmax(probabilities)
        bars[max_idx].set_color('orange')
        bars[max_idx].set_edgecolor('red')
        bars[max_idx].set_linewidth(2)

        ax.set_xlabel('Sound Classes', fontsize=12, fontweight='bold')
        ax.set_ylabel('Probability', fontsize=12, fontweight='bold')
        ax.set_title(f'{model_name} - Prediction Probabilities', fontsize=16, fontweight='bold')
        ax.set_ylim(0, max(probabilities) * 1.1)

        # Add value labels on bars
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{prob:.1%}',
                    ha='center', va='bottom', fontweight='bold')

        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        return fig

def create_comparison_chart(all_predictions: Dict[str, Dict]) -> plt.Figure:
    """Create a comparison chart of all model predictions"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    axes = [ax1, ax2, ax3, ax4]
    models = list(all_predictions.keys())

    for i, (model, ax) in enumerate(zip(models, axes)):
        probabilities = all_predictions[model]['probabilities']
        classes = [CLASS_NAMES[j].replace('_', ' ').title() for j in range(10)]

        bars = ax.bar(classes, probabilities, alpha=0.7)
        max_idx = np.argmax(probabilities)
        bars[max_idx].set_color('orange')

        ax.set_title(f'{model}', fontweight='bold')
        ax.set_ylabel('Probability')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)

        # Add confidence score
        confidence = all_predictions[model]['confidence']
        ax.text(0.02, 0.98, f'Confidence: {confidence:.1%}', 
                transform=ax.transAxes, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.suptitle('Model Comparison - Prediction Probabilities', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def main():
    """Main Streamlit application"""

    # Initialize classifier
    if 'classifier' not in st.session_state:
        st.session_state.classifier = AudioClassifier()

    classifier = st.session_state.classifier

    # Header
    st.title("🎵 UrbanSound8K Audio Classification System")
    st.markdown("### *Advanced Multi-Model Audio Classification with Real-Time Analysis*")
    st.markdown("---")

    # Description
    st.markdown("""
    This application demonstrates state-of-the-art audio classification using the UrbanSound8K dataset. 
    Upload an audio file to analyze it with multiple machine learning models and compare their predictions.

    **Supported Models:**
    - 🎯 **SVM** (Support Vector Machine) - Traditional ML approach
    - 🌲 **Random Forest** - Ensemble method with decision trees  
    - 🚀 **Gradient Boosting** - Advanced boosting technique
    - 🧠 **CNN** (Convolutional Neural Network) - Deep learning approach
    """)

    # Sidebar
    st.sidebar.title("📊 Model Information")

    # Model status
    st.sidebar.subheader("🔧 Model Status")
    for model_name, loaded in classifier.model_loaded.items():
        status = "🟢 Loaded" if loaded else "🟡 Demo Mode"
        st.sidebar.write(f"**{model_name}:** {status}")

    st.sidebar.markdown("---")

    # Performance metrics
    st.sidebar.subheader("📈 Cross-Validation Results")
    performance_data = {
        'Model': ['SVM', 'Random Forest', 'Gradient Boosting', 'CNN'],
        'Accuracy': ['47.95%', '50.08%', '52.54%', '91.55%']
    }
    st.sidebar.table(pd.DataFrame(performance_data))

    st.sidebar.markdown("---")

    # Class information
    st.sidebar.subheader("🎯 Audio Classes")
    for idx, name in CLASS_NAMES.items():
        emoji = ['❄️', '🚗', '👶', '🐕', '🔧', '🚙', '🔫', '🔨', '🚨', '🎵'][idx]
        st.sidebar.write(f"**{idx}.** {emoji} {name.replace('_', ' ').title()}")

    # Main interface
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("📤 Upload Audio File")
        uploaded_file = st.file_uploader(
            "Choose an audio file", 
            type=['wav', 'mp3', 'flac', 'ogg', 'm4a'],
            help="Upload an audio file in WAV, MP3, FLAC, OGG, or M4A format"
        )

        if uploaded_file is not None:
            # File information
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            st.info(f"📊 File size: {uploaded_file.size:,} bytes")

            # Audio player
            st.subheader("🎧 Audio Player")
            st.audio(uploaded_file, format='audio/wav')

            # Processing options
            st.subheader("⚙️ Processing Options")
            show_waveform = st.checkbox("Show Waveform", value=True)
            show_features = st.checkbox("Show MFCC Features", value=True)
            show_comparison = st.checkbox("Show Model Comparison", value=True)

    with col2:
        if uploaded_file is not None:
            st.subheader("🔍 Audio Analysis Results")

            # Reset file pointer
            uploaded_file.seek(0)

            # Processing indicator
            with st.spinner("🔄 Processing audio file..."):
                # Extract features
                mfcc_features = classifier.extract_mfcc_features(uploaded_file)
                uploaded_file.seek(0)  # Reset for traditional features
                traditional_features = classifier.extract_traditional_features(uploaded_file)

            if mfcc_features is not None and traditional_features is not None:
                st.success("✅ Feature extraction completed!")

                # Optional visualizations
                if show_waveform:
                    uploaded_file.seek(0)
                    audio_data = uploaded_file.read()
                    y, sr = librosa.load(io.BytesIO(audio_data), sr=None, mono=True)

                    st.subheader("📊 Audio Waveform")
                    fig_wave = AudioVisualizer.plot_waveform(y, sr)
                    st.pyplot(fig_wave)
                    plt.close(fig_wave)

                if show_features:
                    st.subheader("📈 MFCC Features")
                    fig_mfcc = AudioVisualizer.plot_mfcc_features(mfcc_features)
                    st.pyplot(fig_mfcc)
                    plt.close(fig_mfcc)

                # Model predictions
                st.subheader("🤖 Classification Results")

                models = ['SVM', 'Random Forest', 'Gradient Boosting', 'CNN']
                all_predictions = {}

                # Create results summary
                results_data = []

                for model_name in models:
                    with st.spinner(f"Running {model_name} prediction..."):
                        # Choose appropriate features
                        features = mfcc_features if model_name == 'CNN' else traditional_features
                        pred_class, confidence, probabilities = classifier.predict(features, model_name)

                        all_predictions[model_name] = {
                            'prediction': pred_class,
                            'confidence': confidence,
                            'probabilities': probabilities,
                            'class_name': CLASS_NAMES[pred_class]
                        }

                        results_data.append({
                            'Model': model_name,
                            'Predicted Class': pred_class,
                            'Class Name': CLASS_NAMES[pred_class].replace('_', ' ').title(),
                            'Confidence': f"{confidence:.2%}"
                        })

                # Display results summary
                st.subheader("📋 Results Summary")
                results_df = pd.DataFrame(results_data)
                st.table(results_df)

                # Detailed model results in tabs
                st.subheader("🔍 Detailed Analysis")
                tabs = st.tabs(models)

                for i, model_name in enumerate(models):
                    with tabs[i]:
                        pred_data = all_predictions[model_name]

                        # Key metrics
                        col_a, col_b, col_c = st.columns([1, 1, 1])

                        with col_a:
                            st.metric("Predicted Class", pred_data['prediction'])
                        with col_b:
                            st.metric("Class Name", pred_data['class_name'].replace('_', ' ').title())
                        with col_c:
                            st.metric("Confidence", f"{pred_data['confidence']:.2%}")

                        # Probability distribution
                        fig_prob = AudioVisualizer.plot_prediction_probabilities(
                            pred_data['probabilities'], model_name
                        )
                        st.pyplot(fig_prob)
                        plt.close(fig_prob)

                        # Top predictions
                        st.subheader("🏆 Top 3 Predictions")
                        top_3_indices = np.argsort(pred_data['probabilities'])[::-1][:3]

                        for rank, idx in enumerate(top_3_indices, 1):
                            emoji = ['🥇', '🥈', '🥉'][rank-1]
                            prob = pred_data['probabilities'][idx]
                            class_name = CLASS_NAMES[idx].replace('_', ' ').title()
                            st.write(f"{emoji} **{rank}. {class_name}** - {prob:.2%}")

                # Model comparison
                if show_comparison and len(all_predictions) > 1:
                    st.subheader("⚖️ Model Comparison")

                    fig_comparison = create_comparison_chart(all_predictions)
                    st.pyplot(fig_comparison)
                    plt.close(fig_comparison)

                    # Consensus analysis
                    st.subheader("🎯 Consensus Analysis")
                    predictions = [all_predictions[model]['prediction'] for model in models]
                    confidences = [all_predictions[model]['confidence'] for model in models]

                    # Most common prediction
                    most_common_pred = max(set(predictions), key=predictions.count)
                    agreement_count = predictions.count(most_common_pred)

                    col_x, col_y, col_z = st.columns([1, 1, 1])
                    with col_x:
                        st.metric("Consensus Class", CLASS_NAMES[most_common_pred].replace('_', ' ').title())
                    with col_y:
                        st.metric("Model Agreement", f"{agreement_count}/{len(models)}")
                    with col_z:
                        st.metric("Avg Confidence", f"{np.mean(confidences):.2%}")

            else:
                st.error("❌ Failed to extract features from the audio file. Please try a different file.")

    # Technical details
    with st.expander("🔧 Technical Implementation Details"):
        st.markdown("""
        ### Feature Extraction

        **Traditional Models (SVM, Random Forest, Gradient Boosting):**
        - 13 MFCC coefficients
        - 13 Delta MFCC coefficients  
        - Features averaged over time (26 total features)
        - Parameters: n_fft=2048, n_mels=40, fmin=20Hz, fmax=Nyquist

        **CNN Model:**
        - 40 MFCC coefficients 
        - 174 time frames (padded/truncated)
        - Input shape: (40, 174, 1)
        - Raw spectral-temporal representation

        ### Model Architectures

        **SVM:** RBF kernel with hyperparameter tuning (C, gamma)  
        **Random Forest:** 200 trees with optimized depth and split parameters
        **Gradient Boosting:** Histogram-based gradient boosting with L2 regularization
        **CNN:** 2 Conv2D layers (32, 64 filters) + Dense layers with dropout (30%)

        ### Performance Notes
        - All models trained with 10-fold cross-validation on UrbanSound8K
        - CNN achieves 91.55% accuracy (best performing)
        - Traditional models: 48-53% accuracy range
        - Demo mode uses deterministic simulated predictions when models not loaded
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 14px;'>
    🎵 <strong>UrbanSound8K Audio Classifier</strong> | Built with Streamlit & TensorFlow<br>
    Based on J. Salamon et al. "A Dataset and Taxonomy for Urban Sound Research" (ACM-MM'14)
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
