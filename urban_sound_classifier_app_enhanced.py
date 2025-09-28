
import streamlit as st
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import warnings
from typing import Dict, Any, Optional
warnings.filterwarnings('ignore')

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

class AudioDataLoader:
    """Handles loading audio files and CSV results"""

    def __init__(self):
        self.data_folder = "Data"
        self.csv_file = "all_models_predictions_comparison.csv"
        self.results_df = None
        self.audio_files = []
        self.load_data()

    def load_data(self):
        """Load the CSV results and discover audio files"""
        try:
            # Load CSV results
            if os.path.exists(self.csv_file):
                self.results_df = pd.read_csv(self.csv_file)
                st.sidebar.success(f"Loaded {len(self.results_df)} test results")
            else:
                st.sidebar.error(f"Results file not found: {self.csv_file}")
                return

            # Discover audio files in Data folder
            audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.ogg', '*.m4a']
            for ext in audio_extensions:
                pattern = os.path.join(self.data_folder, "**", ext)
                self.audio_files.extend(glob.glob(pattern, recursive=True))

            # Also check for files directly in Data folder
            for ext in audio_extensions:
                pattern = os.path.join(self.data_folder, ext)
                self.audio_files.extend(glob.glob(pattern, recursive=False))

            # Extract just the filenames for dropdown
            self.audio_files = [os.path.basename(f) for f in self.audio_files]
            self.audio_files = sorted(list(set(self.audio_files)))  # Remove duplicates and sort

            st.sidebar.info(f"Found {len(self.audio_files)} audio files")

        except Exception as e:
            st.sidebar.error(f"Error loading data: {str(e)}")

    def get_file_results(self, filename: str) -> Optional[Dict[str, Any]]:
        """Get results for a specific audio file from CSV"""
        if self.results_df is None:
            return None

        # Find the row matching the filename
        file_results = self.results_df[self.results_df['slice_file_name'] == filename]

        if file_results.empty:
            return None

        # Get the first match (there should only be one)
        result = file_results.iloc[0]

        return {
            'filename': result['slice_file_name'],
            'actual_class': result['actual_class'],
            'actual_class_name': result['actual_class_name'],
            'fold': result.get('fold', 'Unknown'),
            'models': {
                'SVM': {
                    'predicted': result['svm_predicted'],
                    'predicted_name': result['svm_predicted_name'],
                    'confidence': result['svm_confidence'],
                    'correct': bool(result['svm_correct'])
                },
                'Random Forest': {
                    'predicted': result['rf_predicted'],
                    'predicted_name': result['rf_predicted_name'],
                    'confidence': result['rf_confidence'],
                    'correct': bool(result['rf_correct'])
                },
                'Gradient Boosting': {
                    'predicted': result['gb_predicted'],
                    'predicted_name': result['gb_predicted_name'],
                    'confidence': result['gb_confidence'],
                    'correct': bool(result['gb_correct'])
                },
                'CNN': {
                    'predicted': result['cnn_predicted'],
                    'predicted_name': result['cnn_predicted_name'],
                    'confidence': result['cnn_confidence'],
                    'correct': bool(result['cnn_correct'])
                }
            },
            'ensemble': {
                'predicted': result.get('ensemble_predicted', 'N/A'),
                'predicted_name': result.get('ensemble_predicted_name', 'N/A'),
                'correct': bool(result.get('ensemble_correct', False))
            },
            'model_agreement': result.get('model_agreement', 4),
            'models_agree': bool(result.get('models_agree', False))
        }

class AudioVisualizer:
    """Handles all visualization tasks"""

    @staticmethod
    def plot_mfcc_features(audio_path: str) -> plt.Figure:
        """Plot MFCC features visualization - FIXED VERSION"""
        try:
            y, sr = librosa.load(audio_path, sr=None, mono=True)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

            fig, ax = plt.subplots(figsize=(14, 8))

            # Use imshow instead of librosa.display.specshow to fix colorbar issue
            im = ax.imshow(mfcc, 
                          aspect='auto', 
                          origin='lower',
                          cmap='viridis',
                          interpolation='nearest')

            # Now we can add colorbar safely since im is a mappable object
            cbar = plt.colorbar(im, ax=ax, format='%+2.0f dB')
            cbar.set_label('MFCC Magnitude (dB)', rotation=270, labelpad=15)

            ax.set_title('MFCC Features', fontsize=16, fontweight='bold')
            ax.set_xlabel('Time Frames', fontsize=12)
            ax.set_ylabel('MFCC Coefficients', fontsize=12)
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            return fig
        except Exception as e:
            st.error(f"Error plotting MFCC features: {e}")
            return None

    @staticmethod
    def plot_waveform(audio_path: str) -> plt.Figure:
        """Plot audio waveform"""
        try:
            y, sr = librosa.load(audio_path, sr=None, mono=True)

            fig, ax = plt.subplots(figsize=(14, 4))
            time = np.linspace(0, len(y)/sr, len(y))
            ax.plot(time, y, alpha=0.7, color='blue')
            ax.set_xlabel('Time (s)', fontsize=12)
            ax.set_ylabel('Amplitude', fontsize=12)
            ax.set_title('Audio Waveform', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            return fig
        except Exception as e:
            st.error(f"Error plotting waveform: {e}")
            return None

    @staticmethod
    def plot_model_predictions(results: Dict[str, Any]) -> plt.Figure:
        """Plot all model predictions as comparison chart"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        axes = [ax1, ax2, ax3, ax4]
        models = ['SVM', 'Random Forest', 'Gradient Boosting', 'CNN']

        for i, (model, ax) in enumerate(zip(models, axes)):
            model_data = results['models'][model]

            # Create a simple bar showing prediction vs actual
            categories = ['Predicted', 'Actual']
            pred_class = model_data['predicted']
            actual_class = results['actual_class']

            bars = ax.bar(categories, [pred_class, actual_class], 
                         color=['orange' if model_data['correct'] else 'red', 'green'],
                         alpha=0.7)

            ax.set_title(f"{model} - {'Correct' if model_data['correct'] else 'Wrong'}", 
                        fontweight='bold')
            ax.set_ylabel('Class ID')
            ax.set_ylim(0, 9)

            # Add text labels
            ax.text(0, pred_class + 0.1, f"{model_data['predicted_name']}", 
                   ha='center', fontweight='bold')
            ax.text(1, actual_class + 0.1, f"{results['actual_class_name']}", 
                   ha='center', fontweight='bold')

            # Add confidence score
            ax.text(0.02, 0.98, f'Confidence: {model_data["confidence"]:.1%}', 
                    transform=ax.transAxes, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        plt.suptitle(f'Model Predictions for {results["filename"]}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        return fig

    @staticmethod
    def plot_confidence_comparison(results: Dict[str, Any]) -> plt.Figure:
        """Plot confidence scores for all models"""
        fig, ax = plt.subplots(figsize=(12, 6))

        models = list(results['models'].keys())
        confidences = [results['models'][model]['confidence'] for model in models]
        correctness = [results['models'][model]['correct'] for model in models]

        colors = ['green' if correct else 'red' for correct in correctness]
        bars = ax.bar(models, confidences, color=colors, alpha=0.7)

        ax.set_xlabel('Models', fontsize=12, fontweight='bold')
        ax.set_ylabel('Confidence Score', fontsize=12, fontweight='bold')
        ax.set_title('Model Confidence Comparison', fontsize=16, fontweight='bold')
        ax.set_ylim(0, 1)

        # Add value labels on bars
        for bar, conf in zip(bars, confidences):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{conf:.1%}',
                    ha='center', va='bottom', fontweight='bold')

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', alpha=0.7, label='Correct'),
                          Patch(facecolor='red', alpha=0.7, label='Incorrect')]
        ax.legend(handles=legend_elements, loc='upper right')

        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        return fig

def main():
    """Main Streamlit application"""

    # Initialize data loader
    if 'data_loader' not in st.session_state:
        st.session_state.data_loader = AudioDataLoader()

    data_loader = st.session_state.data_loader

    # Header
    st.title("🎵 UrbanSound8K Audio Classification Results")
    st.markdown("### *Pre-tested Results from Cross-Validation Analysis*")
    st.markdown("---")

    # Description
    st.markdown("""
    This application displays the results from our comprehensive testing of UrbanSound8K audio classification 
    using four different machine learning models. Select an audio file from the dropdown to see how each 
    model performed on that specific sample.

    **Models Tested:**
    - 🎯 **SVM** (Support Vector Machine) - Traditional ML approach
    - 🌲 **Random Forest** - Ensemble method with decision trees  
    - 🚀 **Gradient Boosting** - Advanced boosting technique
    - 🧠 **CNN** (Convolutional Neural Network) - Deep learning approach
    """)

    # Sidebar
    st.sidebar.title("Dataset Information")

    if data_loader.results_df is not None:
        # Overall statistics
        st.sidebar.subheader("Overall Accuracies")
        total_samples = len(data_loader.results_df)

        accuracies = {
            'SVM': data_loader.results_df['svm_correct'].mean(),
            'Random Forest': data_loader.results_df['rf_correct'].mean(),
            'Gradient Boosting': data_loader.results_df['gb_correct'].mean(),
            'CNN': data_loader.results_df['cnn_correct'].mean()
        }

        for model, acc in accuracies.items():
            st.sidebar.write(f"**{model}:** {acc:.1%}")

        if 'ensemble_correct' in data_loader.results_df.columns:
            ens_acc = data_loader.results_df['ensemble_correct'].mean()
            st.sidebar.write(f"**Ensemble:** {ens_acc:.1%}")

        st.sidebar.write(f"**Total Samples:** {total_samples:,}")

        st.sidebar.markdown("---")

        # Class distribution
        st.sidebar.subheader("Class Distribution")
        class_counts = data_loader.results_df['actual_class_name'].value_counts()
        for class_name, count in class_counts.items():
            emoji = ['❄️', '🚗', '👶', '🐕', '🔧', '🚙', '🔫', '🔨', '🚨', '🎵'][
                list(CLASS_NAMES.values()).index(class_name)
            ]
            st.sidebar.write(f"{emoji} **{class_name.replace('_', ' ').title()}:** {count}")

    # Main interface
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("🎵 Select Audio File")

        if not data_loader.audio_files:
            st.error("No audio files found in Data folder")
            st.info("Expected folder structure: Data/{audio_files}")
            return

        # Dropdown for file selection
        selected_file = st.selectbox(
            "Choose an audio file to analyze:",
            options=[""] + data_loader.audio_files,
            index=0,
            help="Select from pre-tested audio files in the dataset"
        )

        if selected_file:
            # Get results for selected file
            results = data_loader.get_file_results(selected_file)

            if results is None:
                st.error(f"No results found for {selected_file}")
                st.info("Make sure the file was included in the cross-validation testing.")
                return

            # Display file info
            st.success(f"File selected: {selected_file}")

            # Audio player (try to find and play the file)
            audio_path = None
            possible_paths = [
                os.path.join(data_loader.data_folder, selected_file),
                os.path.join(data_loader.data_folder, f"fold{results['fold']}", selected_file),
            ]

            for path in possible_paths:
                if os.path.exists(path):
                    audio_path = path
                    break

            if audio_path:
                st.subheader("🎧 Audio Player")
                st.audio(audio_path)
            else:
                st.warning("⚠️ Audio file not found for playback")

            # File details
            st.subheader("📋 File Details")
            st.write(f"**Filename:** {results['filename']}")
            st.write(f"**Actual Class:** {results['actual_class_name'].replace('_', ' ').title()}")
            st.write(f"**Class ID:** {results['actual_class']}")
            st.write(f"**Test Fold:** {results['fold']}")

            # Processing options
            st.subheader("⚙️ Display Options")
            show_waveform = st.checkbox("Show Waveform", value=True)
            show_features = st.checkbox("Show MFCC Features", value=True)
            show_confidence = st.checkbox("Show Confidence Comparison", value=True)

    with col2:
        if selected_file and 'results' in locals():
            st.subheader("🔍 Model Prediction Results")

            # Results summary table
            st.subheader("📊 Results Summary")
            summary_data = []

            for model_name, model_data in results['models'].items():
                summary_data.append({
                    'Model': model_name,
                    'Predicted': f"{model_data['predicted']} ({model_data['predicted_name'].replace('_', ' ').title()})",
                    'Actual': f"{results['actual_class']} ({results['actual_class_name'].replace('_', ' ').title()})",
                    'Confidence': f"{model_data['confidence']:.1%}",
                    'Result': "Correct" if model_data['correct'] else "Wrong"
                })

            # Add ensemble if available
            if results['ensemble']['predicted'] != 'N/A':
                summary_data.append({
                    'Model': 'Ensemble',
                    'Predicted': f"{results['ensemble']['predicted']} ({results['ensemble']['predicted_name'].replace('_', ' ').title()})",
                    'Actual': f"{results['actual_class']} ({results['actual_class_name'].replace('_', ' ').title()})",
                    'Confidence': "N/A",
                    'Result': "Correct" if results['ensemble']['correct'] else "Wrong"
                })

            summary_df = pd.DataFrame(summary_data)
            st.table(summary_df)

            # Model agreement info
            st.subheader("🤝 Model Agreement Analysis")
            agreement_text = {
                1: "🟢 All models agree",
                2: "🟡 Models split into 2 groups", 
                3: "🟠 3 different predictions",
                4: "🔴 All models disagree"
            }

            agreement_level = results['model_agreement']
            st.info(f"**Agreement Level:** {agreement_text.get(agreement_level, 'Unknown')}")

            # Visualizations
            if audio_path:
                if show_waveform:
                    st.subheader("📊 Audio Waveform")
                    fig_wave = AudioVisualizer.plot_waveform(audio_path)
                    if fig_wave:
                        st.pyplot(fig_wave)
                        plt.close(fig_wave)

                if show_features:
                    st.subheader("📈 MFCC Features")
                    fig_mfcc = AudioVisualizer.plot_mfcc_features(audio_path)
                    if fig_mfcc:
                        st.pyplot(fig_mfcc)
                        plt.close(fig_mfcc)

            # Model predictions visualization
            st.subheader("🎯 Model Predictions Comparison")
            fig_pred = AudioVisualizer.plot_model_predictions(results)
            st.pyplot(fig_pred)
            plt.close(fig_pred)

            if show_confidence:
                st.subheader("📊 Model Confidence Comparison")
                fig_conf = AudioVisualizer.plot_confidence_comparison(results)
                st.pyplot(fig_conf)
                plt.close(fig_conf)

            # Detailed analysis in tabs
            st.subheader("🔍 Detailed Model Analysis")
            tabs = st.tabs(list(results['models'].keys()))

            for i, (model_name, tab) in enumerate(zip(results['models'].keys(), tabs)):
                with tab:
                    model_data = results['models'][model_name]

                    # Key metrics
                    col_a, col_b, col_c, col_d = st.columns([1, 1, 1, 1])

                    with col_a:
                        st.metric("Predicted Class", model_data['predicted'])
                    with col_b:
                        st.metric("Predicted Name", model_data['predicted_name'].replace('_', ' ').title())
                    with col_c:
                        st.metric("Confidence", f"{model_data['confidence']:.2%}")
                    with col_d:
                        result_color = "normal" if model_data['correct'] else "inverse"
                        st.metric("Result", "Correct" if model_data['correct'] else "Wrong")

                    # Analysis
                    if model_data['correct']:
                        st.success(f"{model_name} correctly identified this as {model_data['predicted_name'].replace('_', ' ').title()}")
                    else:
                        st.error(f"{model_name} incorrectly predicted {model_data['predicted_name'].replace('_', ' ').title()} when it was actually {results['actual_class_name'].replace('_', ' ').title()}")

                    st.write(f"**Confidence Level:** {model_data['confidence']:.1%}")

                    # Confidence interpretation
                    if model_data['confidence'] > 0.8:
                        st.info("High confidence prediction")
                    elif model_data['confidence'] > 0.6:
                        st.info("Moderate confidence prediction")
                    else:
                        st.warning("Low confidence prediction")

    # Technical details
    with st.expander("🔧 About This Analysis"):
        st.markdown("""
        ### Data Source
        - **Dataset:** UrbanSound8K (8,732 audio samples)
        - **Evaluation:** 10-fold cross-validation
        - **Results File:** all_models_predictions_comparison.csv

        ### Model Performance
        - **SVM:** Traditional machine learning with RBF kernel
        - **Random Forest:** 200-tree ensemble with optimized parameters
        - **Gradient Boosting:** Histogram-based gradient boosting
        - **CNN:** Deep learning with convolutional layers

        ### Features Used
        - **Traditional Models:** 26 features (13 MFCC + 13 Delta MFCC coefficients)
        - **CNN Model:** 40 MFCC coefficients × 174 time frames

        ### Results Interpretation
        - **Green bars:** Correct predictions
        - **Red bars:** Incorrect predictions  
        - **Confidence scores:** Model certainty (0-100%)
        - **Agreement analysis:** How many models agreed on the prediction
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 14px;'>
    🎵 <strong>UrbanSound8K Classification Results</strong> | Pre-tested Cross-Validation Data<br>
    Select different audio files to explore model performance across the dataset
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
