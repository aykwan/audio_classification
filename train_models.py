
"""
UrbanSound8K Model Training Script
Based on the Practicum implementation with SVM, Random Forest, Gradient Boosting, and CNN models
"""

import pandas as pd
import numpy as np
import librosa
import time
import pickle
import os
from pathlib import Path

# ML Models
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from keras_tuner import RandomSearch

import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
random_seed = 42
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

class UrbanSound8KTrainer:
    """Complete training pipeline for UrbanSound8K dataset"""

    def __init__(self, data_path="Data", metadata_file="UrbanSound8K.csv"):
        """
        Initialize trainer with dataset path

        Args:
            data_path: Path to folder containing fold1/, fold2/, etc.
            metadata_file: Path to UrbanSound8K.csv metadata file
        """
        self.data_path = data_path
        self.metadata_file = os.path.join(data_path, metadata_file)

        # Create models directory
        self.models_dir = "models"
        os.makedirs(self.models_dir, exist_ok=True)

        # Load metadata
        self.meta = pd.read_csv(self.metadata_file)
        print(f"✅ Loaded metadata: {len(self.meta)} samples")

        # Class names
        self.class_names = {
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

    def extract_features_and_labels(self, metadata, base_dir=None, feature_type="traditional"):
        """
        Extract features from audio files

        Args:
            metadata: DataFrame with file information
            base_dir: Base directory path (uses self.data_path if None)
            feature_type: "traditional" for SVM/RF/GB, "cnn" for CNN
        """
        if base_dir is None:
            base_dir = self.data_path

        features = []
        labels = []

        print(f"Extracting {feature_type} features from {len(metadata)} files...")

        for idx, row in metadata.iterrows():
            if idx % 100 == 0:
                print(f"Processing file {idx+1}/{len(metadata)}")

            # Get file path
            fold = row['fold']
            file_name = row['slice_file_name']
            file_path = f"{base_dir}/fold{fold}/{file_name}"

            # Load audio
            try:
                y, sr = librosa.load(file_path, sr=None, mono=True)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue

            if feature_type == "traditional":
                # Extract traditional features (MFCC + Delta MFCC)
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

                # Combine and average
                combined = np.concatenate([mfcc, mfcc_delta], axis=0)
                combined_mean = np.mean(combined, axis=1)

                features.append(combined_mean)

            elif feature_type == "cnn":
                # Extract CNN features (40 MFCC with fixed length)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

                max_len = 174
                if mfcc.shape[1] < max_len:
                    pad_width = max_len - mfcc.shape[1]
                    mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
                else:
                    mfcc = mfcc[:, :max_len]

                features.append(mfcc)

            labels.append(row['classID'])

        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(labels)

        if feature_type == "cnn":
            X = X[..., np.newaxis]  # Add channel dimension

        print(f"✅ Extracted features shape: {X.shape}, labels: {y.shape}")
        return X, y

    def train_svm(self, use_cv=True):
        """Train SVM model with cross-validation"""
        print("\n🎯 Training SVM Model")
        print("=" * 50)

        param_grid = {
            'C': [1, 10, 100],
            'gamma': ['scale', 0.1]
        }

        if use_cv:
            # 10-fold cross-validation
            accuracy_per_fold = []

            for test_fold in range(1, 11):
                print(f"\nProcessing fold {test_fold}...")

                train_meta = self.meta[self.meta['fold'] != test_fold]
                test_meta = self.meta[self.meta['fold'] == test_fold]

                X_train, y_train = self.extract_features_and_labels(train_meta, feature_type="traditional")
                X_test, y_test = self.extract_features_and_labels(test_meta, feature_type="traditional")

                svm = SVC(random_state=random_seed)
                grid_search = RandomizedSearchCV(
                    svm, param_grid, n_iter=6, cv=2,
                    scoring='accuracy', verbose=1, n_jobs=-1, 
                    random_state=random_seed
                )

                grid_search.fit(X_train, y_train)
                accuracy = grid_search.score(X_test, y_test)
                accuracy_per_fold.append(accuracy)

                print(f"Fold {test_fold} accuracy: {accuracy:.4f}")

            avg_accuracy = np.mean(accuracy_per_fold)
            print(f"\n📊 Average SVM accuracy: {avg_accuracy:.4f}")

            # Train final model on all data with best params
            best_params = {'C': 100, 'gamma': 'scale'}  # Most common from practicum

        else:
            # Train on 90% of data, test on 10%
            X, y = self.extract_features_and_labels(self.meta, feature_type="traditional")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.1, random_state=random_seed, stratify=y
            )
            best_params = {'C': 100, 'gamma': 'scale'}

        # Train final model
        final_svm = SVC(probability=True, random_state=random_seed, **best_params)
        if not use_cv:
            final_svm.fit(X_train, y_train)
        else:
            X_all, y_all = self.extract_features_and_labels(self.meta, feature_type="traditional")
            final_svm.fit(X_all, y_all)

        # Save model
        model_path = os.path.join(self.models_dir, 'svm_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(final_svm, f)

        print(f"✅ SVM model saved to {model_path}")
        return final_svm

    def train_random_forest(self, use_cv=True):
        """Train Random Forest model"""
        print("\n🌲 Training Random Forest Model") 
        print("=" * 50)

        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }

        if use_cv:
            # 10-fold cross-validation
            accuracy_per_fold = []

            for test_fold in range(1, 11):
                print(f"\nProcessing fold {test_fold}...")

                train_meta = self.meta[self.meta['fold'] != test_fold]
                test_meta = self.meta[self.meta['fold'] == test_fold]

                X_train, y_train = self.extract_features_and_labels(train_meta, feature_type="traditional")
                X_test, y_test = self.extract_features_and_labels(test_meta, feature_type="traditional")

                rf = RandomForestClassifier(random_state=random_seed)
                search = RandomizedSearchCV(
                    rf, param_grid, n_iter=10, cv=3,
                    scoring='accuracy', n_jobs=-1, verbose=1,
                    random_state=random_seed
                )

                search.fit(X_train, y_train)
                accuracy = search.score(X_test, y_test)
                accuracy_per_fold.append(accuracy)

                print(f"Fold {test_fold} accuracy: {accuracy:.4f}")

            avg_accuracy = np.mean(accuracy_per_fold)
            print(f"\n📊 Average Random Forest accuracy: {avg_accuracy:.4f}")

            best_params = {'n_estimators': 200, 'max_depth': None, 'min_samples_split': 10}

        else:
            X, y = self.extract_features_and_labels(self.meta, feature_type="traditional")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.1, random_state=random_seed, stratify=y
            )
            best_params = {'n_estimators': 200, 'max_depth': None, 'min_samples_split': 10}

        # Train final model
        final_rf = RandomForestClassifier(random_state=random_seed, **best_params)
        if not use_cv:
            final_rf.fit(X_train, y_train)
        else:
            X_all, y_all = self.extract_features_and_labels(self.meta, feature_type="traditional")
            final_rf.fit(X_all, y_all)

        # Save model
        model_path = os.path.join(self.models_dir, 'rf_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(final_rf, f)

        print(f"✅ Random Forest model saved to {model_path}")
        return final_rf

    def train_gradient_boosting(self, use_cv=True):
        """Train Gradient Boosting model"""
        print("\n🚀 Training Gradient Boosting Model")
        print("=" * 50)

        param_grid = {
            'max_iter': [100, 200, 300],
            'max_depth': [None, 5, 10, 20],
            'min_samples_leaf': [1, 5, 10, 20],
            'learning_rate': [0.01, 0.1, 0.2],
            'l2_regularization': [0, 0.1, 1]
        }

        if use_cv:
            # 10-fold cross-validation
            accuracy_per_fold = []

            for test_fold in range(1, 11):
                print(f"\nProcessing fold {test_fold}...")

                train_meta = self.meta[self.meta['fold'] != test_fold]
                test_meta = self.meta[self.meta['fold'] == test_fold]

                X_train, y_train = self.extract_features_and_labels(train_meta, feature_type="traditional")
                X_test, y_test = self.extract_features_and_labels(test_meta, feature_type="traditional")

                gb = HistGradientBoostingClassifier(random_state=random_seed)
                search = RandomizedSearchCV(
                    gb, param_grid, n_iter=10, cv=3,
                    scoring='accuracy', n_jobs=-1, verbose=1,
                    random_state=random_seed
                )

                search.fit(X_train, y_train)
                accuracy = search.score(X_test, y_test)
                accuracy_per_fold.append(accuracy)

                print(f"Fold {test_fold} accuracy: {accuracy:.4f}")

            avg_accuracy = np.mean(accuracy_per_fold)
            print(f"\n📊 Average Gradient Boosting accuracy: {avg_accuracy:.4f}")

            best_params = {
                'max_iter': 300, 'max_depth': None, 
                'learning_rate': 0.2, 'l2_regularization': 1,
                'min_samples_leaf': 1
            }

        else:
            X, y = self.extract_features_and_labels(self.meta, feature_type="traditional")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.1, random_state=random_seed, stratify=y
            )
            best_params = {
                'max_iter': 300, 'max_depth': None, 
                'learning_rate': 0.2, 'l2_regularization': 1,
                'min_samples_leaf': 1
            }

        # Train final model
        final_gb = HistGradientBoostingClassifier(random_state=random_seed, **best_params)
        if not use_cv:
            final_gb.fit(X_train, y_train)
        else:
            X_all, y_all = self.extract_features_and_labels(self.meta, feature_type="traditional")
            final_gb.fit(X_all, y_all)

        # Save model
        model_path = os.path.join(self.models_dir, 'gb_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(final_gb, f)

        print(f"✅ Gradient Boosting model saved to {model_path}")
        return final_gb

    def build_cnn_model(self):
        """Build CNN model architecture"""
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

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train_cnn(self, use_cv=True, epochs=20):
        """Train CNN model"""
        print("\n🧠 Training CNN Model")
        print("=" * 50)

        if use_cv:
            # 10-fold cross-validation
            accuracy_per_fold = []

            for test_fold in range(1, 11):
                print(f"\nProcessing fold {test_fold}...")

                train_meta = self.meta[self.meta['fold'] != test_fold]
                test_meta = self.meta[self.meta['fold'] == test_fold]

                X_train, y_train = self.extract_features_and_labels(train_meta, feature_type="cnn")
                X_test, y_test = self.extract_features_and_labels(test_meta, feature_type="cnn")

                model = self.build_cnn_model()

                # Train model
                model.fit(
                    X_train, y_train,
                    epochs=epochs,
                    batch_size=32,
                    verbose=2,
                    validation_split=0.1
                )

                # Evaluate
                test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
                accuracy_per_fold.append(test_acc)

                print(f"Fold {test_fold} accuracy: {test_acc:.4f}")

            avg_accuracy = np.mean(accuracy_per_fold)
            print(f"\n📊 Average CNN accuracy: {avg_accuracy:.4f}")

            # Train final model on all data
            X_all, y_all = self.extract_features_and_labels(self.meta, feature_type="cnn")
            final_model = self.build_cnn_model()
            final_model.fit(X_all, y_all, epochs=epochs, batch_size=32, verbose=2)

        else:
            # Train on 90% of data
            X, y = self.extract_features_and_labels(self.meta, feature_type="cnn")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.1, random_state=random_seed, stratify=y
            )

            final_model = self.build_cnn_model()
            final_model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=32,
                verbose=2,
                validation_data=(X_test, y_test)
            )

        # Save model
        model_path = os.path.join(self.models_dir, 'cnn_model.h5')
        final_model.save(model_path)

        print(f"✅ CNN model saved to {model_path}")
        return final_model

    def train_all_models(self, use_cv=True):
        """Train all models in sequence"""
        print("🎵 Starting UrbanSound8K Model Training Pipeline")
        print("=" * 60)

        start_time = time.time()

        # Train all models
        print("Training traditional ML models...")
        svm_model = self.train_svm(use_cv=use_cv)
        rf_model = self.train_random_forest(use_cv=use_cv)
        gb_model = self.train_gradient_boosting(use_cv=use_cv)

        print("\nTraining deep learning model...")
        cnn_model = self.train_cnn(use_cv=use_cv)

        total_time = time.time() - start_time
        print(f"\n🎉 All models trained successfully!")
        print(f"⏱️  Total training time: {total_time:.2f} seconds")
        print(f"📁 Models saved to: {self.models_dir}/")

        return {
            'svm': svm_model,
            'rf': rf_model,
            'gb': gb_model,
            'cnn': cnn_model
        }

def main():
    """Main training script"""
    import argparse

    parser = argparse.ArgumentParser(description='Train UrbanSound8K classification models')
    parser.add_argument('--data_path', type=str, default='Data',
                        help='Path to UrbanSound8K dataset folder')
    parser.add_argument('--no_cv', action='store_true',
                        help='Skip cross-validation (faster training)')
    parser.add_argument('--models', nargs='+', 
                        choices=['svm', 'rf', 'gb', 'cnn', 'all'],
                        default=['all'],
                        help='Which models to train')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs for CNN training')

    args = parser.parse_args()

    # Initialize trainer
    trainer = UrbanSound8KTrainer(data_path=args.data_path)

    use_cv = not args.no_cv

    # Train selected models
    if 'all' in args.models:
        trainer.train_all_models(use_cv=use_cv)
    else:
        if 'svm' in args.models:
            trainer.train_svm(use_cv=use_cv)
        if 'rf' in args.models:
            trainer.train_random_forest(use_cv=use_cv)
        if 'gb' in args.models:
            trainer.train_gradient_boosting(use_cv=use_cv)
        if 'cnn' in args.models:
            trainer.train_cnn(use_cv=use_cv, epochs=args.epochs)

if __name__ == "__main__":
    main()
