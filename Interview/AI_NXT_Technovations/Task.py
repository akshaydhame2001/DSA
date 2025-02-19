# Objective :
# You are required to develop an intelligent signature verification system that can differentiate between valid human signatures and invalid hand-drawn inputs. The invalid inputs may include random scribbles, geometric shapes, arbitrary strokes, symbols, numbers, or any non-signature marks.

# Your solution should generalize well beyond simple classification and ensure that only proper signatures are accepted while rejecting any non-signature-like input.

# Key Challenge:
# Unlike traditional classification problems (where each class has well-defined patterns), this problem requires handling unpredictable improper inputs that can take any shape.

# Instructions & Guidelines
# 📌 Dataset Specifications
# Proper Signatures: A set of genuine human signatures with natural variations in writing styles, pressure, slant, and stroke thickness.
# Improper Signatures (Invalid Inputs): A diverse set of random scribbles, geometric shapes, fake strokes, numbers, doodles, and non-Latin characters etc., that users may enter in place of a valid signature.

# Synthetic Data (You are encouraged to generate synthetic fake signatures using noise augmentation or random stroke generation)

# 📌 Task Breakdown
# Develop a solution that Differentiates Between Valid and Invalid Signatures

# Your system should classify whether the given input is a proper signature or an invalid input.

# Your solution should be able to reject a wide range of non-signature inputs, even those not seen during training.
# It should not memorize dataset-specific patterns but instead focus on generalized approach


# Clearly explain your approach, architecture, and reasoning behind the chosen method.
# Provide a detailed error analysis and explain how your solution handles complex cases.

# 💻 Code Implementation

# Submit a well-structured codebase with clear documentation and reproducible results.
# Include preprocessing scripts for handling signature datasets and generating synthetic scribbles.

# 📊 Performance Metrics & Visualizations

# Provide evaluation metrics (F1-score, precision-recall curve, anomaly detection thresholds, etc.).
# Demonstrate false positives and false negatives and explain how to improve the solution.

# Important Note
# Deadline to submit this task is 19/02/2025
# Attaching a few samples of valid and invalid signatures for your reference.



import tensorflow as tf
from tensorflow.keras import layers, Model, metrics, regularizers
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.image import ssim
import json
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
import seaborn as sns
from datetime import datetime
from tensorflow.keras.utils import register_keras_serializable


def create_save_directory():
    """Create a timestamped directory for saving model artifacts"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_dir = f'signature_model_{timestamp}'
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

def preprocess_signature(image, target_size=(224, 224)):
    """
    Preprocess signature image for model input.

    Parameters:
        image: Can be either a path string or numpy array
        target_size: Tuple of (height, width) for output size

    Returns:
        Preprocessed image as numpy array or None if processing fails
    """
    try:
        # Load image if path is provided
        if isinstance(image, str):
            img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Failed to load image from path: {image}")
                return None
        else:
            # If input is an array, handle both RGB and grayscale cases
            img = image.copy()
            if len(img.shape) == 3:
                if img.shape[2] == 3:  # RGB image
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                elif img.shape[2] == 1:  # Grayscale with extra dimension
                    img = img[:, :, 0]
                else:
                    raise ValueError(f"Unexpected number of channels: {img.shape[2]}")

        # Ensure image is 2D
        if len(img.shape) != 2:
            raise ValueError(f"Expected 2D image after grayscale conversion, got shape {img.shape}")

        # Convert to 8-bit before equalizing histogram
        img = img.astype(np.uint8)
        img = cv2.equalizeHist(img)

        # Remove noise with Gaussian blur
        img = cv2.GaussianBlur(img, (3, 3), 0)

        # Apply adaptive thresholding
        img = cv2.adaptiveThreshold(
            img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )

        # Remove small noise components
        kernel = np.ones((2, 2), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

        # Resize to target size
        if img.shape[:2] != target_size:
            img = cv2.resize(img, target_size)

        # Normalize pixel values
        img = img.astype(np.float32) / 255.0

        # Add channel dimension for model input
        img = np.expand_dims(img, axis=-1)

        # Validate final shape
        if img.shape != (*target_size, 1):
            raise ValueError(f"Unexpected final shape: {img.shape}, expected {(*target_size, 1)}")

        return img

    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

class SignatureVerificationSystem:
    def __init__(self, input_shape=(224, 224, 1), latent_dim=128):
        self.input_shape = input_shape
        self.latent_dim = latent_dim
        # Calculate required shapes for proper reconstruction
        self.conv_shape = (input_shape[0] // 16, input_shape[1] // 16, 256)  # After 4 max pooling layers
        self.encoder, self.decoder, self.autoencoder = self._build_autoencoder()
        self.classifier = self._build_classifier()
        self.anomaly_detector = None
        self.reconstruction_threshold = None
        self.metrics_history = {}
        self.save_dir = create_save_directory()

    def _build_autoencoder(self):
        # Encoder
        encoder_input = layers.Input(shape=self.input_shape)

        # Encoder blocks with shape tracking
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(1e-4))(encoder_input)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)  # 112x112x32
        x = layers.Dropout(0.2)(x)

        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)  # 56x56x64
        x = layers.Dropout(0.2)(x)

        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)  # 28x28x128
        x = layers.Dropout(0.2)(x)

        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                         kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)  # 14x14x256

        # Flatten and encode to latent space
        x = layers.Flatten()(x)
        encoded = layers.Dense(self.latent_dim, activation='relu',
                             kernel_regularizer=regularizers.l2(1e-4))(x)

        # Decoder
        decoder_input = layers.Input(shape=(self.latent_dim,))

        # Calculate the shape for reshaping
        units = self.conv_shape[0] * self.conv_shape[1] * self.conv_shape[2]

        x = layers.Dense(units, activation='relu')(decoder_input)
        x = layers.Reshape(self.conv_shape)(x)  # Reshape to match the encoder's last conv shape

        x = layers.Conv2DTranspose(128, (3, 3), strides=2, activation='relu', padding='same',
                                 kernel_regularizer=regularizers.l2(1e-4))(x)  # 28x28x128
        x = layers.BatchNormalization()(x)

        x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same',
                                 kernel_regularizer=regularizers.l2(1e-4))(x)  # 56x56x64
        x = layers.BatchNormalization()(x)

        x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same',
                                 kernel_regularizer=regularizers.l2(1e-4))(x)  # 112x112x32
        x = layers.BatchNormalization()(x)

        x = layers.Conv2DTranspose(1, (3, 3), strides=2, activation='sigmoid', padding='same')(x)  # 224x224x1

        # Create models
        encoder = Model(encoder_input, encoded, name='encoder')
        decoder = Model(decoder_input, x, name='decoder')
        autoencoder = Model(encoder_input, decoder(encoder(encoder_input)), name='autoencoder')

        # Custom loss combining reconstruction and SSIM
        @register_keras_serializable()
        def combined_loss(y_true, y_pred):
            reconstruction_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
            ssim_loss = 1 - tf.reduce_mean(ssim(y_true, y_pred, max_val=1.0))
            return 0.7 * reconstruction_loss + 0.3 * ssim_loss

        autoencoder.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            loss=combined_loss,
            metrics=['accuracy', metrics.Precision(), metrics.Recall(), metrics.AUC()]
        )

        return encoder, decoder, autoencoder

    def _build_classifier(self):
        """Build secondary CNN classifier for valid/invalid signatures"""
        model = tf.keras.Sequential([
            layers.Input(shape=self.input_shape),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])

        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def prepare_data(self, valid_dir, invalid_dir):
        """Prepare and augment training data with progress tracking"""
        print("Preparing training data...")

        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=False,
            fill_mode='nearest',
            validation_split=0.2
        )

        # Load and preprocess valid signatures
        valid_signatures = []
        total_valid = len(os.listdir(valid_dir))
        print(f"Processing {total_valid} valid signatures...")

        for i, img_path in enumerate(os.listdir(valid_dir)):
            if i % 100 == 0:
                print(f"Processed {i}/{total_valid} valid signatures")
            img = preprocess_signature(os.path.join(valid_dir, img_path))
            if img is not None:
                valid_signatures.append(img)

        # Load and preprocess invalid signatures
        invalid_signatures = []
        total_invalid = len(os.listdir(invalid_dir))
        print(f"Processing {total_invalid} invalid signatures...")

        for i, img_path in enumerate(os.listdir(invalid_dir)):
            if i % 100 == 0:
                print(f"Processed {i}/{total_invalid} invalid signatures")
            img = preprocess_signature(os.path.join(invalid_dir, img_path))
            if img is not None:
                invalid_signatures.append(img)

        X_valid = np.array(valid_signatures)
        X_invalid = np.array(invalid_signatures)

        print(f"Final dataset sizes - Valid: {len(X_valid)}, Invalid: {len(X_invalid)}")
        return X_valid, X_invalid, datagen

    def train(self, valid_dir, invalid_dir, epochs=30, batch_size=32):
        """Enhanced training function with comprehensive logging"""
        print("Starting training process...")
        X_valid, X_invalid, datagen = self.prepare_data(valid_dir, invalid_dir)

        # Train autoencoder
        print("Training autoencoder...")
        history_auto = self.autoencoder.fit(
            datagen.flow(X_valid, X_valid, batch_size=batch_size),
            epochs=2,
            validation_data=datagen.flow(
                X_valid, X_valid, batch_size=batch_size, subset='validation'),
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6),
                ModelCheckpoint(
                    os.path.join(self.save_dir, 'best_autoencoder.keras'),
                    monitor='val_loss',
                    save_best_only=True
                )
            ]
        )

        # Train anomaly detector
        print("Training anomaly detector...")
        latent_valid = self.encoder.predict(X_valid)
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.anomaly_detector.fit(latent_valid)

        # Prepare classifier data
        print("Training classifier...")
        X_combined = np.concatenate([X_valid, X_invalid])
        y_combined = np.concatenate([np.ones(len(X_valid)), np.zeros(len(X_invalid))])

        # Train classifier
        history_class = self.classifier.fit(
            X_combined, y_combined,
            batch_size=batch_size,
            epochs=3,
            validation_split=0.2,
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
                ModelCheckpoint(
                    os.path.join(self.save_dir, 'best_classifier.keras'),
                    monitor='val_loss',
                    save_best_only=True
                )
            ]
        )

        # Store metrics
        self.metrics_history = {
            'autoencoder': history_auto.history,
            'classifier': history_class.history
        }

        # Calculate thresholds
        print("Calculating decision thresholds...")
        reconstructions = self.autoencoder.predict(X_valid)
        reconstruction_errors = np.mean(np.square(X_valid - reconstructions), axis=(1,2,3))
        self.reconstruction_threshold = np.mean(reconstruction_errors) + 2 * np.std(reconstruction_errors)

        # Save everything
        self.save_model()

        # Plot and save metrics
        self.plot_training_metrics()
        self.evaluate_system(X_valid, X_invalid)

        return self.metrics_history

    def plot_training_metrics(self):
        """Enhanced plotting function with additional metrics"""
        print("Plotting training metrics...")

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(3, 2)

        # Autoencoder loss
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(self.metrics_history['autoencoder']['loss'], label='Training Loss')
        ax1.plot(self.metrics_history['autoencoder']['val_loss'], label='Validation Loss')
        ax1.set_title('Autoencoder Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()

        # Classifier accuracy
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(self.metrics_history['classifier']['accuracy'], label='Training Accuracy')
        ax2.plot(self.metrics_history['classifier']['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Classifier Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()

        # Autoencoder metrics
        ax3 = fig.add_subplot(gs[1, 0])
        metrics_to_plot = ['precision_15', 'recall_15', 'auc_15']
        for metric in metrics_to_plot:
            ax3.plot(self.metrics_history['autoencoder'][metric], label=f'Training {metric}')
            ax3.plot(self.metrics_history['autoencoder'][f'val_{metric}'], label=f'Validation {metric}')
        ax3.set_title('Autoencoder Metrics')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Score')
        ax3.legend()

        # Save plot
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_metrics.png'))
        plt.close()

    def verify_signature(self, signature_img):
        """Verify a signature using multiple criteria"""
        processed_img = preprocess_signature(signature_img)
        if processed_img is None:
            return False, "Error processing image"

        # Get reconstruction and latent representation
        reconstruction = self.autoencoder.predict(np.array([processed_img]))
        latent_repr = self.encoder.predict(np.array([processed_img]))

        # Calculate reconstruction error
        recon_error = np.mean(np.square(processed_img - reconstruction[0]))

        # Get anomaly score
        anomaly_score = self.anomaly_detector.score_samples([latent_repr[0]])[0]

        # Get classifier prediction
        classifier_pred = self.classifier.predict(np.array([processed_img]))[0][0]

        # Combine criteria for final decision
        is_valid = (
            recon_error <= self.reconstruction_threshold and
            anomaly_score >= -0.5 and
            classifier_pred >= 0.5)


        return is_valid, {
            'reconstruction_error': float(recon_error),
            'anomaly_score': float(anomaly_score),
            'classifier_confidence': float(classifier_pred)
        }

    def evaluate_system(self, X_valid, X_invalid):
        """Comprehensive system evaluation"""
        print("Evaluating system performance...")

        # Prepare evaluation data
        X_combined = np.concatenate([X_valid, X_invalid])
        y_true = np.concatenate([np.ones(len(X_valid)), np.zeros(len(X_invalid))])

        # Get predictions
        y_pred = []
        for img in X_combined:
            is_valid, _ = self.verify_signature(img)
            y_pred.append(1 if is_valid else 0)

        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)

        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(self.save_dir, 'confusion_matrix.png'))
        plt.close()

        # Calculate and save metrics
        precision = cm[1,1] / (cm[1,1] + cm[0,1])
        recall = cm[1,1] / (cm[1,1] + cm[1,0])
        f1_score = 2 * (precision * recall) / (precision + recall)

        evaluation_metrics = {
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1_score),
            'confusion_matrix': cm.tolist()
        }

        with open(os.path.join(self.save_dir, 'evaluation_metrics.json'), 'w') as f:
            json.dump(evaluation_metrics, f, indent=4)

        print("\nEvaluation Metrics:")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1_score:.4f}")

    def save_model(self):
        """Save all model components and parameters"""
        print(f"Saving model components to {self.save_dir}...")

        # Save neural network models
        self.autoencoder.save(os.path.join(self.save_dir, 'autoencoder.keras'))
        self.encoder.save(os.path.join(self.save_dir, 'encoder.keras'))
        self.decoder.save(os.path.join(self.save_dir, 'decoder.keras'))
        self.classifier.save(os.path.join(self.save_dir, 'classifier.keras'))

        # Save anomaly detector
        import joblib
        joblib.dump(self.anomaly_detector, os.path.join(self.save_dir, 'anomaly_detector.joblib'))

        # Save thresholds and parameters
        params = {
            'reconstruction_threshold': float(self.reconstruction_threshold),
            'input_shape': self.input_shape,
            'latent_dim': self.latent_dim
        }

        with open(os.path.join(self.save_dir, 'parameters.json'), 'w') as f:
            json.dump(params, f, indent=4)

        # Save threshold and metrics
        np.save(os.path.join(self.save_dir, 'reconstruction_threshold.npy'),
                self.reconstruction_threshold)

        with open(os.path.join(self.save_dir, 'metrics_history.json'), 'w') as f:
            json.dump(self.metrics_history, f)

        print("Model saved successfully!")

    @classmethod
    def load_model(cls, model_dir):
        """Load a saved model"""
        print(f"Loading model from {model_dir}...")

        # Load parameters
        with open(os.path.join(model_dir, 'parameters.json'), 'r') as f:
            params = json.load(f)

        # Initialize model with saved parameters
        model = cls(input_shape=tuple(params['input_shape']),
                   latent_dim=params['latent_dim'])

        # Load neural network models
        model.autoencoder = tf.keras.models.load_model(os.path.join(model_dir, 'autoencoder.keras'))
        model.encoder = tf.keras.models.load_model(os.path.join(model_dir, 'encoder.keras'))
        model.decoder = tf.keras.models.load_model(os.path.join(model_dir, 'decoder.keras'))
        model.classifier = tf.keras.models.load_model(os.path.join(model_dir, 'classifier.keras'))

        # Load anomaly detector
        import joblib
        model.anomaly_detector = joblib.load(os.path.join(model_dir, 'anomaly_detector.joblib'))

        # Load threshold
        model.reconstruction_threshold = params['reconstruction_threshold']

        print("Model loaded successfully!")
        return model

def main():
    """Main function to demonstrate usage"""
    # Set paths
    valid_dir = "/content/valid_signatures"
    invalid_dir = "/content/invalid_signatures"

    # Initialize and train model
    model = SignatureVerificationSystem()
    model.train(valid_dir, invalid_dir, epochs=30, batch_size=32)

if __name__ == "__main__":
    main()

    # Example of loading a saved model
    # loaded_model = SignatureVerificationSystem.load_model("path/to/saved/model")

    # Example of verifying a signature
    # is_valid, details = model.verify_signature("path/to/test/signature.png")