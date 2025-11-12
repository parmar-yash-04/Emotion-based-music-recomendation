import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
import pickle

class EmotionDetector:
    def __init__(self, img_size=(48, 48)):
        self.img_size = img_size
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.model = None
        
    def create_model(self):
        """Create CNN model for emotion detection"""
        model = models.Sequential([
            # Input layer
            layers.Input(shape=(*self.img_size, 1)),
            
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Convolutional Block - Reduced filters to prevent negative dimensions
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.GlobalAveragePooling2D(),  # Use GlobalAveragePooling instead of MaxPooling
            layers.Dropout(0.25),
            
            # Dense Layers
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(len(self.emotions), activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_simple_model(self):
        """Create a simpler CNN model for smaller images"""
        model = models.Sequential([
            # Input layer
            layers.Input(shape=(*self.img_size, 1)),
            
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Global Average Pooling to avoid dimension issues
            layers.GlobalAveragePooling2D(),
            
            # Dense Layers
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(len(self.emotions), activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def load_and_preprocess_data(self, data_dir):
        """Load and preprocess the emotion dataset"""
        X, y = [], []
        
        for i, emotion in enumerate(self.emotions):
            emotion_dir = os.path.join(data_dir, emotion)
            for img_file in os.listdir(emotion_dir):
                if img_file.endswith('.jpg'):
                    img_path = os.path.join(emotion_dir, img_file)
                    try:
                        # Load and preprocess image
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        img = cv2.resize(img, self.img_size)
                        img = img.astype('float32') / 255.0
                        img = np.expand_dims(img, axis=-1)
                        
                        X.append(img)
                        y.append(i)
                    except Exception as e:
                        print(f"Error loading {img_path}: {e}")
                        continue
        
        X = np.array(X)
        y = tf.keras.utils.to_categorical(y, len(self.emotions))
        
        return X, y
    
    def train_model(self, data_dir, epochs=50, batch_size=32, validation_split=0.2, use_simple_model=True):
        """Train the emotion detection model"""
        print("Loading and preprocessing data...")
        X, y = self.load_and_preprocess_data(data_dir)
        
        print(f"Dataset shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=np.argmax(y, axis=1)
        )
        
        # Data augmentation
        train_datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True
        )
        
        # Create model - use simple model by default to avoid dimension issues
        try:
            if use_simple_model:
                print("Creating simple model architecture...")
                self.model = self.create_simple_model()
            else:
                print("Creating complex model architecture...")
                self.model = self.create_model()
            
            print("Model created successfully!")
            print(self.model.summary())
        except Exception as e:
            print(f"Error creating complex model: {e}")
            print("Falling back to simple model...")
            self.model = self.create_simple_model()
            print("Simple model created successfully!")
            print(self.model.summary())
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.0001
        )
        
        # Train model
        print("Starting training...")
        history = self.model.fit(
            train_datagen.flow(X_train, y_train, batch_size=batch_size),
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Save model
        self.model.save('emotion_model.h5')
        print("Model saved as 'emotion_model.h5'")
        
        return history
    
    def load_model(self, model_path='emotion_model.h5'):
        """Load a pre-trained model"""
        if os.path.exists(model_path):
            self.model = tf.keras.models.load_model(model_path)
            print(f"Model loaded from {model_path}")
            return True
        else:
            print(f"Model file {model_path} not found!")
            return False
    
    def predict_emotion(self, image):
        """Predict emotion from image"""
        if self.model is None:
            print("Model not loaded! Please train or load a model first.")
            return None
        
        # Preprocess image
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        image = cv2.resize(image, self.img_size)
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=-1)
        image = np.expand_dims(image, axis=0)
        
        # Predict
        predictions = self.model.predict(image)
        emotion_idx = np.argmax(predictions[0])
        confidence = predictions[0][emotion_idx]
        
        return {
            'emotion': self.emotions[emotion_idx],
            'confidence': float(confidence),
            'all_predictions': {
                emotion: float(conf) for emotion, conf in zip(self.emotions, predictions[0])
            }
        }
    
    def predict_from_file(self, image_path):
        """Predict emotion from image file"""
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not load image from {image_path}")
            return None
        return self.predict_emotion(image)

# Example usage
if __name__ == "__main__":
    detector = EmotionDetector()
    
    # Train model (uncomment to train)
    # history = detector.train_model('archive/train')
    
    # Or load pre-trained model
    if detector.load_model():
        # Test prediction
        test_image_path = "archive/train/happy/Training_7215.jpg"  # Example path
        if os.path.exists(test_image_path):
            result = detector.predict_from_file(test_image_path)
            print(f"Predicted emotion: {result['emotion']} (confidence: {result['confidence']:.2f})")
