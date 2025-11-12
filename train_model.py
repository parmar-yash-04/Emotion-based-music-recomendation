"""
Script to train the emotion detection model
Run this script to train the model on your dataset
"""

from emotion_model import EmotionDetector
import os

def main():
    print("🎵 Emotion-Based Music Recommender - Model Training")
    print("=" * 50)
    
    # Check if dataset exists
    dataset_path = "archive/train"
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at {dataset_path}")
        print("Please make sure your dataset is in the correct location.")
        return
    
    print(f"✅ Dataset found at {dataset_path}")
    
    # Initialize detector
    detector = EmotionDetector()
    
    # Train model
    print("\n🚀 Starting model training...")
    print("This may take several hours depending on your hardware.")
    print("You can monitor the progress below:\n")
    
    try:
        history = detector.train_model(
            data_dir=dataset_path,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            use_simple_model=True  # Use simple model to avoid dimension issues
        )
        
        print("\n✅ Model training completed successfully!")
        print("📁 Model saved as 'emotion_model.h5'")
        print("🎯 You can now run the Streamlit app with: streamlit run app.py")
        
        # Display training summary
        if history:
            print(f"\n📊 Training Summary:")
            print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.4f}")
            print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
            print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
            print(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        print("Please check your dataset and try again.")

if __name__ == "__main__":
    main()
