"""
Quick training script with a smaller subset of data for testing
"""

from emotion_model import EmotionDetector
import os
import random

def quick_train():
    """Train the model on a smaller subset for quick testing"""
    print("🚀 Quick Training - Using Subset of Data")
    print("=" * 50)
    
    # Check if dataset exists
    dataset_path = "archive/train"
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at {dataset_path}")
        return
    
    print(f"✅ Dataset found at {dataset_path}")
    
    # Initialize detector
    detector = EmotionDetector()
    
    # Create a subset of data for quick training
    print("\n📊 Creating subset of data for quick training...")
    subset_path = "archive/train_subset"
    
    if os.path.exists(subset_path):
        import shutil
        shutil.rmtree(subset_path)
    
    os.makedirs(subset_path)
    
    # Copy a small subset from each emotion folder
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    subset_size = 100  # 100 images per emotion
    
    for emotion in emotions:
        emotion_dir = os.path.join(dataset_path, emotion)
        subset_emotion_dir = os.path.join(subset_path, emotion)
        os.makedirs(subset_emotion_dir, exist_ok=True)
        
        if os.path.exists(emotion_dir):
            # Get all jpg files
            jpg_files = [f for f in os.listdir(emotion_dir) if f.endswith('.jpg')]
            
            # Select random subset
            selected_files = random.sample(jpg_files, min(subset_size, len(jpg_files)))
            
            # Copy selected files
            for file in selected_files:
                src = os.path.join(emotion_dir, file)
                dst = os.path.join(subset_emotion_dir, file)
                import shutil
                shutil.copy2(src, dst)
            
            print(f"   {emotion}: {len(selected_files)} images")
    
    print(f"\n✅ Subset created with {subset_size} images per emotion")
    
    # Train model on subset
    print("\n🚀 Starting quick training...")
    try:
        history = detector.train_model(
            data_dir=subset_path,
            epochs=10,  # Fewer epochs for quick training
            batch_size=16,  # Smaller batch size
            validation_split=0.2,
            use_simple_model=True
        )
        
        print("\n✅ Quick training completed successfully!")
        print("📁 Model saved as 'emotion_model.h5'")
        print("🎯 You can now run the Streamlit app with: streamlit run app.py")
        
        # Clean up subset
        import shutil
        shutil.rmtree(subset_path)
        print("🧹 Cleaned up temporary subset data")
        
    except Exception as e:
        print(f"❌ Error during training: {str(e)}")
        print("Please check your dataset and try again.")

if __name__ == "__main__":
    quick_train()
