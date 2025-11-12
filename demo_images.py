"""
Script to create demo images for testing the emotion detection system
"""

import os
import shutil
import random

def create_demo_images():
    """Create a demo_images folder with sample images from each emotion category"""
    
    # Create demo_images directory
    demo_dir = "demo_images"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir)
    
    # Source directories
    source_dirs = {
        'angry': 'archive/train/angry',
        'disgust': 'archive/train/disgust', 
        'fear': 'archive/train/fear',
        'happy': 'archive/train/happy',
        'neutral': 'archive/train/neutral',
        'sad': 'archive/train/sad',
        'surprise': 'archive/train/surprise'
    }
    
    # Copy 3 random images from each emotion category
    for emotion, source_dir in source_dirs.items():
        if os.path.exists(source_dir):
            # Get all jpg files
            jpg_files = [f for f in os.listdir(source_dir) if f.endswith('.jpg')]
            
            # Select 3 random files
            selected_files = random.sample(jpg_files, min(3, len(jpg_files)))
            
            # Create emotion subdirectory
            emotion_dir = os.path.join(demo_dir, emotion)
            os.makedirs(emotion_dir, exist_ok=True)
            
            # Copy selected files
            for file in selected_files:
                src = os.path.join(source_dir, file)
                dst = os.path.join(emotion_dir, file)
                shutil.copy2(src, dst)
                print(f"Copied {file} to {emotion} folder")
        else:
            print(f"Warning: {source_dir} not found")
    
    print(f"\n✅ Demo images created in '{demo_dir}' folder")
    print("You can use these images to test the emotion detection system")

if __name__ == "__main__":
    create_demo_images()
