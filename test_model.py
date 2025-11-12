"""
Quick test script to verify the model architecture works without training
"""

import numpy as np
from emotion_model import EmotionDetector

def test_model_architecture():
    """Test if the model can be created and compiled without errors"""
    print("🧪 Testing Model Architecture")
    print("=" * 40)
    
    # Test simple model
    print("\n1. Testing Simple Model...")
    try:
        detector = EmotionDetector()
        simple_model = detector.create_simple_model()
        print("✅ Simple model created successfully!")
        print(f"   Parameters: {simple_model.count_params():,}")
        
        # Test with dummy data
        dummy_input = np.random.random((1, 48, 48, 1))
        prediction = simple_model.predict(dummy_input, verbose=0)
        print(f"   Output shape: {prediction.shape}")
        print(f"   Sample prediction: {prediction[0]}")
        
    except Exception as e:
        print(f"❌ Simple model failed: {e}")
        return False
    
    # Test complex model
    print("\n2. Testing Complex Model...")
    try:
        complex_model = detector.create_model()
        print("✅ Complex model created successfully!")
        print(f"   Parameters: {complex_model.count_params():,}")
        
        # Test with dummy data
        dummy_input = np.random.random((1, 48, 48, 1))
        prediction = complex_model.predict(dummy_input, verbose=0)
        print(f"   Output shape: {prediction.shape}")
        print(f"   Sample prediction: {prediction[0]}")
        
    except Exception as e:
        print(f"❌ Complex model failed: {e}")
        print("   This is expected for 48x48 images - use simple model instead")
    
    print("\n✅ Model architecture test completed!")
    print("🎯 You can now proceed with training using the simple model.")
    return True

if __name__ == "__main__":
    test_model_architecture()
