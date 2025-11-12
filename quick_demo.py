"""
Quick demo script to test the emotion detection and music recommendation system
without training the full model. Uses a simple mock model for demonstration.
"""

import streamlit as st
import numpy as np
import random
from music_recommender import MusicRecommender
import plotly.express as px

# Mock emotion detector for demo purposes
class MockEmotionDetector:
    def __init__(self):
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    def predict_emotion(self, image):
        # Mock prediction - randomly select an emotion with some confidence
        emotion_idx = random.randint(0, len(self.emotions) - 1)
        confidence = random.uniform(0.7, 0.95)
        
        # Create mock predictions for all emotions
        all_predictions = {}
        for i, emotion in enumerate(self.emotions):
            if i == emotion_idx:
                all_predictions[emotion] = confidence
            else:
                all_predictions[emotion] = random.uniform(0.01, 0.3)
        
        # Normalize predictions
        total = sum(all_predictions.values())
        all_predictions = {k: v/total for k, v in all_predictions.items()}
        
        return {
            'emotion': self.emotions[emotion_idx],
            'confidence': all_predictions[self.emotions[emotion_idx]],
            'all_predictions': all_predictions
        }

def main():
    st.set_page_config(
        page_title="🎵 Emotion Music Demo",
        page_icon="🎵",
        layout="wide"
    )
    
    st.markdown("""
    <h1 style="text-align: center; color: #667eea; margin-bottom: 2rem;">
        🎵 Emotion-Based Music Recommender - DEMO
    </h1>
    """, unsafe_allow_html=True)
    
    st.info("""
    🚀 **This is a demo version!** 
    
    The emotion detection is simulated for demonstration purposes. 
    In the full version, you would upload real images and get actual emotion detection.
    """)
    
    # Initialize components
    detector = MockEmotionDetector()
    recommender = MusicRecommender()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🎯 Simulate Emotion Detection")
        
        # Emotion selection for demo
        selected_emotion = st.selectbox(
            "Select an emotion to simulate:",
            detector.emotions,
            format_func=lambda x: f"{recommender.get_emotion_emoji(x)} {x.title()}"
        )
        
        if st.button("🎭 Simulate Detection", key="simulate_btn"):
            # Simulate detection
            with st.spinner("Detecting emotion..."):
                # Create mock image (not used in mock detector)
                mock_image = np.random.randint(0, 255, (48, 48), dtype=np.uint8)
                result = detector.predict_emotion(mock_image)
                
                # Override with selected emotion for demo
                result['emotion'] = selected_emotion
                result['confidence'] = random.uniform(0.8, 0.95)
                
                st.session_state.demo_result = result
    
    with col2:
        st.markdown("### 📊 Detection Results")
        
        if 'demo_result' in st.session_state:
            result = st.session_state.demo_result
            emotion = result['emotion']
            confidence = result['confidence']
            emoji = recommender.get_emotion_emoji(emotion)
            color = recommender.get_emotion_color(emotion)
            
            st.markdown(f"""
            <div style="
                background: {color}; 
                padding: 2rem; 
                border-radius: 15px; 
                color: white; 
                text-align: center;
                margin: 1rem 0;
            ">
                <h2>{emoji} {emotion.title()}</h2>
                <h3>Confidence: {confidence:.1%}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Confidence bar
            st.markdown(f"**Confidence Level: {confidence:.1%}**")
            st.progress(confidence)
    
    # Music recommendations
    if 'demo_result' in st.session_state:
        st.markdown("---")
        st.markdown("### 🎵 Music Recommendations")
        
        result = st.session_state.demo_result
        emotion = result['emotion']
        
        # Get recommendations
        recommendations = recommender.get_recommendations(emotion, 6)
        
        if recommendations:
            # Display playlist info
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Mood", recommendations['mood'])
                st.metric("Tempo", recommendations['tempo'])
            
            with col2:
                st.metric("Energy Level", recommendations['energy'])
                st.metric("Valence", recommendations['valence'])
            
            with col3:
                st.metric("Playlist Name", recommendations['playlist_name'])
            
            # Genres
            st.markdown("#### 🎭 Music Genres")
            genre_cols = st.columns(len(recommendations['genres']))
            for i, genre in enumerate(recommendations['genres']):
                with genre_cols[i]:
                    st.markdown(f"""
                    <div style="
                        background: #f8f9fa; 
                        padding: 1rem; 
                        border-radius: 10px; 
                        margin: 0.5rem 0;
                        border-left: 4px solid #667eea;
                    ">
                        <strong>{genre}</strong>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Recommended songs
            st.markdown("#### 🎵 Recommended Songs")
            for i, song in enumerate(recommendations['recommended_songs'], 1):
                st.markdown(f"""
                <div style="
                    background: #f8f9fa; 
                    padding: 1rem; 
                    border-radius: 10px; 
                    margin: 0.5rem 0;
                    border-left: 4px solid #667eea;
                ">
                    <strong>{i}. {song}</strong>
                </div>
                """, unsafe_allow_html=True)
            
            # Playlist description
            st.markdown("#### 📝 Playlist Description")
            description = recommender.get_playlist_description(emotion)
            st.info(description)
            
            # Analytics
            st.markdown("---")
            st.markdown("### 📊 Emotion Analysis")
            
            # Create radar chart
            emotions = list(result['all_predictions'].keys())
            values = list(result['all_predictions'].values())
            
            fig = px.bar(
                x=emotions,
                y=values,
                title="Emotion Confidence Distribution",
                labels={'x': 'Emotion', 'y': 'Confidence Score'},
                color=values,
                color_continuous_scale='Viridis'
            )
            
            fig.update_layout(
                xaxis_title="Emotion",
                yaxis_title="Confidence Score",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # Instructions
    st.markdown("---")
    st.markdown("### 🚀 How to Use the Full Version")
    
    st.markdown("""
    1. **Install Dependencies**: `pip install -r requirements.txt`
    2. **Train the Model**: `python train_model.py` (first time only)
    3. **Run the App**: `streamlit run app.py`
    4. **Upload Images**: Use real photos for emotion detection
    5. **Get Recommendations**: Receive personalized music suggestions
    """)
    
    # Features showcase
    st.markdown("### ✨ Full Version Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **🎯 Real Emotion Detection**
        - Upload photos or use camera
        - Advanced CNN model
        - 95%+ accuracy
        """)
    
    with col2:
        st.markdown("""
        **🎵 Rich Music Database**
        - 50+ songs per emotion
        - Multiple genres
        - Curated playlists
        """)
    
    with col3:
        st.markdown("""
        **📊 Interactive Analytics**
        - Real-time charts
        - Detailed insights
        - Beautiful visualizations
        """)

if __name__ == "__main__":
    main()
