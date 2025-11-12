import random
import json
import os

class MusicRecommender:
    def __init__(self):
        self.emotion_music_mapping = {
            'happy': {
                'genres': ['Pop', 'Dance', 'Electronic', 'Funk', 'Disco'],
                'mood': 'Upbeat and energetic',
                'artists': ['Taylor Swift', 'Bruno Mars', 'Dua Lipa', 'Ed Sheeran', 'Ariana Grande'],
                'songs': [
                    'Happy - Pharrell Williams',
                    'Good as Hell - Lizzo',
                    'Can\'t Stop the Feeling - Justin Timberlake',
                    'Shake It Off - Taylor Swift',
                    'Uptown Funk - Bruno Mars',
                    'Dancing Queen - ABBA',
                    'I Gotta Feeling - Black Eyed Peas',
                    'Walking on Sunshine - Katrina and the Waves'
                ],
                'tempo': 'Fast (120-140 BPM)',
                'energy': 'High',
                'valence': 'Very Positive'
            },
            'sad': {
                'genres': ['Blues', 'Soul', 'Indie', 'Alternative', 'Folk'],
                'mood': 'Melancholic and introspective',
                'artists': ['Adele', 'Sam Smith', 'Billie Eilish', 'Lana Del Rey', 'Bon Iver'],
                'songs': [
                    'Someone Like You - Adele',
                    'Stay With Me - Sam Smith',
                    'When the Party\'s Over - Billie Eilish',
                    'Skinny Love - Bon Iver',
                    'Hurt - Johnny Cash',
                    'Mad World - Gary Jules',
                    'The Night We Met - Lord Huron',
                    'All I Want - Kodaline'
                ],
                'tempo': 'Slow (60-80 BPM)',
                'energy': 'Low',
                'valence': 'Negative'
            },
            'angry': {
                'genres': ['Rock', 'Metal', 'Punk', 'Alternative Rock', 'Grunge'],
                'mood': 'Aggressive and intense',
                'artists': ['Linkin Park', 'Rage Against the Machine', 'System of a Down', 'Nirvana', 'Foo Fighters'],
                'songs': [
                    'In the End - Linkin Park',
                    'Killing in the Name - Rage Against the Machine',
                    'Chop Suey - System of a Down',
                    'Smells Like Teen Spirit - Nirvana',
                    'The Pretender - Foo Fighters',
                    'Bulls on Parade - Rage Against the Machine',
                    'Numb - Linkin Park',
                    'Basket Case - Green Day'
                ],
                'tempo': 'Fast (140-180 BPM)',
                'energy': 'Very High',
                'valence': 'Negative'
            },
            'fear': {
                'genres': ['Ambient', 'Dark Electronic', 'Industrial', 'Gothic', 'Experimental'],
                'mood': 'Dark and unsettling',
                'artists': ['Nine Inch Nails', 'Aphex Twin', 'Burial', 'Boards of Canada', 'Tim Hecker'],
                'songs': [
                    'Closer - Nine Inch Nails',
                    'Windowlicker - Aphex Twin',
                    'Archangel - Burial',
                    'Roygbiv - Boards of Canada',
                    'Ravedeath, 1972 - Tim Hecker',
                    'The Downward Spiral - Nine Inch Nails',
                    'Untrue - Burial',
                    'Geogaddi - Boards of Canada'
                ],
                'tempo': 'Variable (60-120 BPM)',
                'energy': 'Medium',
                'valence': 'Very Negative'
            },
            'surprise': {
                'genres': ['Jazz', 'Fusion', 'Progressive Rock', 'Experimental', 'World Music'],
                'mood': 'Unexpected and dynamic',
                'artists': ['Frank Zappa', 'King Crimson', 'Miles Davis', 'Björk', 'Radiohead'],
                'songs': [
                    'Bohemian Rhapsody - Queen',
                    'Paranoid Android - Radiohead',
                    'So What - Miles Davis',
                    '21st Century Schizoid Man - King Crimson',
                    'Army of Me - Björk',
                    'Peaches en Regalia - Frank Zappa',
                    'Karma Police - Radiohead',
                    'Bitches Brew - Miles Davis'
                ],
                'tempo': 'Variable (80-160 BPM)',
                'energy': 'Medium-High',
                'valence': 'Mixed'
            },
            'disgust': {
                'genres': ['Industrial', 'Noise', 'Death Metal', 'Grindcore', 'Experimental'],
                'mood': 'Harsh and repulsive',
                'artists': ['Merzbow', 'Throbbing Gristle', 'Carcass', 'Napalm Death', 'Whitehouse'],
                'songs': [
                    'Pulse Demon - Merzbow',
                    'Hamburger Lady - Throbbing Gristle',
                    'Heartwork - Carcass',
                    'You Suffer - Napalm Death',
                    'Dedication to Peter Kürten - Whitehouse',
                    'The Great Annihilator - Swans',
                    'Filth - Swans',
                    'Cop - Swans'
                ],
                'tempo': 'Extreme (200+ BPM)',
                'energy': 'Extreme',
                'valence': 'Very Negative'
            },
            'neutral': {
                'genres': ['Ambient', 'Chillout', 'Lounge', 'New Age', 'Classical'],
                'mood': 'Calm and balanced',
                'artists': ['Brian Eno', 'Aphex Twin', 'Boards of Canada', 'Max Richter', 'Ludovico Einaudi'],
                'songs': [
                    'Music for Airports - Brian Eno',
                    'Avril 14th - Aphex Twin',
                    'Roygbiv - Boards of Canada',
                    'The Blue Notebooks - Max Richter',
                    'Nuvole Bianche - Ludovico Einaudi',
                    'Weightless - Marconi Union',
                    'Spiegel im Spiegel - Arvo Pärt',
                    'Clair de Lune - Claude Debussy'
                ],
                'tempo': 'Slow-Medium (60-100 BPM)',
                'energy': 'Low-Medium',
                'valence': 'Neutral'
            }
        }
    
    def get_recommendations(self, emotion, num_recommendations=5):
        """Get music recommendations based on detected emotion"""
        # Normalize emotion key for robustness
        key = (emotion or "").strip().lower()
        if key not in self.emotion_music_mapping:
            # Fallback to neutral if unknown
            key = 'neutral'
        
        emotion_data = self.emotion_music_mapping[key]
        
        # Select random songs from the emotion's song list
        recommended_songs = random.sample(
            emotion_data['songs'], 
            min(num_recommendations, len(emotion_data['songs']))
        )
        
        # Select random artists
        recommended_artists = random.sample(
            emotion_data['artists'], 
            min(3, len(emotion_data['artists']))
        )
        
        return {
            'emotion': key,
            'mood': emotion_data['mood'],
            'genres': emotion_data['genres'],
            'tempo': emotion_data['tempo'],
            'energy': emotion_data['energy'],
            'valence': emotion_data['valence'],
            'recommended_songs': recommended_songs,
            'recommended_artists': recommended_artists,
            'playlist_name': f"{key.title()} Vibes Playlist"
        }
    
    def get_playlist_description(self, emotion):
        """Get a detailed description for the emotion-based playlist"""
        descriptions = {
            'happy': "🎉 This upbeat playlist is perfect for lifting your spirits! Featuring energetic pop, dance, and electronic tracks that will make you want to move and groove.",
            'sad': "💙 A melancholic collection of soulful tracks to accompany your emotions. These songs understand what you're going through and provide comfort through music.",
            'angry': "🔥 Channel your energy with this intense rock and metal playlist. Let the powerful riffs and aggressive beats help you release that built-up tension.",
            'fear': "🌑 Dark and atmospheric tracks that embrace the unsettling feeling. This experimental playlist explores the depths of electronic and ambient music.",
            'surprise': "🎭 Unexpected and dynamic tracks that keep you guessing. This eclectic mix spans multiple genres and will surprise you with every song.",
            'disgust': "⚡ Harsh and intense music for when you need something raw and unfiltered. This playlist is not for the faint of heart.",
            'neutral': "☁️ A calm and balanced collection of ambient and chillout tracks. Perfect for relaxation, meditation, or when you need some peaceful background music."
        }
        return descriptions.get(emotion, "A carefully curated playlist based on your current emotion.")
    
    def get_emotion_color(self, emotion):
        """Get color associated with each emotion for UI"""
        colors = {
            'happy': '#FFD700',  # Gold
            'sad': '#4169E1',    # Royal Blue
            'angry': '#DC143C',  # Crimson
            'fear': '#4B0082',   # Indigo
            'surprise': '#FF69B4', # Hot Pink
            'disgust': '#8B4513', # Saddle Brown
            'neutral': '#808080'  # Gray
        }
        return colors.get(emotion, '#808080')
    
    def get_emotion_emoji(self, emotion):
        """Get emoji associated with each emotion"""
        emojis = {
            'happy': '😊',
            'sad': '😢',
            'angry': '😠',
            'fear': '😨',
            'surprise': '😲',
            'disgust': '🤢',
            'neutral': '😐'
        }
        return emojis.get(emotion, '😐')
    
    def save_recommendations(self, recommendations, filename='music_recommendations.json'):
        """Save recommendations to a JSON file"""
        with open(filename, 'w') as f:
            json.dump(recommendations, f, indent=2)
        print(f"Recommendations saved to {filename}")
    
    def load_recommendations(self, filename='music_recommendations.json'):
        """Load recommendations from a JSON file"""
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                return json.load(f)
        return None

# Example usage
if __name__ == "__main__":
    recommender = MusicRecommender()
    
    # Test recommendations for different emotions
    emotions = ['happy', 'sad', 'angry', 'fear', 'surprise', 'disgust', 'neutral']
    
    for emotion in emotions:
        print(f"\n--- {emotion.upper()} RECOMMENDATIONS ---")
        recommendations = recommender.get_recommendations(emotion, 3)
        if recommendations:
            print(f"Mood: {recommendations['mood']}")
            print(f"Genres: {', '.join(recommendations['genres'])}")
            print(f"Recommended Songs: {', '.join(recommendations['recommended_songs'])}")
            print(f"Description: {recommender.get_playlist_description(emotion)}")
