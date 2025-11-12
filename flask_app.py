from flask import Flask, render_template, request, redirect, url_for, send_from_directory, session
from werkzeug.utils import secure_filename
import os
import numpy as np
import cv2
from PIL import Image
from emotion_model import EmotionDetector
from music_recommender import MusicRecommender


app = Flask(__name__)
app.config['SECRET_KEY'] = 'change-this-secret'
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


emotion_detector = EmotionDetector()
model_loaded = emotion_detector.load_model()
music_recommender = MusicRecommender()


def process_image_for_model(pil_image: Image.Image):
    image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_eq = clahe.apply(gray)

    frontal_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    alt_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

    faces = ()
    param_grid = [
        (1.1, 5, (60, 60)),
        (1.05, 5, (60, 60)),
        (1.2, 3, (40, 40)),
        (1.3, 3, (30, 30)),
    ]

    for scale, neighbors, min_size in param_grid:
        faces = frontal_cascade.detectMultiScale(gray_eq, scaleFactor=scale, minNeighbors=neighbors, minSize=min_size)
        if len(faces) == 0:
            faces = alt_cascade.detectMultiScale(gray_eq, scaleFactor=scale, minNeighbors=neighbors, minSize=min_size)
        if len(faces) > 0:
            break

    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_roi = gray_eq[y:y+h, x:x+w]
        return face_roi

    h_img, w_img = gray_eq.shape[:2]
    side = min(h_img, w_img)
    cx, cy = w_img // 2, h_img // 2
    half = side // 2
    x0, y0 = max(0, cx - half), max(0, cy - half)
    x1, y1 = x0 + side, y0 + side
    fallback_roi = gray_eq[y0:y1, x0:x1]
    return fallback_roi


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/detect', methods=['GET', 'POST'])
def detect():
    error = None
    result = None

    if request.method == 'POST':
        file = request.files.get('image')
        camera_image = request.form.get('camera_image')

        if not model_loaded:
            error = 'Model not found. Please train the model first.'
        else:
            pil = None

            if camera_image and camera_image.startswith('data:image'):
                try:
                    header, b64 = camera_image.split(',')
                    import base64
                    from io import BytesIO
                    data = base64.b64decode(b64)
                    pil = Image.open(BytesIO(data)).convert('RGB')
                except Exception:
                    error = 'Invalid camera image data.'
            elif file and file.filename != '':
                filename = secure_filename(file.filename)
                path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(path)
                pil = Image.open(path).convert('RGB')
            else:
                error = 'Please choose an image file or use the camera.'

            if pil is not None and error is None:
                processed = process_image_for_model(pil)
                result = emotion_detector.predict_emotion(processed)
                if result:
                    session['last_prediction'] = result
                else:
                    error = 'Failed to detect emotion. Try another image.'

    return render_template('detect.html', error=error, result=result, recommender=music_recommender)


@app.route('/recommendations')
def recommendations():
    last_prediction = session.get('last_prediction')
    emotion = None
    confidence = None
    rec = None

    if last_prediction:
        emotion = last_prediction['emotion']
        confidence = last_prediction['confidence']
        rec = music_recommender.get_recommendations(emotion, 8)

    return render_template('recommendations.html', emotion=emotion, confidence=confidence, rec=rec, recommender=music_recommender)


@app.route('/analytics')
def analytics():
    last_prediction = session.get('last_prediction')
    data = None
    if last_prediction:
        data = last_prediction['all_predictions']
    return render_template('analytics.html', preds=data)


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/static/<path:filename>')
def static_files(filename):
    static_dir = os.path.join(os.path.dirname(__file__), 'static')
    return send_from_directory(static_dir, filename)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


