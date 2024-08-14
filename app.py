from flask import Flask, request, render_template, redirect, url_for, jsonify
import librosa
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads/'

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load your pre-trained model
model = tf.keras.models.load_model('models/cnn_model.h5')

def extract_features(file_path):
    try:
        # Load the audio file and extract MFCC features
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
        return np.expand_dims(mfccs, axis=-1).reshape(1, -1, 1)
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

@app.route('/')
def home():
    # Serve the frontend HTML
    return render_template('index.html')

@app.route('/predict-audio', methods=['POST'])
def predict_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            file.save(file_path)
        except Exception as e:
            print(f"Error saving file: {e}")
            return jsonify({"error": "Failed to save file"}), 500

        features = extract_features(file_path)
        if features is None:
            return jsonify({"error": "Failed to extract features"}), 500
        
        try:
            prediction = model.predict(features)
            predicted_emotion = np.argmax(prediction)
        except Exception as e:
            print(f"Error during prediction: {e}")
            return jsonify({"error": "Prediction failed"}), 500

        os.remove(file_path)

        emotion_map = {0: 'Angry', 1: 'Happy', 2: 'Sad', 3: 'Neutral'}  # Adjust based on your model's classes
        predicted_emotion_text = emotion_map.get(predicted_emotion, "Unknown")

        return jsonify({"emotion": predicted_emotion_text})

if __name__ == "__main__":
    app.run(debug=True)
