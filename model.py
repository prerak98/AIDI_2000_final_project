from flask import Flask, request, jsonify, render_template
import numpy as np
import librosa
import tensorflow as tf
from keras.models import load_model
import os

app = Flask(__name__)

# Load the trained model
model = load_model('emotion_recognition_model.h5')

# Function to extract features from the audio file
def extract_features(file_path):
    try:
        print(f"Extracting features from {file_path}")
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled = np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Error encountered while parsing file: {file_path}. Error: {str(e)}")
        return None
    
    return mfccs_scaled

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    print("Received a request for prediction")
    
    # Check if an audio file was uploaded
    if 'file' not in request.files:
        print("No file part in the request")
        return jsonify({"error": "No file part in the request"}), 400
    
    file = request.files['file']
    
    # Save the file locally
    file_path = os.path.join("uploads", file.filename)
    print(f"Saving file to {file_path}")
    file.save(file_path)
    
    # Extract features and reshape for prediction
    features = extract_features(file_path)
    if features is None:
        return jsonify({"error": "Error extracting features from the audio file"}), 500
    
    features = np.array([features])  # Reshape for model input
    
    # Make prediction
    print("Making prediction")
    prediction = model.predict(features)
    predicted_emotion = np.argmax(prediction, axis=1)[0]
    
    # Clean up the uploaded file
    os.remove(file_path)
    
    print(f"Predicted emotion: {predicted_emotion}")
    return jsonify({'predicted_emotion': int(predicted_emotion)})

if __name__ == '__main__':
    # Create uploads folder if it doesn't exist
    if not os.path.exists("uploads"):
        os.makedirs("uploads")

    print("Starting Flask server")
    app.run(debug=True)
