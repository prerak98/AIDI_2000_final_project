import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np
import pandas as pd
import json

# Load the preprocessed data
with open('../output/preprocessed_data.json', 'r') as f:
    data = [json.loads(line) for line in f]

df = pd.DataFrame(data)

# Convert lists back to numpy arrays
df['mfcc'] = df['mfcc'].apply(lambda x: np.array(x))
df['chroma'] = df['chroma'].apply(lambda x: np.array(x))
df['spectral_contrast'] = df['spectral_contrast'].apply(lambda x: np.array(x))

# Prepare the feature matrix (X) and target vector (y)
X = np.hstack([
    np.vstack(df['mfcc'].values),
    np.vstack(df['chroma'].values),
    np.vstack(df['spectral_contrast'].values)
])
y = df['emotion']

# Encode the labels (y)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Feature normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape the data for the CNN
X_scaled = X_scaled.reshape(-1, X_scaled.shape[1], 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Build the CNN model
model = models.Sequential([
    layers.Conv1D(32, 3, activation='relu', input_shape=(X_train.shape[1], 1)),
    layers.MaxPooling1D(2),
    layers.Conv1D(64, 3, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(np.unique(y_train)), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the CNN model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the CNN model
cnn_accuracy = model.evaluate(X_test, y_test)[1]
print(f"CNN Model Accuracy: {cnn_accuracy:.4f}")

# Save the model in the recommended Keras format
model.save('cnn_model.keras')

