import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_excel("dog_emotion_dataset.xlsx")

# Encode Labels
label_encoder = LabelEncoder()
df["Emotion"] = label_encoder.fit_transform(df["Emotion"])

# Normalize Features
scaler = MinMaxScaler()
feature_cols = ["Heart Rate (bpm)", "Body Temperature (°C)", "Tail Wagging (m/s²)"]
df[feature_cols] = scaler.fit_transform(df[feature_cols])

# Prepare Data
X = df[feature_cols].values.reshape((df.shape[0], 1, len(feature_cols)))
y = df["Emotion"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Update model architecture with Input layer
model = Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2])),  # Explicit input layer
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.2),
    Bidirectional(LSTM(32)),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])

# Verify data shapes
print("X_train shape:", X_train.shape)  # Should be (samples, timesteps=1, features=3)
print("y_train shape:", y_train.shape)

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=50, batch_size=100, validation_data=(X_test, y_test), verbose=1)

# Evaluate Model
y_pred = model.predict(X_test)
y_pred_classes = y_pred.argmax(axis=1)

accuracy = accuracy_score(y_test, y_pred_classes)
f1 = f1_score(y_test, y_pred_classes, average="weighted")

print(f"Model Accuracy: {accuracy:.4f}")
print(f"Model F1 Score: {f1:.4f}")

print("Classification Report:\n", classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_))

# Save Model & Preprocessing Objects
model.save("dog_emotion_model.h5")
np.save("label_encoder_classes.npy", label_encoder.classes_)
import joblib
joblib.dump(scaler, "dog_scaler.pkl")

print("Model, Label Encoder Classes, and Scaler saved successfully!")
