import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load dataset
df = pd.read_excel("dog_emotion_dataset.xlsx")

# Encode labels
label_encoder = LabelEncoder()
df["Emotion"] = label_encoder.fit_transform(df["Emotion"])

# Scale features
scaler = MinMaxScaler()
features = ["Heart Rate (bpm)", "Body Temperature (°C)", "Tail Wagging (m/s²)"]
df[features] = scaler.fit_transform(df[features])

# Prepare data for LSTM
X = df[features].values.reshape((df.shape[0], 1, len(features)))
y = df["Emotion"].values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define model
model = Sequential([
    Input(shape=(X_train.shape[1], X_train.shape[2])),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.2),
    Bidirectional(LSTM(32)),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')
])

model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train model
model.fit(X_train, y_train, epochs=50, batch_size=100, validation_data=(X_test, y_test), verbose=1)

# Save model and preprocessing objects
model.save("dog_emotion_model.h5")
np.save("label_encoder_classes.npy", label_encoder.classes_)
np.save("scaler_min.npy", scaler.min_)
np.save("scaler_scale.npy", scaler.scale_)

print("Model and preprocessing objects saved successfully.")
