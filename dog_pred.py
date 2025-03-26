import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Global variables (Load only once)
model = None
scaler = None
label_encoder = None

def load_resources():
    """Loads model and preprocessing objects only once."""
    global model, scaler, label_encoder
    
    if model is None:
        model = load_model("dog_emotion_model.h5")

    if label_encoder is None:
        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.load("label_encoder_classes.npy", allow_pickle=True)

    if scaler is None:
        scaler = MinMaxScaler()
        scaler.min_ = np.load("scaler_min.npy", allow_pickle=True)
        scaler.scale_ = np.load("scaler_scale.npy", allow_pickle=True)

def predict_emotion(data):
    """Predicts dog's emotion based on input physiological data."""
    # Ensure resources are loaded
    load_resources()

    # Prepare input data
    df = pd.DataFrame([{
        "Heart Rate (bpm)": data["heartRate"],
        "Body Temperature (°C)": data["bodyTemperature"],
        "Tail Wagging (m/s²)": data["accelMagnitude"]
    }])

    # Normalize input
    features = ["Heart Rate (bpm)", "Body Temperature (°C)", "Tail Wagging (m/s²)"]
    df[features] = scaler.transform(df[features])

    # Reshape input for LSTM model
    X_input = df.values.reshape((1, 1, df.shape[1]))

    # Make prediction
    y_pred = model.predict(X_input)
    predicted_class = y_pred.argmax(axis=1)[0]
    predicted_emotion = label_encoder.inverse_transform([predicted_class])[0]

    return {"emotion": predicted_emotion}
