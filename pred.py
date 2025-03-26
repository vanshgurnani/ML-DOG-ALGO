import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

# Load model
model = load_model("dog_emotion_model.h5")

# Load preprocessing objects
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load("label_encoder_classes.npy", allow_pickle=True)

scaler = MinMaxScaler()
scaler.min_ = np.load("scaler_min.npy", allow_pickle=True)
scaler.scale_ = np.load("scaler_scale.npy", allow_pickle=True)

def predict_emotion(data):
    """Predicts dog's emotion based on input physiological data."""
    # Adjust input keys to match expected feature names
    df = pd.DataFrame([{
        "Heart Rate (bpm)": data["hearRate"],
        "Body Temperature (°C)": data["bodyTemperature"],
        "Tail Wagging (m/s²)": data["accelMagnitude"]
    }])

    # Normalize input
    features = ["Heart Rate (bpm)", "Body Temperature (°C)", "Tail Wagging (m/s²)"]
    df[features] = scaler.transform(df[features])

    X_input = df.values.reshape((1, 1, df.shape[1]))

    # Make prediction
    y_pred = model.predict(X_input)
    predicted_class = y_pred.argmax(axis=1)[0]
    predicted_emotion = label_encoder.inverse_transform([predicted_class])[0]
    confidence_score = float(y_pred[0][predicted_class])  # Convert to standard Python float

    return {"emotion": predicted_emotion, "confidence": round(confidence_score, 4)}

if __name__ == "__main__":
    sample_input = {
        "hearRate": 98,
        "bodyTemperature": 38.41,
        "accelMagnitude": 3.91
    }
    result = predict_emotion(sample_input)
    print("Predicted Emotion:", result["emotion"], "| Confidence:", result["confidence"])
