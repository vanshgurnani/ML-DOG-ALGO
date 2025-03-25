import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

# Load Model & Preprocessing Objects
model = load_model("dog_emotion_model.h5")
label_encoder_classes = np.load("label_encoder_classes.npy", allow_pickle=True)
scaler = joblib.load("dog_scaler.pkl")

def predict_emotion(data):
    """
    Predicts the dog's emotion based on input sensor data.

    Parameters:
        data (dict): Contains 'hearRate', 'bodyTemperature', and 'accelMagnitude'.

    Returns:
        dict: Predicted emotion and confidence score.
    """
    # Map received JSON keys to model expected feature names
    formatted_data = {
        "Heart Rate (bpm)": data.get("hearRate"),
        "Body Temperature (°C)": data.get("bodyTemperature"),
        "Tail Wagging (m/s²)": data.get("accelMagnitude")
    }

    # Convert to DataFrame and scale
    df = pd.DataFrame([formatted_data])
    df[df.columns] = scaler.transform(df[df.columns])

    X_input = df.values.reshape((1, 1, df.shape[1]))

    # Make prediction
    y_pred = model.predict(X_input)
    predicted_class = y_pred.argmax(axis=1)[0]
    predicted_emotion = label_encoder_classes[predicted_class]
    confidence_score = float(y_pred[0][predicted_class])  # Convert to float for JSON serialization

    return {"emotion": predicted_emotion, "confidence": confidence_score}
