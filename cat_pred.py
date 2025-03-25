import joblib
import numpy as np
import pandas as pd

# Load trained model and scaler
model = joblib.load("cat_health_model.pkl")
scaler = joblib.load("cat_scaler.pkl")

# Function to predict condition
def predict_condition(heart_rate, temperature, accel_x, accel_y, accel_z):
    # Compute acceleration magnitude
    accel_magnitude = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
    
    print("accel_magnitude: ", accel_magnitude)
    
    # Convert input to DataFrame with correct feature names
    input_data = pd.DataFrame([[heart_rate, temperature, accel_magnitude]],
                              columns=["Heart_Rate", "Temperature", "Accel_Magnitude"])
    
    # Normalize input using the same scaler
    input_scaled = scaler.transform(input_data)

    # Predict condition
    prediction = model.predict(input_scaled)[0]
    return prediction

# Example usage
if __name__ == "__main__":
    # Sample input
    heart_rate = 120
    temperature = 40  # Celsius
    accel_x = 0.1
    accel_y = 0.1
    accel_z = 0.1

    predicted_condition = predict_condition(heart_rate, temperature, accel_x, accel_y, accel_z)
    print(f"Predicted Condition: {predicted_condition}")
