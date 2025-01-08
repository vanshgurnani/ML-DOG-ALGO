import numpy as np
import joblib  # Used to load the saved model and scaler
import pandas as pd  # To create DataFrame with correct column names

class DogHealthPredictor:
    
    def __init__(self, model_file='dog_health_model.pkl', scaler_file='scaler.pkl'):
        # Load the pre-trained model and scaler from files
        self.model = joblib.load(model_file)
        self.scaler = joblib.load(scaler_file)
    
    # Step 1: Compute accelerometer magnitude
    def compute_magnitude(self, x, y, z):
        """
        Compute the magnitude from accelerometer data (x, y, z).
        """
        return np.sqrt(x**2 + y**2 + z**2)
    
    # Step 2: Predict health status based on sensor data
    def predict_health_status(self, temperature, heart_rate, accelerometer_magnitude):
        """
        Predict the dog's health status based on real-time sensor data using the trained model.
        """
        # Create a DataFrame with correct column names (to match the training data)
        input_data = pd.DataFrame([[temperature, heart_rate, accelerometer_magnitude]], 
                                  columns=['Temperature', 'HeartRate', 'AccelMagnitude'])
        
        # Preprocess the input data (scale it)
        input_scaled = self.scaler.transform(input_data)  # Scale the input data using the existing scaler
        
        # Predict health status
        prediction = self.model.predict(input_scaled)
        return prediction[0]

# Example Usage
if __name__ == '__main__':
    # Initialize the Dog Health Predictor with the trained model and scaler
    predictor = DogHealthPredictor()
    
    # Simulated real-time sensor data (can be replaced with actual sensor readings)
    temperature = 29.0  # Simulated real-time temperature reading
    heart_rate = 120    # Simulated real-time heart rate
    accelerometer_magnitude = 1.0  # Simulated real-time accelerometer magnitude
    
    # Get the predicted health status
    health_status = predictor.predict_health_status(temperature, heart_rate, accelerometer_magnitude)
    print(f'Dog Health Status: {health_status}')
