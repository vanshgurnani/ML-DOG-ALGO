import pickle  # Used to load the saved model and scaler
import numpy as np

class DogHealthPredictor:
    
    def __init__(self, model_file='dog_health_model.pkl', scaler_file='scaler.pkl'):
        try:
            # Load the pre-trained model and scaler
            with open(model_file, 'rb') as model_f:
                self.model = pickle.load(model_f)
            
            with open(scaler_file, 'rb') as scaler_f:
                self.scaler = pickle.load(scaler_f)
        except FileNotFoundError:
            print("Error: Model or scaler file not found. Ensure they are trained and saved correctly.")

    def compute_magnitude(self, x, y, z):
        """
        Compute the magnitude from accelerometer data (x, y, z) without numpy or math.
        """
        return (x**2 + y**2 + z**2) ** 0.5  # Equivalent to math.sqrt(x**2 + y**2 + z**2)
    
    def predict_health_status(self, temperature, heart_rate, accelerometer_magnitude):
        """
        Predict the dog's health status based on real-time sensor data using the trained model.
        """
        try:
            # Ensure input values are valid numbers
            temperature = float(temperature)
            heart_rate = int(heart_rate)
            accelerometer_magnitude = float(accelerometer_magnitude)

            # Create a list with correct input features
            input_data = np.array([[temperature, heart_rate, accelerometer_magnitude]])
            
            # Scale the input data
            input_scaled = self.scaler.transform(input_data)
            
            # Predict health status
            prediction = self.model.predict(input_scaled)
            return prediction[0]
        
        except ValueError:
            return "Invalid input: Ensure all values are numeric."
        except Exception as e:
            return f"Prediction error: {str(e)}"
