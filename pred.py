import pickle  # Used to load the saved model and scaler

class DogHealthPredictor:
    
    def __init__(self, model_file='dog_health_model.pkl', scaler_file='scaler.pkl'):
        # Load the pre-trained model and scaler from files using pickle
        with open(model_file, 'rb') as model_f:
            self.model = pickle.load(model_f)
        
        with open(scaler_file, 'rb') as scaler_f:
            self.scaler = pickle.load(scaler_f)

    # Step 1: Compute accelerometer magnitude without using numpy or math
    def compute_magnitude(self, x, y, z):
        """
        Compute the magnitude from accelerometer data (x, y, z) without numpy or math.
        """
        return (x**2 + y**2 + z**2) ** 0.5  # Equivalent to math.sqrt(x**2 + y**2 + z**2)
    
    # Step 2: Predict health status based on sensor data
    def predict_health_status(self, temperature, heart_rate, accelerometer_magnitude):
        """
        Predict the dog's health status based on real-time sensor data using the trained model.
        """
        # Create a list with the correct columns instead of using pandas DataFrame
        input_data = [[temperature, heart_rate, accelerometer_magnitude]]
        
        # Preprocess the input data (scale it)
        input_scaled = self.scaler.transform(input_data)  # Scale the input data using the existing scaler
        
        # Predict health status
        prediction = self.model.predict(input_scaled)
        return prediction[0]
