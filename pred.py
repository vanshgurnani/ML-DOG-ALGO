from tensorflow.keras.models import load_model  # Use TensorFlow's built-in model loading function

class DogHealthPredictor:
    
    def __init__(self, model_file='dog_health_model.h5'):
        # Load the pre-trained model from file
        self.model = load_model(model_file)  # Load the model using Keras' load_model method

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
        
        # Predict health status (probabilities)
        prediction = self.model.predict(input_data)  # Use the model to make a prediction
        probability = prediction[0]  # Get the prediction probabilities for each class
        
        # Assume we have two classes: Healthy (index 0) and Unhealthy (index 1)
        healthy_prob = probability[0]
        unhealthy_prob = probability[1]
        
        # Determine health status based on the probabilities
        if unhealthy_prob > 0.8:  # Threshold for critical condition
            return "Critical"
        elif unhealthy_prob > 0.5:  # Threshold for unwell condition
            return "Unwell"
        elif healthy_prob > 0.8:  # Threshold for healthy condition
            return "Healthy"
        else:
            return "Well"  # Default case if not clearly healthy or unhealthy

# Example Usage
if __name__ == '__main__':
    # Initialize the Dog Health Predictor
    predictor = DogHealthPredictor()
    
    # Example input data: Temperature, Heart Rate, Accelerometer Magnitude
    temperature = 38.5  # Example temperature in Celsius
    heart_rate = 75  # Example heart rate in beats per minute
    accel_magnitude = 1.2  # Example accelerometer magnitude
    
    # Predict health status
    health_status = predictor.predict_health_status(temperature, heart_rate, accel_magnitude)
    print(f"Predicted Health Status: {health_status}")
