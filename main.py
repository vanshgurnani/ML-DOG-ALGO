from flask import Flask, request, jsonify
import joblib
import math

# Initialize Flask app
app = Flask(__name__)

# # Predictor Class (from pred.py)
# class DogHealthPredictor:
#     def __init__(self, model_file='dog_health_model.pkl', scaler_file='scaler.pkl'):
#         self.model = joblib.load(model_file)
#         self.scaler = joblib.load(scaler_file)
    
#     def compute_magnitude(self, x, y, z):
#         # Calculate magnitude without numpy
#         return math.sqrt(x**2 + y**2 + z**2)
    
#     def predict_health_status(self, temperature, heart_rate, accelerometer_magnitude):
#         # Prepare input data as a list of features (no need for pandas)
#         input_data = [[temperature, heart_rate, accelerometer_magnitude]]
        
#         # Scale the input data (still using scaler)
#         input_scaled = self.scaler.transform(input_data)
        
#         # Predict health status
#         prediction = self.model.predict(input_scaled)
#         return prediction[0]

# # Initialize the DogHealthPredictor
# predictor = DogHealthPredictor()

# Define a basic route
@app.route('/')
def read_root():
    return jsonify({"message": "Hello, World!"})

# Define a new route that accepts a query parameter 'name'
@app.route('/greet', methods=['GET'])
def greet_person():
    name = request.args.get('name')
    if name:
        return jsonify({"message": f"Hello, {name}!"})
    return jsonify({"message": "Hello, stranger!"})

# # API endpoint for health prediction using query parameters
# @app.route('/predict-health', methods=['GET'])
# def predict_health():
#     """
#     Predict the health status of a dog based on real-time sensor data.
#     """
#     # Extract query parameters
#     temperature = float(request.args.get('temperature'))
#     heart_rate = int(request.args.get('heart_rate'))
#     accelerometer_x = float(request.args.get('accelerometer_x'))
#     accelerometer_y = float(request.args.get('accelerometer_y'))
#     accelerometer_z = float(request.args.get('accelerometer_z'))
    
#     # Compute the accelerometer magnitude
#     accel_magnitude = predictor.compute_magnitude(accelerometer_x, accelerometer_y, accelerometer_z)
    
#     # Predict health status
#     health_status = predictor.predict_health_status(temperature, heart_rate, accel_magnitude)
    
#     return jsonify({"health_status": health_status})

if __name__ == '__main__':
    app.run(debug=True)
