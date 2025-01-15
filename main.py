from flask import Flask, request, jsonify
from pred import DogHealthPredictor

# Create the Flask app instance
app = Flask(__name__)

# Initialize the DogHealthPredictor
predictor = DogHealthPredictor()

# Define a basic route
@app.route("/", methods=["GET"])
def read_root():
    return jsonify({"message": "Hello, World!"})

# Define a new route that accepts a query parameter 'name'
@app.route("/greet", methods=["GET"])
def greet_person():
    name = request.args.get("name")
    if name:
        return jsonify({"message": f"Hello, {name}!"})
    return jsonify({"message": "Hello, stranger!"})

# API endpoint for health prediction using query parameters
@app.route("/predict-health", methods=["GET"])
def predict_health():
    # Extract query parameters
    temperature = float(request.args.get("temperature"))
    heart_rate = int(request.args.get("heart_rate"))
    accelerometer_x = float(request.args.get("accelerometer_x"))
    accelerometer_y = float(request.args.get("accelerometer_y"))
    accelerometer_z = float(request.args.get("accelerometer_z"))

    # Compute the accelerometer magnitude
    accel_magnitude = predictor.compute_magnitude(accelerometer_x, accelerometer_y, accelerometer_z)
    
    # Predict health status
    health_status = predictor.predict_health_status(temperature, heart_rate, accel_magnitude)
    
    return jsonify({"health_status": health_status})

if __name__ == "__main__":
    app.run(debug=True)
