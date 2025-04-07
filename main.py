from flask import Flask, request, jsonify
from dog_pred import predict_emotion
from cat_pred import cat_predict_emotion

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"message": "Dog Emotion Detection API is running!"})

@app.route("/predict", methods=["GET"])
def predict():
    try:
        # Extract and validate query parameters
        query_params = request.args.to_dict()
        
        # Debugging: Print received query parameters
        print("Received query params:", query_params)

        # Retrieve parameters with correct names
        heart_rate = request.args.get("heartRate", type=float)
        body_temperature = request.args.get("bodyTemperature", type=float)
        accel_magnitude = request.args.get("accelMagnitude", type=float)

        # Debugging: Print extracted values
        print(f"Extracted - heartRate: {heart_rate}, bodyTemperature: {body_temperature}, accelMagnitude: {accel_magnitude}")

        # Check if all required parameters are provided
        if heart_rate is None or body_temperature is None or accel_magnitude is None:
            return jsonify({
                "error": "Missing or incorrect query parameters",
                "received": query_params
            }), 400

        # Create input dictionary
        input_data = {
            "heartRate": heart_rate,
            "bodyTemperature": body_temperature,
            "accelMagnitude": accel_magnitude
        }

        # Call prediction function
        result = predict_emotion(input_data)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict/cat", methods=["GET"])
def cat_predict():
    try:
        # Extract and validate query parameters
        query_params = request.args.to_dict()
        
        # Debugging: Print received query parameters
        print("Received query params:", query_params)

        # Retrieve parameters with correct names
        heart_rate = request.args.get("heartRate", type=float)
        body_temperature = request.args.get("bodyTemperature", type=float)
        accel_magnitude = request.args.get("accelMagnitude", type=float)

        # Debugging: Print extracted values
        print(f"Extracted - heartRate: {heart_rate}, bodyTemperature: {body_temperature}, accelMagnitude: {accel_magnitude}")

        # Check if all required parameters are provided
        if heart_rate is None or body_temperature is None or accel_magnitude is None:
            return jsonify({
                "error": "Missing or incorrect query parameters",
                "received": query_params
            }), 400

        # Create input dictionary
        input_data = {
            "heartRate": heart_rate,
            "bodyTemperature": body_temperature,
            "accelMagnitude": accel_magnitude
        }

        # Call prediction function
        result = cat_predict_emotion(input_data)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
