from flask import Flask, request, jsonify
from dog_pred import predict_emotion

app = Flask(__name__)

@app.route("/", methods=["GET"])
def read_root():
    return jsonify({"message": "Hello, World!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Validate required fields
        required_fields = ["hearRate", "bodyTemperature", "accelMagnitude"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        prediction_result = predict_emotion(data)
        return jsonify(prediction_result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
