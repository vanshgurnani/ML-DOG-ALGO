from flask import Flask, request, jsonify
from dog_pred import predict_emotion

app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"message": "Dog Emotion Detection API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No input data provided"}), 400
        
        result = predict_emotion(data)
        return jsonify(result)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
