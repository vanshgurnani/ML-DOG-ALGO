from tensorflow.keras.models import load_model
import json

# Load the saved model
model = load_model("dog_emotion_model.h5")

# Convert the model architecture to JSON
model_json = model.to_json()

# Save JSON model to a file
with open("dog_emotion_model.json", "w") as json_file:
    json_file.write(model_json)

print("Model architecture saved as JSON successfully!")
