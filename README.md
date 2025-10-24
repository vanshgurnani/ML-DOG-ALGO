# 🐶🐱 Dog & Cat Emotion Detection API

This project provides a RESTful **Flask API** that predicts the emotional state of **dogs** and **cats** based on physiological parameters — including heart rate, body temperature, and acceleration magnitude.

---

## 🚀 Features

* Predicts emotions for **dogs** and **cats** using separate models.
* Accepts **query parameters** for flexible GET requests.
* Provides **detailed error handling** and input validation.
* Lightweight and easy to deploy locally or on cloud platforms.

---

## 🧩 Project Structure

```
📂 emotion-api/
├── app.py                 # Main Flask application
├── dog_pred.py            # Dog emotion prediction logic
├── cat_pred.py            # Cat emotion prediction logic
├── requirements.txt       # Project dependencies
└── README.md              # Documentation
```

---

## ⚙️ Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/emotion-api.git
   cd emotion-api
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate    # On macOS/Linux
   venv\Scripts\activate       # On Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**

   ```bash
   python app.py
   ```

5. Flask will start the API locally at:

   ```
   http://127.0.0.1:5000/
   ```

---

## 📡 API Endpoints

### **1️⃣ Root Endpoint**

**`GET /`**

**Description:**
Check if the API is running.

**Response:**

```json
{
  "message": "Dog Emotion Detection API is running!"
}
```

---

### **2️⃣ Dog Emotion Prediction**

**`GET /predict`**

**Query Parameters:**

| Parameter         | Type  | Required | Description                               |
| ----------------- | ----- | -------- | ----------------------------------------- |
| `heartRate`       | float | ✅        | Dog’s heart rate                          |
| `bodyTemperature` | float | ✅        | Dog’s body temperature                    |
| `accelMagnitude`  | float | ✅        | Acceleration magnitude from motion sensor |

**Example Request:**

```
GET http://127.0.0.1:5000/predict?heartRate=85&bodyTemperature=38.5&accelMagnitude=1.2
```

**Example Response:**

```json
{
  "emotion": "Happy",
  "confidence": 0.92
}
```

---

### **3️⃣ Cat Emotion Prediction**

**`GET /predict/cat`**

**Query Parameters:**

| Parameter         | Type  | Required | Description                               |
| ----------------- | ----- | -------- | ----------------------------------------- |
| `heartRate`       | float | ✅        | Cat’s heart rate                          |
| `bodyTemperature` | float | ✅        | Cat’s body temperature                    |
| `accelMagnitude`  | float | ✅        | Acceleration magnitude from motion sensor |

**Example Request:**

```
GET http://127.0.0.1:5000/predict/cat?heartRate=120&bodyTemperature=39.0&accelMagnitude=0.9
```

**Example Response:**

```json
{
  "emotion": "Anxious",
  "confidence": 0.87
}
```

---

## 🧠 Model Logic

Both `dog_pred.py` and `cat_pred.py` should export prediction functions:

* `predict_emotion(input_data)` for dogs
* `cat_predict_emotion(input_data)` for cats

Each function must accept a dictionary like:

```python
{
  "heartRate": 85.0,
  "bodyTemperature": 38.5,
  "accelMagnitude": 1.2
}
```

and return a dictionary containing emotion and confidence score.

Example:

```python
def predict_emotion(data):
    return {"emotion": "Happy", "confidence": 0.95}
```

---

## 🧾 Example `requirements.txt`

```txt
Flask==3.0.3
scikit-learn==1.5.0
numpy==1.26.4
pandas==2.2.2
```

---

## 🧪 Testing

Once the app is running, open your browser or use **curl/Postman**:

```bash
curl "http://127.0.0.1:5000/predict?heartRate=90&bodyTemperature=38.6&accelMagnitude=1.5"
```

Expected Output:

```json
{
  "emotion": "Playful",
  "confidence": 0.91
}
```

---

## 💡 Notes

* Ensure that the model files (`dog_pred.py`, `cat_pred.py`) are correctly configured.
* The API returns HTTP 400 if parameters are missing or invalid.
* Debug mode is enabled by default — disable it for production.

---

## 🐾 Author

**Vansh Gurnani**
💻 Passionate about AI, IoT, and smart animal behavior analysis.

---
