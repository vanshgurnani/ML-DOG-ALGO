import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv("synthetic_data.csv")

# Check for missing values
df.dropna(inplace=True)

# Compute acceleration magnitude
df["Accel_Magnitude"] = np.sqrt(df["Accel_X"]**2 + df["Accel_Y"]**2 + df["Accel_Z"]**2)

# Remove original X, Y, Z columns
df.drop(["Accel_X", "Accel_Y", "Accel_Z"], axis=1, inplace=True)

# Define classification function (same as before)
def classify_condition(hr, temp, accel_mag):
    if hr > 220 and temp > 39.17 and accel_mag > 2:
        return "Excited / Stressed / Anxious"
    elif hr > 220 and 38.05 <= temp <= 39.17 and accel_mag > 1.5:
        return "Excited / Happy"
    elif hr > 220 and temp > 39.17 and accel_mag < 0.5:
        return "Fever / Pain / Infection"
    elif hr > 220 and temp < 38.05 and accel_mag < 0.2:
        return "Shock / Heart Condition / Serious Illness"
    elif 140 <= hr <= 220 and 38.05 <= temp <= 39.17 and 0.5 <= accel_mag <= 1.5:
        return "Healthy & Relaxed"
    elif 140 <= hr <= 220 and temp > 39.17 and accel_mag > 1.5:
        return "Overheating / Mild Fever"
    elif hr < 140 and temp < 38.05 and accel_mag < 0.1:
        return "Hypothermia / Severe Illness"
    elif hr < 140 and temp > 39.17 and accel_mag < 0.2:
        return "Serious Infection"
    else:
        return "Unknown"

# Apply classification function
df["Condition"] = df.apply(lambda row: classify_condition(row["Heart_Rate"], row["Temperature"], row["Accel_Magnitude"]), axis=1)

# Remove excessive "Unknown" labels
df = df[df["Condition"] != "Unknown"]

# Prepare dataset
X = df[["Heart_Rate", "Temperature", "Accel_Magnitude"]]  # Features
y = df["Condition"]  # Target labels

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")  # Print model accuracy

# Save model and scaler
joblib.dump(model, "cat_health_model.pkl")
joblib.dump(scaler, "cat_scaler.pkl")  # Ensure correct scaler is saved

print("Model training complete. Saved as 'cat_health_model.pkl'.")

print(df["Condition"].value_counts())


