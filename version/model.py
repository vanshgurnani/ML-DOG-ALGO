import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import pickle

class DogHealthMonitor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()

    def compute_magnitude(self, x, y, z):
        return np.sqrt(x**2 + y**2 + z**2)

    def classify_health_status(self, heart_rate, temperature, accel_magnitude):
        """
        Classify the dog's health status based on predefined conditions from the table.
        """
        if heart_rate > 140 and temperature > 104 and accel_magnitude < 0.5:
            return "Heatstroke"
        elif heart_rate < 50 and temperature < 98 and accel_magnitude < 0.5:
            return "Hypothermia"
        elif heart_rate < 50 and accel_magnitude < 0.1:
            return "Shock"
        elif heart_rate > 120 and temperature > 103:
            return "Fever"
        elif heart_rate > 120 and accel_magnitude > 1.5:
            return "Excitement/Anxiety"
        elif heart_rate > 120 and accel_magnitude < 0.5:
            return "Pain/Injury"
        elif heart_rate > 90 and temperature > 101 and accel_magnitude > 1.0:
            return "Active Play"
        elif heart_rate < 60 and accel_magnitude < 0.5:
            return "Lethargy"
        elif heart_rate > 140 and accel_magnitude > 1.5:
            return "Hyperthyroidism"
        elif heart_rate < 40 or heart_rate > 150:
            return "Heart Disease/Arrhythmia"
        elif heart_rate > 100 and temperature > 102:
            return "Exhaustion"
        else:
            return "Normal"

    def collect_data(self, num_samples=100):
        data = []
        labels = []
        
        for _ in range(num_samples):
            temperature = np.random.uniform(90, 105)
            heart_rate = np.random.randint(25, 180)
            x, y, z = np.random.uniform(-1, 1, 3)
            accel_magnitude = self.compute_magnitude(x, y, z)
            
            health_status = self.classify_health_status(heart_rate, temperature, accel_magnitude)
            data.append([temperature, heart_rate, accel_magnitude])
            labels.append(health_status)
        
        df = pd.DataFrame(data, columns=['Temperature', 'HeartRate', 'AccelMagnitude'])
        df['HealthStatus'] = labels
        df.to_csv('dog_health_data.csv', index=False)
        return df

    def preprocess_data(self, df):
        X = df[['Temperature', 'HeartRate', 'AccelMagnitude']]
        y = df['HealthStatus']
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, y

    def train_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Model Evaluation
        y_pred = self.model.predict(X_test)
        accuracy = self.model.score(X_test, y_test)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        
        print(f'Model accuracy: {accuracy * 100:.2f}%')
        print("Confusion Matrix:")
        print(conf_matrix)
        print("Classification Report:")
        print(class_report)

    def save_model(self, model_file='dog_health_model.pkl', scaler_file='scaler.pkl'):
        with open(model_file, 'wb') as mf:
            pickle.dump(self.model, mf)
        with open(scaler_file, 'wb') as sf:
            pickle.dump(self.scaler, sf)
        print("Model and scaler saved.")

if __name__ == '__main__':
    monitor = DogHealthMonitor()
    df = monitor.collect_data(num_samples=5000)
    X_scaled, y = monitor.preprocess_data(df)
    print(df.head())
    monitor.train_model(X_scaled, y)
    monitor.save_model()