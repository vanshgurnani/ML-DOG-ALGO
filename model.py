import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib  # Used to save the trained model

class DogHealthMonitor:
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
    
    # Step 1: Compute accelerometer magnitude
    def compute_magnitude(self, x, y, z):
        """
        Compute the magnitude from accelerometer data (x, y, z).
        """
        return np.sqrt(x**2 + y**2 + z**2)
    
    # Step 2: Simulate the generation of sensor data
    def generate_sensor_data(self):
        """
        Simulate random sensor data for temperature, heart rate, and accelerometer readings.
        """
        temperature = np.random.uniform(37.0, 41.0)  # Random temperature between 37 and 41°C
        heart_rate = np.random.randint(40, 180)  # Random heart rate between 40 and 180 BPM
        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1, 1)
        z = np.random.uniform(-1, 1)
        accelerometer_magnitude = self.compute_magnitude(x, y, z)
        return temperature, heart_rate, accelerometer_magnitude
    
    # Step 3: Collect Data and Label it
    def collect_data(self, num_samples=100):
        """
        Collects sensor data and labels it according to predefined rules.
        """
        data = []
        labels = []
        
        # Simulate sensor data collection
        for _ in range(num_samples):
            temperature, heart_rate, accelerometer_magnitude = self.generate_sensor_data()
            
            # Classify health status based on the rules
            if 38.3 <= temperature <= 39.2 and 60 <= heart_rate <= 140 and 0.5 <= accelerometer_magnitude <= 1.5:
                health_status = 'Healthy'
            elif temperature > 39.5 or temperature < 38.0 or heart_rate > 150 or heart_rate < 50:
                health_status = 'Unwell'
            elif temperature > 40.0 or temperature < 37.5 or heart_rate > 180 or heart_rate < 40 or accelerometer_magnitude > 2.5:
                health_status = 'Critical'
            else:
                health_status = 'Unwell'
            
            data.append([temperature, heart_rate, accelerometer_magnitude])
            labels.append(health_status)
        
        # Create DataFrame for the collected data
        df = pd.DataFrame(data, columns=['Temperature', 'HeartRate', 'AccelMagnitude'])
        df['HealthStatus'] = labels
        return df
    
    # Step 4: Preprocess Data
    def preprocess_data(self, df):
        """
        Preprocess the data by normalizing the features.
        """
        # Extract features and labels
        X = df[['Temperature', 'HeartRate', 'AccelMagnitude']]
        y = df['HealthStatus']
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    # Step 5: Train the Predictive Model
    def train_predictive_model(self, X, y):
        """
        Train a Random Forest Classifier on the data.
        """
        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the Random Forest Classifier
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Test the model
        accuracy = self.model.score(X_test, y_test)
        print(f'Model accuracy: {accuracy * 100:.2f}%')
        
    # Step 6: Save the Model
    def save_model(self, model_file='dog_health_model.pkl', scaler_file='scaler.pkl'):
        """
        Save the trained model and scaler to files.
        """
        joblib.dump(self.model, model_file)
        joblib.dump(self.scaler, scaler_file)
        print("Model and scaler saved.")

# Example Usage
if __name__ == '__main__':
    # Initialize the Dog Health Monitor
    monitor = DogHealthMonitor()
    
    # Step 3: Collect Data and Preprocess
    df = monitor.collect_data(num_samples=200)  # Collect 200 samples
    X_scaled, y = monitor.preprocess_data(df)
    
    # Step 5: Train the Model
    monitor.train_predictive_model(X_scaled, y)
    
    # Step 6: Save the Model
    monitor.save_model()
