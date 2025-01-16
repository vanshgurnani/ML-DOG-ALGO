import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

class DogHealthMonitor:
    
    def __init__(self):
        self.model = None
    
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
    
    # Step 3: Classify individual sensor statuses
    def classify_sensor_status(self, temperature, heart_rate, accelerometer_magnitude):
        """
        Classify the status for each sensor.
        """
        # Temperature status
        if temperature < 37.5:
            temp_status = 'Low'
        elif 37.5 <= temperature <= 39.2:
            temp_status = 'Normal'
        else:
            temp_status = 'High'

        # Heart rate status
        if heart_rate < 50:
            heart_rate_status = 'Low'
        elif 50 <= heart_rate <= 140:
            heart_rate_status = 'Normal'
        else:
            heart_rate_status = 'High'

        # Accelerometer status
        if accelerometer_magnitude < 0.5:
            accel_status = 'Low'
        elif 0.5 <= accelerometer_magnitude <= 1.5:
            accel_status = 'Normal'
        else:
            accel_status = 'High'

        return temp_status, heart_rate_status, accel_status
    
    # Step 4: Collect Data and Label it
    def collect_data(self, num_samples=100):
        """
        Collects sensor data and labels it according to predefined rules.
        """
        data = []
        labels = []
        
        # Simulate sensor data collection
        for _ in range(num_samples):
            temperature, heart_rate, accelerometer_magnitude = self.generate_sensor_data()
            
            # Classify overall health status based on the rules
            if 38.3 <= temperature <= 39.2 and 60 <= heart_rate <= 140 and 0.5 <= accelerometer_magnitude <= 1.5:
                health_status = 'Healthy'
            elif temperature > 39.5 or temperature < 38.0 or heart_rate > 150 or heart_rate < 50:
                health_status = 'Unwell'
            elif temperature > 40.0 or temperature < 37.5 or heart_rate > 180 or heart_rate < 40 or accelerometer_magnitude > 2.5:
                health_status = 'Critical'
            else:
                health_status = 'Unwell'
            
            # Store the data and corresponding statuses
            data.append([temperature, heart_rate, accelerometer_magnitude])
            labels.append(health_status)
        
        # Create DataFrame for the collected data
        df = pd.DataFrame(data, columns=['Temperature', 'HeartRate', 'AccelMagnitude'])
        df['HealthStatus'] = labels
        
        return df
    
    # Step 5: Preprocess Data
    def preprocess_data(self, df):
        """
        Preprocess the data without scaling the features.
        """
        # Extract features and labels
        X = df[['Temperature', 'HeartRate', 'AccelMagnitude']].values
        y = df['HealthStatus']
        
        # Convert labels to one-hot encoding
        label_mapping = {'Healthy': 0, 'Unwell': 1, 'Critical': 2}
        y_encoded = to_categorical(y.map(label_mapping))
        
        return X, y_encoded
    
    # Step 6: Train the Predictive Model
    def train_predictive_model(self, X, y):
        """
        Train a Neural Network on the data using TensorFlow.
        """
        # Split data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Build the neural network model
        self.model = Sequential([
            Dense(32, activation='relu', input_shape=(X.shape[1],)),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(y.shape[1], activation='softmax')  # Softmax for multi-class classification
        ])
        
        # Compile the model
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        # Train the model
        history = self.model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))
        
        # Evaluate the model
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print(f'Model accuracy: {accuracy * 100:.2f}%')
    
    # Step 7: Save the Model
    def save_model(self, model_file='dog_health_model.h5'):
        """
        Save the trained model to a file.
        """
        self.model.save(model_file)
        print("Model saved.")

# Example Usage
if __name__ == '__main__':
    # Initialize the Dog Health Monitor
    monitor = DogHealthMonitor()
    
    # Step 3: Collect Data and Preprocess
    df = monitor.collect_data(num_samples=200)  # Collect 200 samples
    X, y_encoded = monitor.preprocess_data(df)
    
    # Display the first few rows of the data
    print(df.head())
    
    # Step 5: Train the Model
    monitor.train_predictive_model(X, y_encoded)
    
    # Step 6: Save the Model
    monitor.save_model()
