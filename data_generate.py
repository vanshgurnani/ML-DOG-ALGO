import random
import math
import csv


def generate_synthetic_data():
    temp = 39.5  # Initial fever temp
    hr = 170  # Initial heart rate
    base_accel = (-0.5, 0.5)  # Movement range for sick cat
    data_limit = 5000  # Limit the number of records
    count = 0

    # Open CSV file in append mode
    with open("synthetic_data.csv", "a", newline="") as file:
        writer = csv.writer(file)
        # Write header if file is empty
        if file.tell() == 0:
            writer.writerow(["Accel_X", "Accel_Y", "Accel_Z", "Magnitude", "Temperature", "Heart_Rate"])

        while count < data_limit:
            # Temperature fluctuates slightly
            temp += random.uniform(-0.2, 0.2)
            temp = max(39.0, min(temp, 41.0))  # Keeping it in fever range

            # Heart Rate is correlated with temperature
            hr = 160 + (temp - 39) * 20 + random.randint(-5, 5)
            hr = max(160, min(hr, 220))  # Keeping it within reasonable bounds

            # Movement decreases as fever rises
            movement_factor = max(0.1, 1.5 - (temp - 39) * 0.5)  # Less movement with higher fever
            x = random.uniform(base_accel[0] * movement_factor, base_accel[1] * movement_factor)
            y = random.uniform(base_accel[0] * movement_factor, base_accel[1] * movement_factor)
            z = -9.81 + random.uniform(-0.5, 0.5) * movement_factor  # Slight variations in gravity axis

            # Magnitude of acceleration
            mag = math.sqrt(x ** 2 + y ** 2 + z ** 2)

            # Print synthetic data
            print(f"Accel: ({x:.2f}, {y:.2f}, {z:.2f}) | Mag: {mag:.2f} | Temp: {temp:.2f}°C | HR: {hr} BPM")
            
            # Write data to CSV
            writer.writerow([round(x, 2), round(y, 2), round(z, 2), round(mag, 2), round(temp, 2), hr])
            file.flush()  # Ensure data is written immediately
            
            count += 1

    print("Data generation complete. 5000 records stored.")


# Run the data generator
generate_synthetic_data()
