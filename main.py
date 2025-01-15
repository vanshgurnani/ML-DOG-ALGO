from fastapi import FastAPI, Query
from pred import DogHealthPredictor

# Create the FastAPI app instance
app = FastAPI()

# Initialize the DogHealthPredictor
predictor = DogHealthPredictor()

# Define a basic GET route
@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}

# Define a new GET route that accepts a query parameter 'name'
@app.get("/greet")
async def greet_person(name: str = None):
    if name:
        return {"message": f"Hello, {name}!"}
    return {"message": "Hello, stranger!"}

# API endpoint for health prediction using query parameters
@app.get("/predict-health")
async def predict_health(
    temperature: float = Query(..., description="Temperature reading of the dog"),
    heart_rate: int = Query(..., description="Heart rate of the dog"),
    accelerometer_x: float = Query(..., description="Accelerometer x-axis reading"),
    accelerometer_y: float = Query(..., description="Accelerometer y-axis reading"),
    accelerometer_z: float = Query(..., description="Accelerometer z-axis reading"),
):
    """
    Predict the health status of a dog based on real-time sensor data.
    """
    # Compute the accelerometer magnitude
    accel_magnitude = predictor.compute_magnitude(accelerometer_x, accelerometer_y, accelerometer_z)
    
    # Predict health status
    health_status = predictor.predict_health_status(temperature, heart_rate, accel_magnitude)
    
    return {"health_status": health_status}
