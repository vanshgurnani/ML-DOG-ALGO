# main.py

from fastapi import FastAPI

# Create the FastAPI app instance
app = FastAPI()

# Define a basic GET route
@app.get("/")
async def read_root():
    return {"message": "Hello, World!"}
