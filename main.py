# main.py

from fastapi import FastAPI

# Create the FastAPI app instance
app = FastAPI()

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
