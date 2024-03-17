from fastapi import FastAPI

# Create FastAPI instance
app = FastAPI()

# Define an endpoint
@app.get("/")
def read_root():
    return {"Hello": "World"}

# Run the FastAPI application using uvicorn
import uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
