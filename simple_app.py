from fastapi import FastAPI
import uvicorn

# Create a very simple FastAPI application
app = FastAPI(title="Simple Test API")

@app.get("/")
def read_root():
    return {"message": "Hello World"}

@app.get("/health")
def health_check():
    return {"status": "ok", "version": "test"}

# This will run when the script is executed directly
if __name__ == "__main__":
    print("Starting simple test server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000) 