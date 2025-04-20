import os
import sys

print("Python version:", sys.version)
print("Current directory:", os.getcwd())
print("Directory contents:", os.listdir("."))
print("Python path:", sys.path)

# Try to import app components
try:
    import app
    print("✅ Successfully imported app package")
    
    try:
        from app.core.config import settings
        print("✅ Successfully imported settings")
        print("App environment:", settings.APP_ENV)
        print("Debug mode:", settings.DEBUG)
    except Exception as e:
        print("❌ Error importing settings:", str(e))
    
    try:
        from app.services.reddit_client import reddit_client
        print("✅ Successfully imported reddit_client")
        print("Reddit client authenticated:", reddit_client.authenticated)
    except Exception as e:
        print("❌ Error importing reddit_client:", str(e))
    
    try:
        import app.main
        print("✅ Successfully imported app.main")
    except Exception as e:
        print("❌ Error importing app.main:", str(e))
        
    # Try to create a test app
    try:
        from fastapi import FastAPI
        test_app = FastAPI()
        
        @test_app.get("/test")
        def test_endpoint():
            return {"status": "ok"}
        
        print("✅ Successfully created test FastAPI app")
        
        # Try running the app with uvicorn programmatically
        print("\nAttempting to start test server...")
        import uvicorn
        
        if __name__ == "__main__":
            # This won't run when imported but will run when the script is executed
            print("Starting uvicorn server on http://127.0.0.1:8001/test")
            uvicorn.run(test_app, host="127.0.0.1", port=8001)
    except Exception as e:
        print("❌ Error setting up test app:", str(e))
    
except Exception as e:
    print("❌ Error importing app package:", str(e)) 