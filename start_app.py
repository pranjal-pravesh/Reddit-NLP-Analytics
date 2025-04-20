#!/usr/bin/env python
"""
Reddit Analysis Application Launcher

This script provides an easy way to start the Reddit Analysis application
with configurable options for host, port, and development mode.
"""

import os
import sys
import argparse
import subprocess
import webbrowser
import re
from pathlib import Path

def fix_env_file():
    """Check and fix common issues in the .env file"""
    env_file = Path(".env")
    if not env_file.exists():
        print("No .env file found. Creating default configuration...")
        create_default_env_file()
        return
    
    print("Checking .env file for configuration issues...")
    
    # Read the file line by line
    with open(env_file, "r") as f:
        lines = f.readlines()
    
    # Fix each line
    fixed_lines = []
    for line in lines:
        # Special case for CACHE_TTL, which is causing the validation error
        if line.strip().startswith("CACHE_TTL="):
            # Extract just the numeric value
            match = re.search(r"CACHE_TTL=(\d+)", line)
            if match:
                value = match.group(1)
                fixed_lines.append(f"CACHE_TTL={value}\n")
                print(f"Fixed CACHE_TTL setting: {value}")
            else:
                # If we can't extract a value, set a default
                fixed_lines.append("CACHE_TTL=3600\n")
                print("Reset CACHE_TTL to default value: 3600")
        else:
            fixed_lines.append(line)
    
    # Write the fixed content back
    with open(env_file, "w") as f:
        f.writelines(fixed_lines)
    
    print("Configuration file fixed.")
    
    # Additionally set the environment variable directly
    os.environ["CACHE_TTL"] = "3600"

def create_default_env_file():
    """Create a default .env file with minimal required settings"""
    content = """# Application settings
APP_ENV=development
DEBUG=true
LOG_LEVEL=INFO
# Cache time-to-live in seconds
CACHE_TTL=3600

# Reddit API Credentials
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=
REDDIT_USER_AGENT=RedditAnalysisApp/1.0

# LLM settings (optional)
ENABLE_LLM_INTEGRATION=false

# Cache settings
# CACHE_TYPE=memory
"""
    with open(".env", "w") as f:
        f.write(content)
    print("Default .env file created")

def start_application(python_executable, host, port, dev_mode, open_browser):
    """Start the FastAPI application using uvicorn"""
    try:
        # Set environment variables
        os.environ["APP_ENV"] = "development" if dev_mode else "production"
        os.environ["DEBUG"] = "true" if dev_mode else "false"
        
        # This is critical for avoiding the pydantic validation error
        os.environ["CACHE_TTL"] = "3600"
        
        print(f"Starting Reddit Analysis on http://{host}:{port}")
        print(f"Mode: {'Development' if dev_mode else 'Production'}")
        
        # Debug raw env vars
        for key, value in os.environ.items():
            if key in ["APP_ENV", "DEBUG", "CACHE_TTL", "LOG_LEVEL"]:
                print(f"Raw environment variable: {key}={repr(value)}")
        
        # Start application
        cmd = [
            python_executable, 
            "-m", "uvicorn", 
            "app.main:app", 
            "--host", host, 
            "--port", str(port),
            "--reload" if dev_mode else ""
        ]
        cmd = [item for item in cmd if item]  # Remove empty strings
        
        # Open browser if requested
        if open_browser:
            webbrowser.open(f"http://{host}:{port}")
        
        # Run the server
        subprocess.run(cmd, env=os.environ)
        
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except Exception as e:
        print(f"Error starting application: {e}")
        sys.exit(1)

def setup_environment():
    """Set up the environment variables and check dependencies"""
    # Ensure we're in the right directory
    os.chdir(Path(__file__).parent)
    
    # Check if virtual environment exists
    venv_dir = "venv"
    if sys.platform == "win32":
        python_executable = os.path.join(venv_dir, "Scripts", "python.exe")
        pip_executable = os.path.join(venv_dir, "Scripts", "pip.exe")
    else:
        python_executable = os.path.join(venv_dir, "bin", "python")
        pip_executable = os.path.join(venv_dir, "bin", "pip")
    
    if not os.path.exists(python_executable):
        print("Virtual environment not found. Setting up environment...")
        create_virtual_environment(venv_dir)
        install_dependencies(pip_executable)
    
    return python_executable

def create_virtual_environment(venv_dir):
    """Create a virtual environment"""
    print("Creating virtual environment...")
    try:
        import venv
        venv.create(venv_dir, with_pip=True)
        print(f"Virtual environment created at {venv_dir}")
    except Exception as e:
        print(f"Error creating virtual environment: {e}")
        sys.exit(1)

def install_dependencies(pip_executable):
    """Install required dependencies"""
    print("Installing dependencies...")
    try:
        subprocess.run([pip_executable, "install", "-r", "requirements.txt"], check=True)
        print("Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        sys.exit(1)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Start the Reddit Analysis application")
    parser.add_argument("--host", default="127.0.0.1", help="Host address to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default: 8000)")
    parser.add_argument("--dev", action="store_true", help="Run in development mode with auto-reload")
    parser.add_argument("--open", action="store_true", help="Open browser after starting")
    
    args = parser.parse_args()
    
    # Setup environment
    python_executable = setup_environment()
    
    # Explicitly fix environment variables
    fix_env_file()
    
    # Start application
    start_application(
        python_executable=python_executable,
        host=args.host,
        port=args.port,
        dev_mode=args.dev,
        open_browser=args.open
    )

if __name__ == "__main__":
    main() 