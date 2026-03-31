import uvicorn
import sys
import os

# Add the parent directory to the path so it can find your real app.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import app

def main():
    """Standard entry point required by OpenEnv."""
    uvicorn.run("app:app", host="0.0.0.0", port=8000)

if __name__ == '__main__':
    main()