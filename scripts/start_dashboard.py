"""
Launch script for the CDF Real-LLM Dashboard.
Run from the project root:  python scripts/start_dashboard.py
"""
import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn

if __name__ == "__main__":
    print("Starting CDF Real-LLM Dashboard at http://127.0.0.1:8000")
    print("Press Ctrl+C to stop.")
    uvicorn.run(
        "cdf.real.api:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
