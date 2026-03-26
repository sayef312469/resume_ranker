#!/usr/bin/env python3
"""
Startup script for Replit
Serves both FastAPI backend and frontend static files
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Frontend is already handled in main.py

if __name__ == "__main__":
    # Change to backend directory
    os.chdir("backend")
    
    # Run on 0.0.0.0:8000 so Replit can access it
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
