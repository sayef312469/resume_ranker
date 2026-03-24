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

# Import the existing FastAPI app from main.py
from main import app

# Serve frontend static files
frontend_dir = os.path.join(os.path.dirname(__file__), 'frontend')
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="static")

if __name__ == "__main__":
    # Run on 0.0.0.0:8000 so Replit can access it
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        cwd="backend"
    )
