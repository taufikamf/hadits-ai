#!/usr/bin/env python3
"""
Simple Test Server - Fixed V1
============================

Minimal test server to verify Flask and routing work correctly.
"""

from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return jsonify({
        "success": True,
        "message": "Simple test server is working",
        "service": "Hadith AI API - Test Mode"
    })

@app.route('/health')
def health():
    return jsonify({
        "success": True,
        "status": "healthy", 
        "message": "Test health endpoint is working"
    })

@app.route('/test')
def test():
    return jsonify({
        "success": True,
        "message": "Test endpoint is working",
        "routes": ["/", "/health", "/test"]
    })

if __name__ == "__main__":
    print("🧪 Starting simple test server...")
    print("Test URLs:")
    print("  http://localhost:5000/")
    print("  http://localhost:5000/health") 
    print("  http://localhost:5000/test")
    
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
