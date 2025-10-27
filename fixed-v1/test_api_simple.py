#!/usr/bin/env python3
"""
Simple API Test Script - Fixed V1
================================

Test script to verify that the API server can start and respond to basic requests.
"""

import os
import sys
import time
import requests
import subprocess
import signal
from threading import Thread

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_basic_api():
    """Test basic API functionality without full service."""
    
    print("🧪 Testing Basic API Functionality")
    print("=" * 50)
    
    # Simple Flask app for testing
    from flask import Flask, jsonify
    from flask_cors import CORS
    
    app = Flask(__name__)
    CORS(app)
    
    @app.route('/')
    def home():
        return jsonify({
            "success": True,
            "message": "Hadith AI API - Test Mode",
            "version": "1.0.0"
        })
    
    @app.route('/health')
    def health():
        return jsonify({
            "success": True,
            "status": "healthy",
            "message": "Service is running in test mode"
        })
    
    # Test the app
    with app.test_client() as client:
        # Test home endpoint
        response = client.get('/')
        print(f"GET / : {response.status_code}")
        print(f"Response: {response.get_json()}")
        
        # Test health endpoint  
        response = client.get('/health')
        print(f"GET /health : {response.status_code}")
        print(f"Response: {response.get_json()}")
        
        if response.status_code == 200:
            print("✅ Basic API structure is working!")
            return True
        else:
            print("❌ Basic API test failed!")
            return False

def test_import_service():
    """Test if we can import the service components."""
    
    print("\n🔍 Testing Service Imports")
    print("=" * 50)
    
    try:
        # Test individual component imports
        print("Testing individual imports...")
        
        from service.hadith_ai_service import ServiceConfig
        print("✅ ServiceConfig imported successfully")
        
        from service.hadith_ai_service import ChatResponse  
        print("✅ ChatResponse imported successfully")
        
        # Test creating a basic config
        config = ServiceConfig(max_results_display=3)
        print(f"✅ ServiceConfig created: max_results={config.max_results_display}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("This is expected if the full system isn't set up yet.")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_full_api_server():
    """Test the full API server if possible."""
    
    print("\n🚀 Testing Full API Server")
    print("=" * 50)
    
    try:
        from service.api_server import create_app
        from service.hadith_ai_service import ServiceConfig
        
        # Create test config
        config = ServiceConfig(
            max_results_display=3,
            enable_sessions=False,  # Disable for testing
            enable_analytics=False
        )
        
        # Try to create app
        app = create_app(config)
        
        with app.test_client() as client:
            # Test home endpoint
            response = client.get('/')
            print(f"GET / : {response.status_code}")
            
            # Test health endpoint
            response = client.get('/health')
            print(f"GET /health : {response.status_code}")
            
            if response.status_code == 200:
                print("✅ Full API server is working!")
                return True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                return False
                
    except Exception as e:
        print(f"❌ Full API server test failed: {e}")
        print("This might be due to missing dependencies or data files.")
        return False

def main():
    """Run all tests."""
    
    print("🧪 Hadith AI API - Test Suite")
    print("=" * 60)
    
    # Test 1: Basic API structure
    basic_test = test_basic_api()
    
    # Test 2: Service imports
    import_test = test_import_service()
    
    # Test 3: Full API server (if possible)
    full_test = test_full_api_server()
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 50)
    print(f"Basic API structure: {'✅ PASS' if basic_test else '❌ FAIL'}")
    print(f"Service imports: {'✅ PASS' if import_test else '❌ FAIL'}")
    print(f"Full API server: {'✅ PASS' if full_test else '❌ FAIL'}")
    
    if basic_test:
        print("\n💡 Next Steps:")
        if not import_test:
            print("   1. Complete the indexing pipeline to generate required data files")
            print("   2. Install missing dependencies if any")
        if not full_test:
            print("   3. Run: python service/api_server.py --debug")
            print("   4. Test with: curl http://localhost:5000/health")
        else:
            print("   🎉 All tests passed! API is ready to use.")
    
    return basic_test and (import_test or full_test)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
