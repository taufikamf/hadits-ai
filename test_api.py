#!/usr/bin/env python3
"""
Script to test the API endpoint.
"""
import requests
import json
import time

def test_api():
    """Test the API endpoint"""
    
    # Start the server in background
    print("Starting server...")
    import subprocess
    import os
    
    # Start server in background
    server_process = subprocess.Popen([
        "bash", "-c", 
        "source venv-py311/bin/activate && python main.py"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for server to start
    print("Waiting for server to start...")
    time.sleep(5)
    
    try:
        # Test health endpoint
        print("Testing health endpoint...")
        response = requests.get("http://localhost:8000/health")
        print(f"Health status: {response.status_code}")
        print(f"Response: {response.json()}")
        
        # Test ask endpoint
        print("\nTesting ask endpoint...")
        test_queries = [
            "apa itu niat",
            "bagaimana cara berwudhu",
            "apa hukum shalat"
        ]
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            response = requests.get(f"http://localhost:8000/ask?q={query}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"Status: {response.status_code}")
                print(f"Answer: {result.get('answer', 'No answer')[:200]}...")
                print(f"Retrieved {len(result.get('retrieved_documents', []))} documents")
                
                # Show first retrieved document
                if result.get('retrieved_documents'):
                    first_doc = result['retrieved_documents'][0]
                    print(f"Top result: {first_doc.get('kitab', 'Unknown')} - {first_doc.get('terjemah', '')[:100]}...")
            else:
                print(f"Error: {response.status_code}")
                print(f"Response: {response.text}")
        
    except Exception as e:
        print(f"Error testing API: {e}")
    
    finally:
        # Stop server
        print("\nStopping server...")
        server_process.terminate()
        server_process.wait()

if __name__ == "__main__":
    test_api()