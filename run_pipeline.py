#!/usr/bin/env python3
"""
Script to run the entire Hadits-AI pipeline automatically.
"""
import subprocess
import sys
import time
import requests
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, description):
    """Run a command and handle errors"""
    logger.info(f"Running: {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.info(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False

def check_file_exists(file_path, description):
    """Check if a file exists"""
    if Path(file_path).exists():
        logger.info(f"✅ {description} exists")
        return True
    else:
        logger.error(f"❌ {description} not found: {file_path}")
        return False

def test_api():
    """Test the API endpoints"""
    logger.info("Testing API endpoints...")
    
    try:
        # Test health endpoint
        response = requests.get("http://localhost:8000/api/v1/health", timeout=10)
        if response.status_code == 200:
            logger.info("✅ Health endpoint working")
        else:
            logger.error(f"❌ Health endpoint failed: {response.status_code}")
            return False
        
        # Test ask endpoint
        response = requests.get("http://localhost:8000/api/v1/ask?q=apa%20itu%20niat", timeout=30)
        if response.status_code == 200:
            result = response.json()
            logger.info(f"✅ Ask endpoint working - Found {result.get('total_results', 0)} results")
            logger.info(f"   Answer: {result.get('answer', '')[:100]}...")
        else:
            logger.error(f"❌ Ask endpoint failed: {response.status_code}")
            return False
        
        return True
        
    except requests.exceptions.RequestException as e:
        logger.error(f"❌ API test failed: {e}")
        return False

def main():
    """Main pipeline execution"""
    logger.info("🚀 Starting Hadits-AI Pipeline")
    
    # Check prerequisites
    logger.info("Checking prerequisites...")
    
    if not check_file_exists(".env", "Environment file"):
        logger.error("Please create .env file from .env.example")
        return False
    
    if not check_file_exists("data/hadits.csv", "Dataset CSV"):
        logger.error("Please ensure data/hadits.csv exists")
        return False
    
    # Step 1: Preprocess data
    if not run_command("bash -c 'source venv-py311/bin/activate && python preprocess_data.py'", "Data preprocessing"):
        return False
    
    # Step 2: Build index
    if not run_command("bash -c 'source venv-py311/bin/activate && python build_index_simple.py'", "Index building"):
        return False
    
    # Step 3: Start server
    logger.info("Starting API server...")
    server_process = subprocess.Popen([
        "bash", "-c", 
        "source venv-py311/bin/activate && python simple_main.py"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait for server to start
    logger.info("Waiting for server to start...")
    time.sleep(10)
    
    try:
        # Step 4: Test API
        if not test_api():
            logger.error("❌ API testing failed")
            return False
        
        logger.info("🎉 Pipeline completed successfully!")
        logger.info("📡 API is running at http://localhost:8000")
        logger.info("📚 Documentation at http://localhost:8000/docs")
        logger.info("🔍 Test with: curl 'http://localhost:8000/api/v1/ask?q=apa%20itu%20niat'")
        
        # Keep server running
        logger.info("Server is running. Press Ctrl+C to stop.")
        server_process.wait()
        
    except KeyboardInterrupt:
        logger.info("Stopping server...")
        server_process.terminate()
        server_process.wait()
        logger.info("Server stopped.")
    
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        server_process.terminate()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)