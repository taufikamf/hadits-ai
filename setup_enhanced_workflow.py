#!/usr/bin/env python3
"""
Enhanced Hadits AI Setup Script
===============================

This script sets up the complete Enhanced Hadits AI workflow:
1. Installs dependencies
2. Generates enhanced keywords
3. Creates embeddings
4. Builds search index
5. Tests the system
6. Starts the API service

Author: Hadith AI Team - Enhanced Setup
Date: 2024
"""

import os
import sys
import subprocess
import time
import json
from pathlib import Path

def run_command(cmd, description, check=True):
    """Run a command with logging"""
    print(f"\n🔄 {description}")
    print(f"Command: {cmd}")
    print("-" * 50)
    
    start_time = time.time()
    
    try:
        if isinstance(cmd, str):
            result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
        else:
            result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        
        duration = time.time() - start_time
        
        if result.stdout:
            print(result.stdout)
        
        print(f"✅ {description} completed in {duration:.2f}s")
        return True
        
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        print(f"❌ {description} failed after {duration:.2f}s")
        print(f"Error: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False

def check_prerequisites():
    """Check if basic prerequisites are met"""
    print("🔍 Checking Prerequisites...")
    print("=" * 60)
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8+ required")
        return False
    
    # Check if data exists
    data_file = Path("data/processed/hadits_docs.json")
    if not data_file.exists():
        print(f"❌ Required data file not found: {data_file}")
        print("Please ensure hadits documents are prepared first")
        return False
    
    print("✅ Prerequisites check passed")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing Dependencies...")
    
    # Check if pip is available
    try:
        subprocess.run([sys.executable, "-m", "pip", "--version"], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        print("❌ pip not available")
        return False
    
    # Install requirements
    cmd = [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
    return run_command(cmd, "Installing Python packages")

def generate_enhanced_keywords_and_embeddings():
    """Generate enhanced keywords and embeddings"""
    print("🚀 Generating Enhanced Keywords & Embeddings...")
    
    cmd = [sys.executable, "embedding/embed_model_optimized.py"]
    return run_command(cmd, "Generating enhanced keywords and embeddings")

def build_search_index():
    """Build FAISS search index"""
    print("🔍 Building Search Index...")
    
    cmd = [sys.executable, "indexing/build_index.py"]
    return run_command(cmd, "Building FAISS search index")

def test_system():
    """Test the complete system"""
    print("🧪 Testing System...")
    
    cmd = [sys.executable, "test_enhanced_workflow.py"]
    success = run_command(cmd, "Running system tests", check=False)
    
    # Check test results
    results_file = Path("test_results_enhanced.json")
    if results_file.exists():
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
            
            passed = sum(1 for r in results.values() if r.get("success", False))
            total = len(results)
            success_rate = (passed / total) * 100 if total > 0 else 0
            
            print(f"\n📊 Test Results: {passed}/{total} passed ({success_rate:.1f}%)")
            
            if passed == total:
                print("🎉 All tests passed!")
                return True
            else:
                print("⚠️ Some tests failed. Check the output above.")
                failed_tests = [name for name, result in results.items() if not result.get("success", False)]
                print(f"Failed tests: {', '.join(failed_tests)}")
                return False
                
        except Exception as e:
            print(f"❌ Could not parse test results: {e}")
            return False
    
    return success

def create_env_file():
    """Create .env file if it doesn't exist"""
    env_file = Path(".env")
    if not env_file.exists():
        print("📝 Creating .env file...")
        
        env_content = """# Enhanced Hadits AI Configuration

# Data paths
DATA_CLEAN_PATH=data/processed/hadits_docs.json

# Database paths
FAISS_INDEX_PATH=./db/hadits_faiss.index
FAISS_METADATA_PATH=./db/hadits_metadata.pkl
CHROMA_DB_PATH=./db/hadits_index

# API Configuration
GEMINI_API_KEY=your_gemini_api_key_here

# Optional: Advanced settings
MIN_FREQUENCY=15
MAX_NGRAM=3
"""
        
        with open(env_file, 'w', encoding='utf-8') as f:
            f.write(env_content)
        
        print("✅ .env file created")
        print("⚠️ Please update GEMINI_API_KEY in .env file for full functionality")
    else:
        print("✅ .env file already exists")

def main():
    """Main setup process"""
    print("🚀 Enhanced Hadits AI Setup")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Step 1: Check prerequisites
        if not check_prerequisites():
            print("❌ Prerequisites check failed. Please fix the issues and try again.")
            return False
        
        # Step 2: Create env file
        create_env_file()
        
        # Step 3: Install dependencies
        if not install_dependencies():
            print("❌ Dependency installation failed. Please check the errors above.")
            return False
        
        # Step 4: Generate enhanced keywords and embeddings
        if not generate_enhanced_keywords_and_embeddings():
            print("❌ Enhanced keyword/embedding generation failed.")
            return False
        
        # Step 5: Build search index
        if not build_search_index():
            print("❌ Search index building failed.")
            return False
        
        # Step 6: Test system
        if not test_system():
            print("⚠️ System tests failed, but setup may still work.")
            print("You can try starting the API service manually.")
        
        # Final summary
        total_time = time.time() - start_time
        
        print("\n" + "=" * 60)
        print("🎉 ENHANCED HADITS AI SETUP COMPLETE!")
        print("=" * 60)
        print(f"Total setup time: {total_time/60:.1f} minutes")
        print()
        print("🚀 To start the API service:")
        print("   python main.py")
        print()
        print("📖 For documentation:")
        print("   Check ENHANCED_WORKFLOW_DOCUMENTATION.md")
        print()
        print("🔍 API will be available at:")
        print("   http://localhost:8000")
        print("   http://localhost:8000/docs (Swagger UI)")
        print()
        print("🧪 To run tests again:")
        print("   python test_enhanced_workflow.py")
        print("=" * 60)
        
        return True
        
    except KeyboardInterrupt:
        print("\n❌ Setup interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error during setup: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)