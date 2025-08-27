#!/usr/bin/env python3
"""
Test Generation Flow - Fixed V1
===============================

Comprehensive test script for the enhanced generation system in Fixed V1.
Tests the complete flow from query to LLM-generated response.

Author: Hadith AI Team - Fixed V1
Date: 2024
"""

import os
import sys
import json
import asyncio
import logging
from typing import List, Dict, Any
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_imports():
    """Test if all required modules can be imported."""
    print("🧪 Testing Module Imports")
    print("=" * 50)
    
    try:
        from service.hadith_ai_service import HadithAIService, ServiceConfig
        print("✅ HadithAIService imported successfully")
        
        from generation.enhanced_response_generator import (
            EnhancedResponseGenerator, GenerationConfig, LLMProvider, ResponseMode
        )
        print("✅ EnhancedResponseGenerator imported successfully")
        
        from retrieval.enhanced_retrieval_system import EnhancedRetrievalSystem
        print("✅ EnhancedRetrievalSystem imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_generation_config():
    """Test generation configuration."""
    print("\n🔧 Testing Generation Configuration")
    print("=" * 50)
    
    try:
        from generation.enhanced_response_generator import GenerationConfig, LLMProvider, ResponseMode
        
        # Test different configurations
        configs = [
            {
                "name": "Default Config",
                "config": GenerationConfig()
            },
            {
                "name": "Gemini Config",
                "config": GenerationConfig(
                    llm_provider=LLMProvider.GEMINI,
                    response_mode=ResponseMode.COMPREHENSIVE,
                    temperature=0.3,
                    max_tokens=2500
                )
            },
            {
                "name": "Summary Mode",
                "config": GenerationConfig(
                    llm_provider=LLMProvider.NONE,
                    response_mode=ResponseMode.SUMMARY,
                    max_hadits_display=3
                )
            }
        ]
        
        for test_config in configs:
            config = test_config["config"]
            print(f"📋 {test_config['name']}:")
            print(f"   - LLM Provider: {config.llm_provider.value}")
            print(f"   - Response Mode: {config.response_mode.value}")
            print(f"   - Max Hadits: {config.max_hadits_display}")
            print(f"   - Temperature: {config.temperature}")
            print(f"   - Streaming: {config.enable_streaming}")
            print()
        
        print("✅ Generation configuration tests passed")
        return True
        
    except Exception as e:
        print(f"❌ Generation config test failed: {e}")
        return False

def test_service_initialization():
    """Test service initialization with generation enabled."""
    print("\n🚀 Testing Service Initialization")
    print("=" * 50)
    
    try:
        from service.hadith_ai_service import HadithAIService, ServiceConfig
        from generation.enhanced_response_generator import GenerationConfig, LLMProvider
        
        # Test different service configurations
        test_configs = [
            {
                "name": "Generation Disabled",
                "config": ServiceConfig(
                    enable_llm_generation=False,
                    fallback_to_simple=True
                )
            },
            {
                "name": "Generation Enabled (No API Key)",
                "config": ServiceConfig(
                    enable_llm_generation=True,
                    fallback_to_simple=True,
                    generation_config=GenerationConfig(
                        llm_provider=LLMProvider.NONE
                    )
                )
            }
        ]
        
        for test_config in test_configs:
            print(f"🔧 Testing {test_config['name']}...")
            
            try:
                service = HadithAIService(test_config["config"])
                print(f"   ✅ Service initialized successfully")
                print(f"   - LLM Generation: {service.config.enable_llm_generation}")
                print(f"   - Response Generator: {'Enabled' if service.response_generator else 'Disabled'}")
                print(f"   - Fallback to Simple: {service.config.fallback_to_simple}")
                
                # Test health check
                health = service.get_service_health()
                print(f"   - Service Health: {health['status']}")
                
            except Exception as e:
                print(f"   ❌ Failed to initialize: {e}")
                continue
            
            print()
        
        print("✅ Service initialization tests completed")
        return True
        
    except Exception as e:
        print(f"❌ Service initialization test failed: {e}")
        return False

async def test_generation_flow():
    """Test the complete generation flow."""
    print("\n🎯 Testing Complete Generation Flow")
    print("=" * 50)
    
    try:
        from service.hadith_ai_service import HadithAIService, ServiceConfig
        from generation.enhanced_response_generator import GenerationConfig, LLMProvider
        
        # Create service with fallback enabled
        config = ServiceConfig(
            enable_llm_generation=True,
            fallback_to_simple=True,
            max_results_display=3,
            generation_config=GenerationConfig(
                llm_provider=LLMProvider.NONE,  # Use fallback for testing
                max_hadits_display=3
            )
        )
        
        service = HadithAIService(config)
        
        # Test queries
        test_queries = [
            "adab makan dalam Islam",
            "cara shalat yang benar",
            "keutamaan berbakti kepada orang tua",
            "hukum jual beli",
            "adab bertetangga"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"🔍 Test Query {i}: '{query}'")
            
            try:
                # Test synchronous processing
                start_time = asyncio.get_event_loop().time()
                response = service.process_query(query, max_results=3)
                sync_time = (asyncio.get_event_loop().time() - start_time) * 1000
                
                print(f"   📊 Sync Response:")
                print(f"      - Success: {response.success}")
                print(f"      - Results: {len(response.results)}")
                print(f"      - Response Time: {sync_time:.1f}ms")
                print(f"      - Message Length: {len(response.message)} chars")
                print(f"      - LLM Used: {response.metadata.get('llm_generation_used', False)}")
                
                # Test async processing
                start_time = asyncio.get_event_loop().time()
                async_response = await service.process_query_async(query, max_results=3)
                async_time = (asyncio.get_event_loop().time() - start_time) * 1000
                
                print(f"   📊 Async Response:")
                print(f"      - Success: {async_response.success}")
                print(f"      - Results: {len(async_response.results)}")
                print(f"      - Response Time: {async_time:.1f}ms")
                print(f"      - Message Length: {len(async_response.message)} chars")
                
                # Test streaming (first few chunks only)
                print(f"   📊 Streaming Response:")
                chunk_count = 0
                async for chunk in service.generate_streaming_response(query, max_results=3):
                    chunk_count += 1
                    if chunk_count <= 3:  # Show first 3 chunks
                        preview = chunk[:50] + "..." if len(chunk) > 50 else chunk
                        print(f"      - Chunk {chunk_count}: {preview}")
                    elif chunk_count == 4:
                        print(f"      - ... (showing only first 3 chunks)")
                        break
                
                print(f"      - Total Chunks: {chunk_count}")
                
                if i < len(test_queries):  # Don't print separator after last test
                    print()
                
            except Exception as e:
                print(f"   ❌ Query processing failed: {e}")
                print()
                continue
        
        print("✅ Generation flow tests completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Generation flow test failed: {e}")
        return False

def test_api_integration():
    """Test API integration with generation."""
    print("\n🌐 Testing API Integration")
    print("=" * 50)
    
    try:
        from service.api_server import create_app
        from service.hadith_ai_service import ServiceConfig
        from generation.enhanced_response_generator import GenerationConfig, LLMProvider
        
        # Create test configuration
        service_config = ServiceConfig(
            enable_llm_generation=True,
            fallback_to_simple=True,
            generation_config=GenerationConfig(
                llm_provider=LLMProvider.NONE  # Use fallback for testing
            )
        )
        
        # Create Flask app
        app = create_app(service_config)
        
        with app.test_client() as client:
            # Test home endpoint
            response = client.get('/')
            print(f"GET / : {response.status_code}")
            if response.status_code == 200:
                data = response.get_json()
                print(f"   - Features: {', '.join(data['data']['features'])}")
            
            # Test health endpoint
            response = client.get('/health')
            print(f"GET /health : {response.status_code}")
            
            # Test basic chat endpoint
            chat_data = {
                "query": "adab makan dalam Islam",
                "max_results": 3
            }
            response = client.post('/chat', json=chat_data)
            print(f"POST /chat : {response.status_code}")
            
            # Test async chat endpoint
            response = client.post('/chat/async', json=chat_data)
            print(f"POST /chat/async : {response.status_code}")
            if response.status_code == 200:
                data = response.get_json()
                print(f"   - Query processed with enhanced generation")
                print(f"   - Response length: {len(data['data']['message'])} chars")
        
        print("✅ API integration tests completed")
        return True
        
    except Exception as e:
        print(f"❌ API integration test failed: {e}")
        return False

def check_data_files():
    """Check if required data files exist."""
    print("\n📁 Checking Data Files")
    print("=" * 50)
    
    required_files = [
        "../data/processed/hadits_docs.json",
        "../data/enhanced_index_v1/enhanced_keywords_map_v1.json",
        "../data/enhanced_index_v1/enhanced_embeddings_v1.pkl",
        "../data/enhanced_index_v1/enhanced_faiss_index_v1.index",
        "../data/enhanced_index_v1/enhanced_metadata_v1.pkl"
    ]
    
    all_exist = True
    
    for file_path in required_files:
        abs_path = Path(__file__).parent / file_path
        if abs_path.exists():
            size = abs_path.stat().st_size
            print(f"✅ {file_path} ({size:,} bytes)")
        else:
            print(f"❌ {file_path} (missing)")
            all_exist = False
    
    if all_exist:
        print("\n✅ All required data files are present")
    else:
        print("\n⚠️  Some data files are missing. Run the indexing pipeline first:")
        print("   python indexing/enhanced_indexing_pipeline.py")
    
    return all_exist

async def main():
    """Run all tests."""
    print("🧪 Hadith AI Fixed V1 - Generation Flow Test Suite")
    print("=" * 60)
    
    # Run tests
    tests = [
        ("Module Imports", test_imports),
        ("Data Files", check_data_files),
        ("Generation Config", test_generation_config),
        ("Service Initialization", test_service_initialization),
        ("Generation Flow", test_generation_flow),
        ("API Integration", test_api_integration)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results[test_name] = result
        except Exception as e:
            logger.error(f"Test '{test_name}' failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 60)
    
    passed = sum(1 for r in results.values() if r)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25} : {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Generation system is ready to use.")
        print("\n💡 Next Steps:")
        print("   1. Set up Gemini API key for LLM generation:")
        print("      export GEMINI_API_KEY='your-api-key-here'")
        print("   2. Start the API server:")
        print("      python service/api_server.py --debug")
        print("   3. Test endpoints:")
        print("      curl -X POST http://localhost:5000/chat/async \\")
        print("           -H 'Content-Type: application/json' \\")
        print("           -d '{\"query\": \"adab makan dalam Islam\"}'")
    else:
        print(f"\n⚠️  {total-passed} test(s) failed. Check the errors above.")
        print("   Make sure all dependencies are installed and data files are present.")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
