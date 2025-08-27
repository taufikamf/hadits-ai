#!/usr/bin/env python3
"""
Enhanced Hadits AI Workflow Test Script
=======================================

This script tests the complete workflow using the enhanced Islamic keyword extractor:
1. Enhanced keyword extraction
2. Document embedding with enhanced keywords 
3. Index building with FAISS
4. Query retrieval testing
5. Full pipeline integration test

Author: Hadith AI Team - Enhanced Testing
Date: 2024
"""

import os
import sys
import json
import time
import logging
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".")))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_enhanced_keyword_extraction():
    """Test the enhanced keyword extraction system"""
    logger.info("🧪 Testing Enhanced Keyword Extraction...")
    
    try:
        from utils.improved_keyword_extractor import EnhancedIslamicKeywordExtractor
        
        # Test with sample texts
        sample_texts = [
            "Diriwayatkan dari Abu Hurairah ra bahwa Rasulullah saw bersabda: Shalat berjamaah lebih baik daripada shalat sendirian dengan 27 derajat.",
            "Telah menceritakan kepada kami Abdullah bin Masud ra: Barang siapa yang berpuasa pada bulan Ramadan dengan iman dan mengharap pahala, diampuni dosanya yang telah lalu.",
            "Dari Anas bin Malik ra berkata: Rasulullah saw bersabda tentang zakat: Zakat adalah rukun Islam yang ketiga setelah syahadat dan shalat."
        ]
        
        extractor = EnhancedIslamicKeywordExtractor(min_frequency=1, max_ngram=2)
        keywords_map = extractor.create_enhanced_keywords_map(sample_texts)
        
        logger.info(f"✅ Enhanced keyword extraction successful")
        logger.info(f"   Generated {len(keywords_map)} keyword groups")
        
        # Show sample groups
        for i, (group, terms) in enumerate(list(keywords_map.items())[:3]):
            logger.info(f"   Sample group {i+1}: {group} -> {terms[:3]}")
        
        return True, keywords_map
        
    except Exception as e:
        logger.error(f"❌ Enhanced keyword extraction failed: {e}")
        return False, None

def test_enhanced_embedding():
    """Test document embedding with enhanced keywords"""
    logger.info("🧪 Testing Enhanced Document Embedding...")
    
    try:
        from embedding.embed_model_optimized import generate_enhanced_keywords, load_documents, load_keyword_map
        
        # Check if documents exist
        data_path = "data/processed/hadits_docs.json"
        if not os.path.exists(data_path):
            logger.warning(f"⚠️ Document file not found at {data_path}")
            return False, "Documents not found"
        
        # Test keyword map loading (will generate if not exists)
        keyword_map = load_keyword_map()
        logger.info(f"✅ Enhanced keyword map loaded with {len(keyword_map)} groups")
        
        # Test document loading
        docs = load_documents()[:5]  # Test with first 5 docs
        logger.info(f"✅ Loaded {len(docs)} documents for testing")
        
        # Test semantic tag building
        from embedding.embed_model_optimized import build_semantic_tags
        test_doc = docs[0]
        tags = build_semantic_tags(test_doc, keyword_map)
        logger.info(f"✅ Semantic tags generated: {len(tags.split(', ')) if tags else 0} tags")
        
        return True, "Enhanced embedding components working"
        
    except Exception as e:
        logger.error(f"❌ Enhanced embedding test failed: {e}")
        return False, str(e)

def test_index_building():
    """Test index building with enhanced keywords"""
    logger.info("🧪 Testing Enhanced Index Building...")
    
    try:
        # Check if embeddings exist
        embedding_path = "data/processed/hadits_embeddings.pkl"
        if not os.path.exists(embedding_path):
            logger.warning(f"⚠️ Embeddings file not found at {embedding_path}")
            logger.info("   You need to run embedding generation first")
            return False, "Embeddings not found"
        
        from indexing.build_index import load_embeddings
        embeddings, documents = load_embeddings(embedding_path)
        
        logger.info(f"✅ Loaded embeddings: {embeddings.shape}")
        logger.info(f"✅ Loaded documents: {len(documents)}")
        
        # Test metadata preparation
        from indexing.build_index import load_keyword_map, build_semantic_tags
        keyword_map = load_keyword_map()
        
        sample_doc = documents[0]
        tags = build_semantic_tags(sample_doc, keyword_map)
        logger.info(f"✅ Sample document tags: {len(tags.split(', ')) if tags else 0} tags")
        
        return True, "Index building components working"
        
    except Exception as e:
        logger.error(f"❌ Index building test failed: {e}")
        return False, str(e)

def test_query_retrieval():
    """Test query retrieval system"""
    logger.info("🧪 Testing Query Retrieval System...")
    
    try:
        # Check if retrieval system is available
        from retriever import query_runner
        
        # Test basic query
        test_query = "shalat"
        results = query_runner.query_hadits_return(test_query, top_k=3)
        
        if results and len(results) > 0:
            logger.info(f"✅ Query retrieval successful: {len(results)} results found")
            sample_result = results[0]
            logger.info(f"   Sample result from: {sample_result.get('kitab', 'Unknown')}")
            return True, "Query retrieval working"
        else:
            logger.warning("⚠️ Query retrieval returned no results")
            return False, "No results returned"
            
    except ImportError as e:
        logger.error(f"❌ Query retrieval modules not available: {e}")
        return False, "Retrieval modules not available"
    except Exception as e:
        logger.error(f"❌ Query retrieval test failed: {e}")
        return False, str(e)

def test_full_api_integration():
    """Test full API integration"""
    logger.info("🧪 Testing Full API Integration...")
    
    try:
        # Test query optimization
        from utils import query_optimizer
        
        test_question = "Bagaimana cara shalat yang benar?"
        optimized_query, keywords = query_optimizer.optimize_query(test_question, return_keywords=True)
        
        logger.info(f"✅ Query optimization working")
        logger.info(f"   Original: {test_question}")
        logger.info(f"   Optimized: {optimized_query}")
        logger.info(f"   Keywords: {keywords}")
        
        return True, "API integration components working"
        
    except Exception as e:
        logger.error(f"❌ API integration test failed: {e}")
        return False, str(e)

def generate_test_report():
    """Generate comprehensive test report"""
    logger.info("\n" + "="*60)
    logger.info("🚀 ENHANCED HADITS AI WORKFLOW TEST SUITE")
    logger.info("="*60)
    
    test_results = {}
    
    # Run all tests
    tests = [
        ("Enhanced Keyword Extraction", test_enhanced_keyword_extraction),
        ("Enhanced Document Embedding", test_enhanced_embedding),
        ("Enhanced Index Building", test_index_building),
        ("Query Retrieval System", test_query_retrieval),
        ("Full API Integration", test_full_api_integration)
    ]
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running: {test_name}")
        logger.info("-" * 40)
        
        start_time = time.time()
        success, message = test_func()
        duration = time.time() - start_time
        
        test_results[test_name] = {
            "success": success,
            "message": message,
            "duration": duration
        }
        
        status = "✅ PASSED" if success else "❌ FAILED"
        logger.info(f"{status} ({duration:.2f}s)")
        if not success:
            logger.info(f"   Error: {message}")
    
    # Generate summary
    logger.info("\n" + "="*60)
    logger.info("📊 TEST SUMMARY")
    logger.info("="*60)
    
    passed = sum(1 for r in test_results.values() if r["success"])
    total = len(test_results)
    
    logger.info(f"Tests Passed: {passed}/{total}")
    logger.info(f"Success Rate: {(passed/total)*100:.1f}%")
    
    if passed == total:
        logger.info("🎉 ALL TESTS PASSED - Enhanced workflow is ready!")
    else:
        logger.info("⚠️ Some tests failed - check the errors above")
        
        # Show failed tests
        failed_tests = [name for name, result in test_results.items() if not result["success"]]
        logger.info(f"Failed tests: {', '.join(failed_tests)}")
    
    logger.info("="*60)
    
    return test_results

def main():
    """Main test execution"""
    try:
        logger.info("🚀 Starting Enhanced Hadits AI Workflow Tests...")
        results = generate_test_report()
        
        # Save results to file
        results_file = "test_results_enhanced.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            # Convert results to JSON-serializable format
            json_results = {
                name: {
                    "success": result["success"],
                    "message": str(result["message"]),
                    "duration": result["duration"]
                }
                for name, result in results.items()
            }
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 Test results saved to: {results_file}")
        
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()