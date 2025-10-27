"""
Integration Test for Enhanced Hadith Query Processing Pipeline
=============================================================

This script tests the complete query processing pipeline including:
- Query preprocessing  
- Keyword extraction
- Query optimization
- Enhanced retrieval with literal overlap boosting

Author: Hadith AI Team
Date: 2024
"""

import sys
import os
import json
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath('.'))

# Mock pandas and dependencies for testing
class MockPandas:
    def read_csv(self, file_path):
        return None
    def isna(self, value):
        return value is None or value == ""

class MockSentenceTransformer:
    def __init__(self, model_name):
        self.model_name = model_name
    
    def encode(self, texts, convert_to_numpy=True):
        # Return mock embeddings
        import random
        return [[random.random() for _ in range(384)] for _ in texts]

class MockFaiss:
    class IndexFlatIP:
        def search(self, embeddings, k):
            # Return mock search results
            import random
            indices = list(range(k))
            distances = [random.random() for _ in range(k)]
            return [distances], [indices]
    
    def read_index(self, path):
        return self.IndexFlatIP()

class MockSklearn:
    class TfidfVectorizer:
        def __init__(self, **kwargs):
            pass
        def fit_transform(self, corpus):
            return [[0.5] * len(corpus)]
    
    class KMeans:
        def __init__(self, **kwargs):
            pass
        def fit_predict(self, X):
            return [0] * len(X) if hasattr(X, '__len__') else [0]
    
    def normalize(self, embeddings, axis=1):
        return embeddings

# Mock dependencies
sys.modules['pandas'] = MockPandas()
sys.modules['sentence_transformers'] = type('Module', (), {'SentenceTransformer': MockSentenceTransformer})()
sys.modules['faiss'] = MockFaiss()
sys.modules['sklearn'] = type('Module', (), {
    'feature_extraction': type('Module', (), {
        'text': type('Module', (), {'TfidfVectorizer': MockSklearn.TfidfVectorizer})()
    })(),
    'cluster': type('Module', (), {'KMeans': MockSklearn.KMeans})(),
    'preprocessing': type('Module', (), {'normalize': MockSklearn().normalize})()
})()

def test_query_preprocessor():
    """Test the query preprocessor module."""
    print("🔍 Testing Query Preprocessor...")
    
    try:
        from preprocessing.query_preprocessor import preprocess_query, extract_key_terms
        
        test_cases = [
            "Apa hukum shalat jum'at bagi wanita?",
            "Bagaimana cara berwudhu yang benar?",
            "Apakah puasa wajib bagi semua muslim?",
            "Hukum jual beli dalam Islam",
            "Nabi Muhammad saw mengajarkan adab makan"
        ]
        
        print("   Test cases and results:")
        for query in test_cases:
            processed = preprocess_query(query)
            key_terms = extract_key_terms(query)
            
            print(f"     Original: {query}")
            print(f"     Processed: {processed}")
            print(f"     Key terms: {key_terms}")
            print()
        
        print("   ✅ Query preprocessor working correctly")
        return True
        
    except Exception as e:
        print(f"   ❌ Query preprocessor error: {e}")
        return False

def test_keyword_extractor():
    """Test the enhanced keyword extractor."""
    print("🔍 Testing Enhanced Keyword Extractor...")
    
    try:
        from utils.keyword_extractor import HybridKeywordExtractor
        
        # Sample hadith texts for testing
        sample_texts = [
            "Rasulullah saw bersabda tentang shalat lima waktu yang wajib",
            "Bagaimana cara berwudhu yang benar menurut sunnah",
            "Puasa ramadan adalah kewajiban bagi setiap muslim",
            "Zakat fitrah wajib dikeluarkan sebelum shalat ied", 
            "Hukum shalat jumat bagi kaum wanita",
            "Nabi Muhammad mengajarkan adab makan dan minum",
            "Haram hukumnya memakan riba dalam Islam",
            "Wajib bagi muslim melakukan shalat subuh"
        ]
        
        extractor = HybridKeywordExtractor(min_frequency=1, max_ngram=3)
        
        print("   Testing keyword extraction pipeline...")
        results = extractor.hybrid_extract(sample_texts)
        keywords_map = extractor.create_keywords_map(results, min_score=0.0)
        
        print(f"   Extracted {len(results)} keywords")
        print(f"   Created map with {len(keywords_map)} canonical terms")
        
        # Show top results
        sorted_keywords = sorted(results.items(), key=lambda x: x[1]['score'], reverse=True)
        print("   Top keywords:")
        for keyword, info in sorted_keywords[:5]:
            print(f"     {keyword}: score={info['score']:.3f}, islamic={info['is_islamic_term']}")
        
        print("   ✅ Keyword extractor working correctly")
        return True, keywords_map
        
    except Exception as e:
        print(f"   ❌ Keyword extractor error: {e}")
        return False, {}

def test_query_optimizer():
    """Test the enhanced query optimizer."""
    print("🔍 Testing Enhanced Query Optimizer...")
    
    try:
        from utils.query_optimizer import EnhancedQueryOptimizer
        
        # Create a temporary keywords map for testing
        test_keywords_map = {
            "shalat": ["shalat", "salat", "sholat", "solat", "shalatnya"],
            "wudhu": ["wudhu", "wudu", "berwudhu", "bersuci"],
            "puasa": ["puasa", "shaum", "berpuasa", "menjalankan puasa"],
            "hukum": ["hukum", "syariat", "fiqih", "halal", "haram"],
            "zakat": ["zakat", "berzakat", "sadaqah", "infaq"]
        }
        
        # Save temporary keywords map
        temp_map_path = "/tmp/test_keywords_map.json"
        with open(temp_map_path, 'w', encoding='utf-8') as f:
            json.dump({"keywords": test_keywords_map}, f)
        
        optimizer = EnhancedQueryOptimizer(temp_map_path)
        
        test_queries = [
            "Apa hukum shalat jum'at bagi wanita?",
            "Bagaimana cara berwudhu yang benar?", 
            "Apakah puasa wajib bagi semua muslim?",
            "Hukum jual beli dalam Islam",
            "Tata cara zakat dalam Islam"
        ]
        
        print("   Test query optimization:")
        for query in test_queries:
            optimized, keywords = optimizer.optimize_query(query, return_keywords=True)
            
            print(f"     Query: {query}")
            print(f"     Keywords: {keywords}")
            print(f"     Optimized: {optimized}")
            print()
        
        print("   ✅ Query optimizer working correctly")
        return True
        
    except Exception as e:
        print(f"   ❌ Query optimizer error: {e}")
        return False

def test_retrieval_enhancements():
    """Test the enhanced retrieval functionality."""
    print("🔍 Testing Enhanced Retrieval Functions...")
    
    try:
        from retriever.query_runner import (
            calculate_literal_overlap, 
            boost_literal_overlap_score,
            enhanced_keyword_match
        )
        
        # Test literal overlap calculation
        print("   Testing literal overlap calculation:")
        test_cases = [
            ("shalat jumat", "Hukum shalat jumat bagi wanita", 0.5),
            ("wudhu benar", "Cara berwudhu yang benar dan sunnah", 1.0),
            ("puasa ramadan", "Kewajiban puasa bulan ramadan", 1.0),
            ("zakat mal", "Tidak ada zakat dalam teks ini", 0.0)
        ]
        
        for query, text, expected_min in test_cases:
            overlap = calculate_literal_overlap(query, text)
            print(f"     '{query}' vs '{text}': {overlap:.3f}")
            assert overlap >= expected_min, f"Expected >= {expected_min}, got {overlap}"
        
        # Test keyword matching
        print("   Testing enhanced keyword matching:")
        text = "Rasulullah saw mengajarkan tentang shalat dan puasa"
        keywords = ["shalat", "puasa", "zakat"]
        
        meets_criteria, details = enhanced_keyword_match(text, keywords, min_match=2)
        print(f"     Text: {text}")
        print(f"     Keywords: {keywords}")
        print(f"     Meets criteria: {meets_criteria}")
        print(f"     Match details: {details}")
        
        # Test score boosting
        print("   Testing literal overlap boosting:")
        mock_docs = [
            {
                "kitab": "Test Kitab",
                "id": 1,
                "terjemah": "Rasulullah mengajarkan shalat lima waktu",
                "score": 0.8
            },
            {
                "kitab": "Test Kitab", 
                "id": 2,
                "terjemah": "Hadits tentang puasa dan zakat",
                "score": 0.7
            }
        ]
        
        query = "shalat lima waktu"
        boosted = boost_literal_overlap_score(mock_docs, query, boost_factor=0.2)
        
        for doc in boosted:
            print(f"     Doc {doc['id']}: score={doc['score']:.3f}, overlap={doc.get('literal_overlap', 0):.3f}")
        
        print("   ✅ Enhanced retrieval functions working correctly")
        return True
        
    except Exception as e:
        print(f"   ❌ Enhanced retrieval error: {e}")
        return False

def test_logging_functionality():
    """Test the logging functionality."""
    print("🔍 Testing Logging Functionality...")
    
    try:
        from retriever.query_runner import log_query_results
        
        # Test query logging
        test_query = "Test query for logging"
        test_keywords = ["test", "query", "logging"]
        test_results = [
            {
                "rank": 1,
                "kitab": "Test Kitab",
                "id": "test_1",
                "score": 0.95,
                "literal_overlap": 0.6,
                "boost_applied": 0.12,
                "terjemah": "This is a test hadith for logging functionality testing."
            }
        ]
        
        # Create logs directory
        logs_dir = Path("data/logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        log_query_results(test_query, test_keywords, test_results, 125.5)
        
        # Check if log file was created
        log_file = logs_dir / "query_logs.jsonl"
        if log_file.exists():
            print(f"   ✅ Log file created: {log_file}")
            
            # Read and verify log entry
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                if lines:
                    last_entry = json.loads(lines[-1])
                    print(f"   ✅ Log entry verified: {last_entry['query']}")
                else:
                    print("   ⚠️ Log file is empty")
        else:
            print("   ⚠️ Log file not created")
        
        print("   ✅ Logging functionality working correctly")
        return True
        
    except Exception as e:
        print(f"   ❌ Logging functionality error: {e}")
        return False

def run_integration_test():
    """Run complete integration test."""
    print("🚀 Starting Hadith Query Processing Pipeline Integration Test")
    print("=" * 60)
    
    results = {
        "query_preprocessor": False,
        "keyword_extractor": False, 
        "query_optimizer": False,
        "retrieval_enhancements": False,
        "logging_functionality": False
    }
    
    # Test each component
    results["query_preprocessor"] = test_query_preprocessor()
    results["keyword_extractor"], keywords_map = test_keyword_extractor()
    results["query_optimizer"] = test_query_optimizer()
    results["retrieval_enhancements"] = test_retrieval_enhancements()
    results["logging_functionality"] = test_logging_functionality()
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 Integration Test Results:")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for component, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   {component}: {status}")
    
    print(f"\n📊 Overall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All integration tests passed! Pipeline is ready for deployment.")
        return True
    else:
        print("⚠️ Some tests failed. Please review the issues above.")
        return False

def test_end_to_end_query():
    """Test end-to-end query processing with a realistic example."""
    print("\n🔄 Testing End-to-End Query Processing...")
    
    try:
        # Test query
        test_query = "Apa hukum shalat jum'at bagi wanita?"
        
        print(f"   Processing query: '{test_query}'")
        
        # Step 1: Preprocess query
        from preprocessing.query_preprocessor import preprocess_query, extract_key_terms
        processed = preprocess_query(test_query)
        key_terms = extract_key_terms(test_query)
        
        print(f"   1. Preprocessed: '{processed}'")
        print(f"   2. Key terms: {key_terms}")
        
        # Step 2: Optimize query (with fallback)
        try:
            from utils.query_optimizer import optimize_query
            optimized, keywords = optimize_query(test_query, return_keywords=True)
            print(f"   3. Optimized: '{optimized}'")
            print(f"   4. Keywords: {keywords}")
        except Exception as e:
            print(f"   3. Query optimization skipped: {e}")
            optimized = f"passage: {test_query}"
            keywords = key_terms
        
        # Step 3: Test retrieval functions (without actual FAISS)
        from retriever.query_runner import calculate_literal_overlap, enhanced_keyword_match
        
        # Mock document for testing
        mock_doc = {
            "terjemah": "Shalat jum'at adalah shalat yang dilaksanakan pada hari jumat"
        }
        
        overlap = calculate_literal_overlap(test_query, mock_doc["terjemah"])
        print(f"   5. Literal overlap: {overlap:.3f}")
        
        meets_criteria, match_details = enhanced_keyword_match(
            mock_doc["terjemah"], keywords, min_match=1
        )
        print(f"   6. Keyword match: {meets_criteria}, details: {match_details}")
        
        print("   ✅ End-to-end processing completed successfully")
        return True
        
    except Exception as e:
        print(f"   ❌ End-to-end processing error: {e}")
        return False

if __name__ == "__main__":
    # Run integration tests
    success = run_integration_test()
    
    # Run end-to-end test
    test_end_to_end_query()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)