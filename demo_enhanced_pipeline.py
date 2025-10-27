"""
Demonstration of Enhanced Hadith Query Processing Pipeline
=========================================================

This script demonstrates the complete enhanced query processing pipeline
with realistic examples showing the improvements made.

Author: Hadith AI Team
Date: 2024
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath('.'))

def demonstrate_query_processing():
    """Demonstrate the enhanced query processing pipeline."""
    print("🚀 Hadith Query Processing Pipeline Demonstration")
    print("=" * 55)
    
    # Import core components
    from preprocessing.query_preprocessor import preprocess_query, extract_key_terms
    
    # Sample queries that showcase different aspects
    sample_queries = [
        "Apa hukum shalat jum'at bagi wanita?",
        "Bagaimana cara berwudhu yang benar menurut sunnah?",
        "Apakah puasa ramadan wajib bagi semua muslim dewasa?",
        "Hukum jual beli dengan sistem riba dalam Islam",
        "Tata cara nikah dan persyaratannya dalam syariat"
    ]
    
    print("🔍 STAGE 1: Query Preprocessing")
    print("-" * 30)
    
    for i, query in enumerate(sample_queries, 1):
        print(f"Query {i}: {query}")
        
        # Preprocess query
        processed = preprocess_query(query)
        key_terms = extract_key_terms(query)
        
        print(f"  Preprocessed: {processed}")
        print(f"  Key terms: {key_terms}")
        print()
    
    print("🎯 STAGE 2: Enhanced Keyword Extraction & Optimization")
    print("-" * 50)
    
    # Create test keywords map for demonstration
    import json
    from types import ModuleType
    
    # Mock pandas for testing
    mock_pandas = ModuleType('pandas')
    mock_pandas.isna = lambda x: x is None or x == ""
    sys.modules['pandas'] = mock_pandas
    
    # Test keywords map
    demo_keywords_map = {
        "shalat": ["shalat", "salat", "sholat", "solat", "shalatnya", "bershalat"],
        "hukum": ["hukum", "syariat", "halal", "haram", "makruh", "sunnah", "wajib"],
        "wudhu": ["wudhu", "wudu", "berwudhu", "bersuci", "thaharah"], 
        "puasa": ["puasa", "shaum", "berpuasa", "sahur", "iftar", "ramadan"],
        "nikah": ["nikah", "kawin", "menikah", "pernikahan", "perkawinan"],
        "jual": ["jual", "beli", "jual beli", "perdagangan", "dagang", "riba"],
        "wanita": ["wanita", "perempuan", "kaum hawa"],
        "muslim": ["muslim", "muslimah", "umat islam", "kaum muslim"]
    }
    
    # Save temporary keywords map
    temp_map_path = Path("/tmp/demo_keywords.json")
    with open(temp_map_path, 'w', encoding='utf-8') as f:
        json.dump({"keywords": demo_keywords_map}, f)
    
    from utils.query_optimizer import EnhancedQueryOptimizer
    
    optimizer = EnhancedQueryOptimizer(str(temp_map_path))
    
    for i, query in enumerate(sample_queries, 1):
        print(f"Query {i}: {query}")
        
        # Extract keywords
        matched_phrases, canonical_keywords = optimizer.extract_keywords_from_query(query)
        
        # Optimize query
        optimized, keywords = optimizer.optimize_query(query, return_keywords=True)
        
        print(f"  Matched phrases: {matched_phrases}")
        print(f"  Canonical keywords: {canonical_keywords}")
        print(f"  Final keywords: {keywords}")
        print(f"  Optimized query: {optimized}")
        print()
    
    print("📊 STAGE 3: Enhanced Retrieval Simulation")
    print("-" * 40)
    
    from retriever.query_runner import (
        calculate_literal_overlap,
        boost_literal_overlap_score,
        enhanced_keyword_match
    )
    
    # Mock hadith documents for testing
    mock_hadiths = [
        {
            "id": 1,
            "kitab": "Shahih Bukhari",
            "terjemah": "Shalat jum'at tidak wajib bagi wanita, tetapi jika mereka hadir maka pahalanya sama",
            "score": 0.85
        },
        {
            "id": 2, 
            "kitab": "Shahih Muslim",
            "terjemah": "Cara berwudhu yang benar dimulai dengan niat, mencuci muka, tangan, dan kaki",
            "score": 0.82
        },
        {
            "id": 3,
            "kitab": "Sunan Abu Daud", 
            "terjemah": "Puasa ramadan adalah kewajiban bagi setiap muslim yang sudah baligh dan berakal",
            "score": 0.79
        },
        {
            "id": 4,
            "kitab": "Sunan Tirmidzi",
            "terjemah": "Haram hukumnya memakan hasil riba dalam jual beli dan perdagangan",
            "score": 0.76
        }
    ]
    
    for i, query in enumerate(sample_queries[:4], 1):
        print(f"Query {i}: {query}")
        
        # Get keywords for this query
        _, keywords = optimizer.optimize_query(query, return_keywords=True)
        
        # Test retrieval with the corresponding mock hadith
        hadith = mock_hadiths[i-1]
        
        # Calculate literal overlap
        overlap = calculate_literal_overlap(query, hadith["terjemah"])
        
        # Test keyword matching
        meets_criteria, match_details = enhanced_keyword_match(
            hadith["terjemah"], keywords, min_match=1
        )
        
        # Apply score boosting
        boosted_docs = boost_literal_overlap_score([hadith], query, boost_factor=0.2)
        boosted_hadith = boosted_docs[0]
        
        print(f"  Keywords: {keywords}")
        print(f"  Hadith: {hadith['terjemah']}")
        print(f"  Literal overlap: {overlap:.3f}")
        print(f"  Keyword match: {meets_criteria} (matched: {match_details['matched_keywords']})")
        print(f"  Original score: {hadith['score']:.3f}")
        print(f"  Boosted score: {boosted_hadith['score']:.3f} (+{boosted_hadith.get('boost_applied', 0):.3f})")
        print()
    
    print("📝 STAGE 4: Logging and Analytics")
    print("-" * 35)
    
    from retriever.query_runner import log_query_results
    
    # Demonstrate logging
    sample_query = sample_queries[0]
    sample_keywords = ["hukum", "shalat", "jumat", "wanita"]
    sample_results = [
        {
            "rank": 1,
            "kitab": "Shahih Bukhari",
            "id": "bukhari_1234",
            "score": 0.95,
            "literal_overlap": 0.8,
            "boost_applied": 0.16,
            "terjemah": "Shalat jum'at tidak wajib bagi wanita..."
        }
    ]
    
    print(f"Logging query: {sample_query}")
    print(f"Keywords: {sample_keywords}")
    print(f"Results: {len(sample_results)} hadith(s)")
    
    # Log the results
    log_query_results(sample_query, sample_keywords, sample_results, 123.5)
    
    # Check log file
    log_file = Path("data/logs/query_logs.jsonl")
    if log_file.exists():
        print(f"✅ Query logged to: {log_file}")
        print(f"📊 Log file size: {log_file.stat().st_size} bytes")
    
    print("\n" + "=" * 55)
    print("🎉 DEMONSTRATION COMPLETE!")
    print("\n📋 Key Improvements Demonstrated:")
    print("   ✅ Advanced query preprocessing with lemmatization")
    print("   ✅ Hybrid keyword extraction (statistical + rule-based)")
    print("   ✅ Enhanced query optimization with semantic expansion")
    print("   ✅ Literal overlap boosting for better ranking")
    print("   ✅ Comprehensive logging for analytics")
    print("   ✅ Multi-concept query handling")
    print("\n💡 The enhanced pipeline provides:")
    print("   - Better keyword detection for Islamic terms")
    print("   - More accurate query understanding")
    print("   - Improved search result ranking")
    print("   - Better handling of complex queries")
    print("   - Comprehensive query analytics")

if __name__ == "__main__":
    demonstrate_query_processing()