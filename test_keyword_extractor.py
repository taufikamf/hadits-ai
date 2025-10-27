"""
Test the Enhanced Keyword Extractor without external dependencies
"""

import sys
import os
sys.path.append(os.path.abspath('.'))

# Mock pandas for testing
class MockDataFrame:
    def __init__(self, data):
        self.data = data
        self.columns = list(data.keys()) if data else []
    
    def dropna(self):
        # Mock dropna method
        return self
    
    def tolist(self):
        return self.data.get('terjemah', [])
    
    def __getitem__(self, key):
        return MockDataFrame({key: self.data.get(key, [])})

def mock_read_csv(file_path):
    # Mock CSV reader for testing
    sample_data = {
        'terjemah': [
            "Rasulullah saw bersabda tentang shalat lima waktu yang wajib",
            "Bagaimana cara berwudhu yang benar menurut sunnah",
            "Puasa ramadan adalah kewajiban bagi setiap muslim", 
            "Zakat fitrah wajib dikeluarkan sebelum shalat ied",
            "Hukum shalat jumat bagi kaum wanita"
        ]
    }
    return MockDataFrame(sample_data)

# Mock pandas module
class MockPandas:
    def read_csv(self, file_path):
        return mock_read_csv(file_path)
    
    def isna(self, value):
        return value is None or value == ""

# Monkey patch for testing
sys.modules['pandas'] = MockPandas()

# Now we can test our module
from utils.keyword_extractor import HybridKeywordExtractor

def test_keyword_extractor():
    print("=== Testing Enhanced Keyword Extractor ===")
    
    # Test data
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
    
    # Initialize extractor
    extractor = HybridKeywordExtractor(min_frequency=1, max_ngram=3)
    
    print("1. Testing text normalization...")
    test_text = "Rasulullah shallallahu 'alaihi wasallam bersabda..."
    normalized = extractor.normalize_text(test_text)
    print(f"   Original: {test_text}")
    print(f"   Normalized: {normalized}")
    
    print("\n2. Testing n-gram generation...")
    ngrams = extractor.generate_ngrams(sample_texts[0])
    print(f"   N-grams from '{sample_texts[0]}': {ngrams[:10]}")
    
    print("\n3. Testing meaningful term detection...")
    test_terms = ["shalat", "al", "saw", "yang", "wajib", "abc"]
    for term in test_terms:
        is_meaningful = extractor.is_meaningful_term(term)
        print(f"   '{term}': {'✓' if is_meaningful else '✗'}")
    
    print("\n4. Testing hybrid extraction...")
    try:
        results = extractor.hybrid_extract(sample_texts)
        print(f"   Extracted {len(results)} keywords with metadata")
        
        # Show top keywords
        sorted_keywords = sorted(results.items(), key=lambda x: x[1]['score'], reverse=True)
        print("   Top keywords:")
        for keyword, info in sorted_keywords[:10]:
            print(f"     {keyword}: score={info['score']:.3f}, islamic={info['is_islamic_term']}")
    
    except Exception as e:
        print(f"   Error in extraction: {e}")
    
    print("\n5. Testing keywords map creation...")
    try:
        keywords_map = extractor.create_keywords_map(results, min_score=0.0)
        print(f"   Created keywords map with {len(keywords_map)} entries")
        
        # Show sample mappings
        print("   Sample mappings:")
        for canonical, variants in list(keywords_map.items())[:5]:
            print(f"     {canonical}: {variants}")
            
    except Exception as e:
        print(f"   Error in map creation: {e}")
    
    print("\n✅ Keyword extractor testing completed!")

if __name__ == "__main__":
    test_keyword_extractor()