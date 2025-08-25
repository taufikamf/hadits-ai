#!/usr/bin/env python3
"""
Test script to demonstrate the usage of extracted keywords
"""

import json
from pathlib import Path

def load_keywords(file_path="data/processed/keywords_map_grouped.json"):
    """Load the generated keyword map"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if 'keywords' in data:
        return data['keywords'], data.get('metadata', {})
    else:
        return data, {}

def search_keywords(keywords, query_term):
    """Find all keyword variations for a given term"""
    results = {}
    query_lower = query_term.lower()
    
    for category, terms in keywords.items():
        # Check if query matches category name
        if query_lower in category.lower():
            results[category] = terms
        else:
            # Check if query matches any term in the category
            matching_terms = [term for term in terms if query_lower in term.lower()]
            if matching_terms:
                results[category] = matching_terms
    
    return results

def enhance_query(keywords, user_query):
    """Enhance a user query with related keywords"""
    query_words = user_query.lower().split()
    enhanced_terms = set()
    
    for word in query_words:
        matches = search_keywords(keywords, word)
        for category, terms in matches.items():
            enhanced_terms.update(terms[:5])  # Top 5 terms per category
    
    return list(enhanced_terms)

def main():
    """Demonstrate keyword usage"""
    print("🔍 Testing Hadits Keyword System")
    print("=" * 40)
    
    # Load keywords
    try:
        keywords, metadata = load_keywords()
        print(f"✅ Loaded keywords: {metadata.get('total_groups', len(keywords))} categories")
        print(f"📊 Extraction method: {metadata.get('extraction_method', 'unknown')}")
        print(f"🎯 Min frequency: {metadata.get('min_frequency', 'unknown')}")
    except Exception as e:
        print(f"❌ Error loading keywords: {e}")
        return
    
    print("\n" + "=" * 40)
    print("🕌 Available Islamic Categories:")
    print("=" * 40)
    
    # Show categories with counts
    for category, terms in sorted(keywords.items()):
        if len(terms) > 5:  # Only show substantial categories
            print(f"  • {category}: {len(terms)} terms")
    
    print("\n" + "=" * 40)
    print("🔍 Query Testing:")
    print("=" * 40)
    
    # Test queries
    test_queries = [
        "shalat",
        "puasa ramadhan", 
        "hukum riba",
        "nikah",
        "zakat fitrah"
    ]
    
    for query in test_queries:
        print(f"\n🔹 Query: '{query}'")
        
        # Find related keywords
        matches = search_keywords(keywords, query)
        if matches:
            for category, terms in matches.items():
                print(f"   📂 {category}: {terms[:3]}...")
        else:
            print("   ❌ No matches found")
        
        # Show enhanced query
        enhanced = enhance_query(keywords, query)
        if enhanced:
            print(f"   🚀 Enhanced: {enhanced[:5]}")
    
    print("\n" + "=" * 40)
    print("💡 Usage Examples:")
    print("=" * 40)
    
    print("""
# Load keywords in your application:
keywords, metadata = load_keywords()

# Enhance search queries:
enhanced_terms = enhance_query(keywords, "hukum puasa")

# Find related concepts:
shalat_terms = search_keywords(keywords, "shalat")

# Use for semantic indexing:
for category, terms in keywords.items():
    index_terms_for_search(category, terms)
    """)

if __name__ == "__main__":
    main() 