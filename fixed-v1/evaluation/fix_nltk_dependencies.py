#!/usr/bin/env python3
"""
Fix NLTK Dependencies - Fixed V1
===============================

Script untuk mengatasi masalah NLTK dependencies dan download semua data yang diperlukan.

Usage:
    python fix_nltk_dependencies.py

Author: Hadith AI Team - Fixed V1
Date: 2024
"""

import sys

def fix_nltk_dependencies():
    """Download dan setup semua NLTK dependencies yang diperlukan."""
    
    print("🔧 Fixing NLTK Dependencies...")
    print("=" * 50)
    
    # Import NLTK
    try:
        import nltk
        print("✅ NLTK imported successfully")
    except ImportError:
        print("❌ NLTK not installed. Install with: pip install nltk")
        return False
    
    # List of required NLTK data
    required_data = [
        'punkt',
        'punkt_tab',
        'stopwords',
        'wordnet',
        'averaged_perceptron_tagger'
    ]
    
    successful_downloads = 0
    
    for data_name in required_data:
        print(f"📥 Downloading {data_name}...")
        try:
            # Try to find the data first
            if data_name == 'punkt':
                try:
                    nltk.data.find('tokenizers/punkt')
                    print(f"   ✅ {data_name} already available")
                    successful_downloads += 1
                    continue
                except LookupError:
                    pass
            elif data_name == 'punkt_tab':
                try:
                    nltk.data.find('tokenizers/punkt_tab')
                    print(f"   ✅ {data_name} already available")
                    successful_downloads += 1
                    continue
                except LookupError:
                    pass
            
            # Download the data
            result = nltk.download(data_name, quiet=False)
            if result:
                print(f"   ✅ {data_name} downloaded successfully")
                successful_downloads += 1
            else:
                print(f"   ⚠️  {data_name} download may have failed")
                
        except Exception as e:
            print(f"   ❌ Failed to download {data_name}: {e}")
    
    print()
    print(f"📊 Summary: {successful_downloads}/{len(required_data)} NLTK data packages available")
    
    return successful_downloads >= 2  # At least punkt and punkt_tab


def test_nltk_functionality():
    """Test NLTK functionality after setup."""
    
    print("\n🧪 Testing NLTK Functionality...")
    print("=" * 50)
    
    try:
        import nltk
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        
        # Test tokenization
        test_text = "Ini adalah test tokenisasi NLTK untuk bahasa Indonesia."
        
        try:
            tokens = nltk.word_tokenize(test_text)
            print(f"✅ Tokenization successful: {len(tokens)} tokens")
            print(f"   Sample tokens: {tokens[:5]}")
        except Exception as e:
            print(f"❌ Tokenization failed: {e}")
            return False
        
        # Test BLEU calculation
        try:
            reference = ["ini", "adalah", "test", "referensi"]
            candidate = ["ini", "adalah", "test", "kandidat"]
            
            smoothing = SmoothingFunction().method1
            score = sentence_bleu([reference], candidate, smoothing_function=smoothing)
            
            print(f"✅ BLEU calculation successful: {score:.4f}")
        except Exception as e:
            print(f"❌ BLEU calculation failed: {e}")
            return False
        
        print("✅ All NLTK functionality tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ NLTK import failed: {e}")
        return False


def test_rouge_functionality():
    """Test ROUGE functionality."""
    
    print("\n📊 Testing ROUGE Functionality...")
    print("=" * 50)
    
    try:
        from rouge import Rouge
        
        rouge = Rouge()
        
        # Test ROUGE calculation
        reference = "Ini adalah teks referensi untuk testing ROUGE."
        hypothesis = "Ini merupakan teks hipotesis untuk pengujian ROUGE."
        
        scores = rouge.get_scores(hypothesis, reference)
        
        print("✅ ROUGE calculation successful:")
        print(f"   ROUGE-1: {scores[0]['rouge-1']['f']:.4f}")
        print(f"   ROUGE-2: {scores[0]['rouge-2']['f']:.4f}")
        print(f"   ROUGE-L: {scores[0]['rouge-l']['f']:.4f}")
        
        return True
        
    except ImportError:
        print("❌ ROUGE library not available. Install with: pip install rouge")
        return False
    except Exception as e:
        print(f"❌ ROUGE calculation failed: {e}")
        return False


def main():
    """Main function."""
    
    print("🔧 NLTK Dependencies Fix Tool - Fixed V1")
    print("=" * 60)
    
    # Fix NLTK dependencies
    nltk_success = fix_nltk_dependencies()
    
    # Test functionality
    if nltk_success:
        nltk_test = test_nltk_functionality()
        rouge_test = test_rouge_functionality()
        
        print("\n📋 Final Summary")
        print("=" * 60)
        print(f"NLTK Setup: {'✅ SUCCESS' if nltk_success else '❌ FAILED'}")
        print(f"NLTK Functionality: {'✅ SUCCESS' if nltk_test else '❌ FAILED'}")
        print(f"ROUGE Functionality: {'✅ SUCCESS' if rouge_test else '❌ FAILED'}")
        
        if nltk_success and nltk_test:
            print("\n🎉 NLTK setup completed successfully!")
            print("💡 You can now run the evaluation system:")
            print("   python run_evaluation.py --generate-report")
        else:
            print("\n⚠️  Some issues remain. Check the errors above.")
            
        if not rouge_test:
            print("\n🔧 To fix ROUGE:")
            print("   pip install rouge")
        
        return nltk_success and nltk_test
    else:
        print("\n❌ NLTK setup failed. Please install NLTK first:")
        print("   pip install nltk")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
