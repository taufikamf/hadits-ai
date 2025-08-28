#!/usr/bin/env python3
"""
Test Evaluation System - Fixed V1
=================================

Quick test script untuk memastikan evaluation system berfungsi dengan baik
sebelum menjalankan evaluasi lengkap.

Usage:
    python test_evaluation_system.py

Author: Hadith AI Team - Fixed V1
Date: 2024
"""

import os
import sys
import asyncio
import logging
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test if all required modules can be imported."""
    print("🧪 Testing Imports...")
    print("-" * 40)
    
    tests = [
        ("Core evaluation module", "rag_evaluation", "RAGEvaluator"),
        ("ROUGE metrics", "rouge", "Rouge"),
        ("NLTK", "nltk", None),
        ("Sentence Transformers", "sentence_transformers", "SentenceTransformer"),
        ("Scikit-learn", "sklearn", None),
        ("NumPy", "numpy", None),
    ]
    
    success_count = 0
    
    for test_name, module_name, class_name in tests:
        try:
            module = __import__(module_name)
            if class_name:
                getattr(module, class_name)
            print(f"✅ {test_name}")
            success_count += 1
        except ImportError as e:
            print(f"❌ {test_name}: {e}")
        except AttributeError as e:
            print(f"❌ {test_name}: {e}")
        except Exception as e:
            print(f"❌ {test_name}: {e}")
    
    print(f"\nImport Tests: {success_count}/{len(tests)} passed")
    return success_count == len(tests)


def test_nltk_data():
    """Test if NLTK data is available."""
    print("\n📚 Testing NLTK Data...")
    print("-" * 40)
    
    try:
        import nltk
        
        # Check required NLTK data
        required_data = ['punkt', 'punkt_tab']
        available_count = 0
        
        for data_name in required_data:
            try:
                if data_name == 'punkt':
                    nltk.data.find('tokenizers/punkt')
                    print(f"✅ NLTK {data_name} available")
                    available_count += 1
                elif data_name == 'punkt_tab':
                    nltk.data.find('tokenizers/punkt_tab')
                    print(f"✅ NLTK {data_name} available")
                    available_count += 1
            except LookupError:
                print(f"❌ NLTK {data_name} missing")
        
        if available_count == 0:
            print("💡 Run: python fix_nltk_dependencies.py")
            return False
        elif available_count < len(required_data):
            print("⚠️  Some NLTK data missing but basic functionality may work")
            return True
        else:
            print("✅ All required NLTK data available")
            return True
            
    except ImportError:
        print("❌ NLTK not available")
        print("💡 Run: pip install nltk")
        return False


def test_data_files():
    """Test if required data files exist."""
    print("\n📁 Testing Data Files...")
    print("-" * 40)
    
    required_files = [
        "../../data/processed/hadits_docs.json",
        "../../data/enhanced_index_v1/enhanced_keywords_map_v1.json",
        "../../data/enhanced_index_v1/enhanced_embeddings_v1.pkl",
        "../../data/enhanced_index_v1/enhanced_faiss_index_v1.index",
        "../../data/enhanced_index_v1/enhanced_metadata_v1.pkl"
    ]
    
    success_count = 0
    
    for file_path in required_files:
        abs_path = Path(__file__).parent / file_path
        if abs_path.exists():
            size = abs_path.stat().st_size
            print(f"✅ {file_path} ({size:,} bytes)")
            success_count += 1
        else:
            print(f"❌ {file_path} (missing)")
    
    print(f"\nData Files: {success_count}/{len(required_files)} found")
    return success_count == len(required_files)


def test_ground_truth_dataset():
    """Test ground truth dataset loading."""
    print("\n📊 Testing Ground Truth Dataset...")
    print("-" * 40)
    
    try:
        from rag_evaluation import RAGEvaluator
        
        evaluator = RAGEvaluator()
        ground_truth = evaluator.ground_truth
        
        print(f"✅ Ground truth loaded: {len(ground_truth)} queries")
        
        # Test first item
        if ground_truth:
            first_item = ground_truth[0]
            print(f"✅ Sample query: '{first_item.query}'")
            print(f"✅ Sample topic: '{first_item.topic}'")
            print(f"✅ Reference length: {len(first_item.reference_answer)} chars")
        
        # Test categories
        categories = set(item.topic for item in ground_truth)
        print(f"✅ Categories: {', '.join(categories)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Ground truth test failed: {e}")
        return False


def test_metric_calculations():
    """Test metric calculation functions."""
    print("\n🔢 Testing Metric Calculations...")
    print("-" * 40)
    
    try:
        from rag_evaluation import RAGEvaluator
        
        evaluator = RAGEvaluator()
        
        # Test data
        reference = "Wudhu adalah bersuci dengan air untuk menghilangkan hadats kecil sebelum shalat."
        generated = "Wudhu merupakan cara bersuci menggunakan air untuk membersihkan hadats kecil sebelum melaksanakan shalat."
        
        # Test ROUGE
        rouge_scores = evaluator.calculate_rouge_scores(reference, generated)
        print(f"✅ ROUGE calculated: {rouge_scores}")
        
        # Test BLEU
        bleu_score = evaluator.calculate_bleu_score(reference, generated)
        if bleu_score > 0:
            print(f"✅ BLEU calculated: {bleu_score:.4f}")
        else:
            print(f"⚠️  BLEU calculation returned 0 (may use fallback method)")
        
        
        # Test Semantic Similarity
        semantic_score = evaluator.calculate_semantic_similarity(reference, generated)
        print(f"✅ Semantic similarity calculated: {semantic_score:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Metric calculation test failed: {e}")
        return False


async def test_service_initialization():
    """Test service initialization."""
    print("\n🚀 Testing Service Initialization...")
    print("-" * 40)
    
    try:
        from rag_evaluation import RAGEvaluator
        
        evaluator = RAGEvaluator()
        await evaluator.initialize_services()
        
        # Check services
        if evaluator.rag_service:
            print("✅ RAG service initialized")
        else:
            print("❌ RAG service initialization failed")
            return False
            
        if evaluator.baseline_service:
            print("✅ Baseline service initialized")
        else:
            print("❌ Baseline service initialization failed")
            return False
        
        print("✅ Services initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Service initialization test failed: {e}")
        return False


async def test_single_evaluation():
    """Test single query evaluation."""
    print("\n🎯 Testing Single Query Evaluation...")
    print("-" * 40)
    
    try:
        from rag_evaluation import RAGEvaluator
        
        evaluator = RAGEvaluator()
        await evaluator.initialize_services()
        
        # Get first ground truth item
        gt_item = evaluator.ground_truth[0]
        print(f"Testing query: '{gt_item.query}'")
        
        # Test RAG evaluation
        rag_result = await evaluator.evaluate_single_query(gt_item, use_rag=True)
        
        if rag_result.success:
            print("✅ RAG evaluation successful")
            print(f"   ROUGE-1: {rag_result.rouge_scores.get('rouge-1', 0):.3f}")
            print(f"   BLEU: {rag_result.bleu_score:.3f}")
            print(f"   Semantic: {rag_result.semantic_similarity:.3f}")
            print(f"   Response time: {rag_result.response_time_ms:.1f}ms")
        else:
            print(f"❌ RAG evaluation failed: {rag_result.error}")
            return False
        
        # Test baseline evaluation
        baseline_result = await evaluator.evaluate_single_query(gt_item, use_rag=False)
        
        if baseline_result.success:
            print("✅ Baseline evaluation successful")
            print(f"   ROUGE-1: {baseline_result.rouge_scores.get('rouge-1', 0):.3f}")
            print(f"   BLEU: {baseline_result.bleu_score:.3f}")
            print(f"   Semantic: {baseline_result.semantic_similarity:.3f}")
        else:
            print(f"❌ Baseline evaluation failed: {baseline_result.error}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Single evaluation test failed: {e}")
        return False


def test_output_directory():
    """Test output directory creation."""
    print("\n📁 Testing Output Directory...")
    print("-" * 40)
    
    try:
        test_dir = Path("test_evaluation_output")
        test_dir.mkdir(exist_ok=True)
        
        # Test file creation
        test_file = test_dir / "test.txt"
        test_file.write_text("Test content")
        
        if test_file.exists():
            print("✅ Output directory writable")
            
            # Cleanup
            test_file.unlink()
            test_dir.rmdir()
            print("✅ Cleanup successful")
            
            return True
        else:
            print("❌ Cannot write to output directory")
            return False
            
    except Exception as e:
        print(f"❌ Output directory test failed: {e}")
        return False


async def main():
    """Run all tests."""
    print("🧪 RAG Evaluation System - Test Suite")
    print("=" * 60)
    
    tests = [
        ("Import Tests", test_imports),
        ("NLTK Data", test_nltk_data),
        ("Data Files", test_data_files),
        ("Ground Truth Dataset", test_ground_truth_dataset),
        ("Metric Calculations", test_metric_calculations),
        ("Service Initialization", test_service_initialization),
        ("Single Evaluation", test_single_evaluation),
        ("Output Directory", test_output_directory),
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
            print(f"❌ {test_name} failed with exception: {e}")
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
        print("\n🎉 All tests passed! Evaluation system is ready.")
        print("\n💡 Next steps:")
        print("   1. Run full evaluation: python run_evaluation.py --generate-report")
        print("   2. Check results in evaluation_results/ directory")
        print("   3. Use generated reports for thesis subbab 5.3")
    else:
        failed_tests = [name for name, result in results.items() if not result]
        print(f"\n⚠️  {total-passed} test(s) failed: {', '.join(failed_tests)}")
        print("\n🔧 Troubleshooting:")
        
        if not results.get("Import Tests", True):
            print("   - Install missing dependencies: pip install -r requirements.txt")
        
        if not results.get("NLTK Data", True):
            print("   - Download NLTK data: python -c \"import nltk; nltk.download('punkt')\"")
        
        if not results.get("Data Files", True):
            print("   - Run indexing pipeline to generate required data files")
            print("   - Check: python indexing/enhanced_indexing_pipeline.py")
        
        if not results.get("Service Initialization", True):
            print("   - Ensure RAG system components are properly installed")
            print("   - Check service configuration and dependencies")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
