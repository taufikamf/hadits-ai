"""
Enhanced Retrieval Evaluator for Hadith AI
==========================================

Enhanced evaluation system to test the complete retrieval workflow with:
- Enhanced keyword extraction integration
- Auto-adaptive keyword filtering  
- Comprehensive result analysis
- Detailed performance metrics

Author: Hadith AI Team  
Date: 2024
"""

import sys
import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
from utils import query_optimizer
from retriever import query_runner

load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

OUTPUT_PATH = "retriever/enhanced_retrieval_results.json"
ENHANCED_BATCH_QUERIES = [
    "apa hukum riba?",
    "apa itu zakat fitrah?", 
    "bagaimana cara shalat malam?",
    "apa hukum minuman keras?",
    "berikan hadis tentang mati syahid",
    "berikan hadis tentang shalat jum'at", 
    "berikan hadis tentang perang",
    "berikan hadis tentang sedekah",
    "berikan hadits tentang shalat dhuhur",
    "bagaimana cara berwudhu yang benar?",
    "apa hukum jual beli dalam islam?",
    "berikan hadis tentang puasa ramadan",
    "bagaimana hukum nikah dalam islam?",
    "apa yang dimaksud dengan jihad?",
    "berikan hadis tentang berbakti kepada orang tua"
]

def run_enhanced_batch_evaluation():
    """
    Run enhanced batch evaluation with the improved retrieval system.
    Tests auto-adaptive keyword filtering and enhanced scoring.
    """
    logger.info("🚀 Starting Enhanced Hadith Retrieval Evaluation")
    logger.info(f"📊 Testing {len(ENHANCED_BATCH_QUERIES)} queries")
    
    all_results = []
    summary_stats = {
        "total_queries": 0,
        "successful_queries": 0,
        "total_results": 0,
        "average_processing_time": 0,
        "keyword_extraction_stats": {},
        "retrieval_performance": {}
    }
    
    processing_times = []

    for i, query in enumerate(ENHANCED_BATCH_QUERIES):
        logger.info(f"\n📋 [{i+1}/{len(ENHANCED_BATCH_QUERIES)}] Processing: '{query}'")
        logger.info("-" * 60)
        
        start_time = time.time()
        
        try:
            # Step 1: Get enhanced optimization with keywords
            optimized_query, required_keywords = query_optimizer.optimize_query(query, return_keywords=True)
            
            logger.info(f"🔍 Enhanced Keywords Extracted: {len(required_keywords)} keywords")
            logger.info(f"📝 Keywords: {required_keywords}")
            
            # Step 2: Query with enhanced system (auto min_match)
            results = query_runner.query_hadits_return(
                raw_query=query,
                optimized_query=optimized_query, 
                top_k=5,
                required_keywords=required_keywords,
                min_match=None,  # Let system auto-determine
                apply_literal_boost=True,
                boost_factor=0.2
            )
            
            processing_time = time.time() - start_time
            processing_times.append(processing_time)
            
            logger.info(f"✅ Retrieved: {len(results)} results in {processing_time:.2f}s")
            
            # Analyze results quality
            quality_metrics = analyze_result_quality(results, query, required_keywords)
            
            # Store detailed result
            query_result = {
                "query_id": i + 1,
                "original_query": query,
                "optimized_query": optimized_query,
                "required_keywords": required_keywords,
                "keywords_count": len(required_keywords),
                "results_count": len(results),
                "processing_time_seconds": processing_time,
                "quality_metrics": quality_metrics,
                "results": results,
                "timestamp": datetime.now().isoformat()
            }
            
            all_results.append(query_result)
            
            # Update summary stats
            summary_stats["successful_queries"] += 1
            summary_stats["total_results"] += len(results)
            
            # Track keyword extraction patterns
            keywords_key = f"{len(required_keywords)}_keywords"
            if keywords_key not in summary_stats["keyword_extraction_stats"]:
                summary_stats["keyword_extraction_stats"][keywords_key] = 0
            summary_stats["keyword_extraction_stats"][keywords_key] += 1
            
            logger.info(f"🎯 Quality Score: {quality_metrics.get('overall_quality', 0):.2f}/10")
            
        except Exception as e:
            logger.error(f"❌ Query failed: {e}")
            
            error_result = {
                "query_id": i + 1,
                "original_query": query,
                "error": str(e),
                "processing_time_seconds": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }
            all_results.append(error_result)

    # Finalize summary statistics
    summary_stats["total_queries"] = len(ENHANCED_BATCH_QUERIES)
    summary_stats["average_processing_time"] = sum(processing_times) / len(processing_times) if processing_times else 0
    summary_stats["success_rate"] = (summary_stats["successful_queries"] / summary_stats["total_queries"]) * 100
    
    # Calculate retrieval performance metrics
    if summary_stats["successful_queries"] > 0:
        summary_stats["retrieval_performance"] = {
            "avg_results_per_query": summary_stats["total_results"] / summary_stats["successful_queries"],
            "queries_with_results": sum(1 for r in all_results if r.get("results_count", 0) > 0),
            "zero_result_queries": sum(1 for r in all_results if r.get("results_count", 0) == 0)
        }

    # Compile final output
    evaluation_output = {
        "evaluation_metadata": {
            "evaluation_date": datetime.now().isoformat(),
            "system_version": "enhanced_v1.0",
            "total_queries_tested": len(ENHANCED_BATCH_QUERIES),
            "evaluation_type": "enhanced_batch_evaluation"
        },
        "summary_statistics": summary_stats,
        "detailed_results": all_results
    }

    # Save results to file
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(evaluation_output, f, indent=2, ensure_ascii=False)

    # Print final summary
    logger.info("\n" + "="*70)
    logger.info("📊 ENHANCED RETRIEVAL EVALUATION SUMMARY")
    logger.info("="*70)
    logger.info(f"✅ Success Rate: {summary_stats['success_rate']:.1f}% ({summary_stats['successful_queries']}/{summary_stats['total_queries']})")
    logger.info(f"⏱️  Average Processing Time: {summary_stats['average_processing_time']:.3f}s")
    logger.info(f"📈 Average Results per Query: {summary_stats['retrieval_performance'].get('avg_results_per_query', 0):.2f}")
    logger.info(f"🎯 Queries with Results: {summary_stats['retrieval_performance'].get('queries_with_results', 0)}")
    logger.info(f"⚠️  Zero Result Queries: {summary_stats['retrieval_performance'].get('zero_result_queries', 0)}")
    
    logger.info(f"\n📄 Detailed results saved to: {OUTPUT_PATH}")
    logger.info("="*70)

    return evaluation_output

def analyze_result_quality(results, original_query, keywords):
    """
    Analyze the quality of retrieval results.
    
    Args:
        results: List of result documents
        original_query: Original user query  
        keywords: Extracted keywords
        
    Returns:
        Dictionary with quality metrics
    """
    if not results:
        return {"overall_quality": 0, "metrics": {}}
    
    metrics = {}
    
    # 1. Keyword coverage in results
    if keywords:
        keyword_coverage_scores = []
        for result in results:
            terjemah = result.get("terjemah", "").lower()
            matched_keywords = [kw for kw in keywords if kw.lower() in terjemah]
            coverage = len(matched_keywords) / len(keywords) if keywords else 0
            keyword_coverage_scores.append(coverage)
        
        metrics["avg_keyword_coverage"] = sum(keyword_coverage_scores) / len(keyword_coverage_scores)
    else:
        metrics["avg_keyword_coverage"] = 0
    
    # 2. Literal overlap with query
    literal_overlap_scores = [result.get("literal_overlap", 0) for result in results]
    metrics["avg_literal_overlap"] = sum(literal_overlap_scores) / len(literal_overlap_scores)
    
    # 3. Semantic coherence (based on scores)
    semantic_scores = [1 / (1 + result.get("score", 1)) for result in results]  # Convert distance to similarity
    metrics["avg_semantic_score"] = sum(semantic_scores) / len(semantic_scores)
    
    # 4. Result diversity (unique kitabs)
    unique_kitabs = len(set(result.get("kitab", "") for result in results))
    metrics["source_diversity"] = unique_kitabs / len(results) if results else 0
    
    # 5. Overall quality score (weighted combination)
    quality_components = [
        ("keyword_coverage", metrics["avg_keyword_coverage"], 0.3),
        ("literal_overlap", metrics["avg_literal_overlap"], 0.25),
        ("semantic_score", metrics["avg_semantic_score"], 0.25),
        ("source_diversity", metrics["source_diversity"], 0.2)
    ]
    
    overall_quality = sum(score * weight for _, score, weight in quality_components) * 10
    
    return {
        "overall_quality": min(overall_quality, 10),  # Cap at 10
        "metrics": metrics,
        "quality_components": dict((name, score) for name, score, _ in quality_components)
    }

def run_single_query_analysis(query: str, detailed: bool = True):
    """
    Run detailed analysis for a single query.
    
    Args:
        query: Query to analyze
        detailed: Whether to show detailed analysis
        
    Returns:
        Analysis results
    """
    logger.info(f"🔍 Analyzing single query: '{query}'")
    
    start_time = time.time()
    
    # Get enhanced optimization
    optimized_query, required_keywords = query_optimizer.optimize_query(query, return_keywords=True)
    
    # Query with enhanced system
    results = query_runner.query_hadits_return(
        raw_query=query,
        optimized_query=optimized_query,
        top_k=10,  # Get more for analysis
        required_keywords=required_keywords,
        min_match=None,
        apply_literal_boost=True
    )
    
    processing_time = time.time() - start_time
    quality_metrics = analyze_result_quality(results, query, required_keywords)
    
    analysis = {
        "query": query,
        "optimized_query": optimized_query,
        "keywords": required_keywords,
        "results_count": len(results),
        "processing_time": processing_time,
        "quality_metrics": quality_metrics,
        "results": results[:5] if not detailed else results  # Limit for display
    }
    
    if detailed:
        logger.info(f"📊 Query Analysis Results:")
        logger.info(f"   Keywords: {required_keywords}")
        logger.info(f"   Results: {len(results)}")
        logger.info(f"   Quality Score: {quality_metrics['overall_quality']:.2f}/10")
        logger.info(f"   Processing Time: {processing_time:.3f}s")
        
        if results:
            logger.info(f"   Top Result: {results[0]['kitab']} - {results[0]['terjemah'][:100]}...")
    
    return analysis

if __name__ == "__main__":
    print("Enhanced Hadith Retrieval Evaluator")
    print("===================================")
    
    import argparse
    parser = argparse.ArgumentParser(description="Enhanced Hadith Retrieval Evaluation")
    parser.add_argument("--mode", choices=["batch", "single"], default="batch",
                       help="Evaluation mode: batch or single query")
    parser.add_argument("--query", type=str, help="Single query to analyze (for single mode)")
    
    args = parser.parse_args()
    
    if args.mode == "batch":
        run_enhanced_batch_evaluation()
    elif args.mode == "single":
        if args.query:
            run_single_query_analysis(args.query, detailed=True)
        else:
            # Interactive mode
            while True:
                query = input("\n🔍 Enter query (or 'quit' to exit): ").strip()
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                if query:
                    run_single_query_analysis(query, detailed=True)