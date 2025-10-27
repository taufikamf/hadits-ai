#!/usr/bin/env python3
"""
RAG Evaluation System - Fixed V1
===============================

Comprehensive evaluation system for Retrieval-Augmented Generation (RAG) using:
- ROUGE metrics (ROUGE-1, ROUGE-2, ROUGE-L)
- BLEU score
- Semantic Similarity (Cosine Similarity)
- Baseline comparison

Features:
- Ground truth dataset evaluation
- Multiple evaluation metrics
- Statistical analysis
- Baseline comparison
- Detailed reporting

Author: Hadith AI Team - Fixed V1
Date: 2024
"""

import os
import sys
import json
import time
import asyncio
import logging
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Evaluation metrics libraries
try:
    from rouge import Rouge
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("⚠️  Warning: rouge library not available. Install with: pip install rouge")

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    # Download required NLTK data
    required_nltk_data = ['punkt', 'punkt_tab']
    for resource in required_nltk_data:
        try:
            if resource == 'punkt':
                nltk.data.find('tokenizers/punkt')
            elif resource == 'punkt_tab':
                nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            try:
                print(f"⬇️  Downloading NLTK {resource}...")
                nltk.download(resource, quiet=True)
            except Exception as e:
                print(f"⚠️  Warning: Failed to download NLTK {resource}: {e}")
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False
    print("⚠️  Warning: nltk library not available. Install with: pip install nltk")

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False
    print("⚠️  Warning: sentence-transformers not available. Install with: pip install sentence-transformers scikit-learn")

# Import RAG system components
from service.hadith_ai_service import HadithAIService, ServiceConfig
from generation.enhanced_response_generator import GenerationConfig, LLMProvider, ResponseMode

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GroundTruthItem:
    """Ground truth item for evaluation."""
    query: str
    reference_answer: str
    topic: str
    difficulty: str = "medium"
    expected_hadits: List[str] = field(default_factory=list)


@dataclass
class EvaluationResult:
    """Individual evaluation result."""
    query: str
    generated_answer: str
    reference_answer: str
    rouge_scores: Dict[str, float] = field(default_factory=dict)
    bleu_score: float = 0.0
    semantic_similarity: float = 0.0
    response_time_ms: float = 0.0
    num_retrieved_hadits: int = 0
    success: bool = True
    error: str = ""


@dataclass
class EvaluationSummary:
    """Overall evaluation summary."""
    total_queries: int
    successful_queries: int
    avg_rouge_1: float
    avg_rouge_2: float
    avg_rouge_l: float
    avg_bleu_score: float
    avg_semantic_similarity: float
    avg_response_time_ms: float
    results: List[EvaluationResult] = field(default_factory=list)


class RAGEvaluator:
    """
    Comprehensive RAG evaluation system with multiple metrics.
    """
    
    def __init__(self, service_config: ServiceConfig = None):
        """
        Initialize RAG evaluator.
        
        Args:
            service_config (ServiceConfig): RAG service configuration
        """
        self.service_config = service_config or self._create_default_config()
        self.rag_service = None
        self.baseline_service = None
        
        # Initialize evaluation tools
        self.rouge = Rouge() if ROUGE_AVAILABLE else None
        self.semantic_model = None
        
        if SEMANTIC_AVAILABLE:
            try:
                self.semantic_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                logger.info("Semantic similarity model loaded")
            except Exception as e:
                logger.warning(f"Failed to load semantic model: {e}")
                self.semantic_model = None
        
        # Ground truth dataset
        self.ground_truth = self._load_ground_truth()
        
        logger.info(f"RAG Evaluator initialized with {len(self.ground_truth)} test cases")
    
    def _create_default_config(self) -> ServiceConfig:
        """Create default configuration for evaluation."""
        return ServiceConfig(
            enable_llm_generation=True,
            fallback_to_simple=True,
            max_results_display=5,
            enable_sessions=False,
            enable_analytics=False,
            generation_config=GenerationConfig(
                llm_provider=LLMProvider.NONE,  # Use fallback for consistent evaluation
                response_mode=ResponseMode.COMPREHENSIVE
            )
        )
    
    def _load_ground_truth(self) -> List[GroundTruthItem]:
        """Load ground truth dataset for evaluation."""
        
        # Ground truth dataset based on user's requirements
        ground_truth_data = [
            {
                "query": "cara berwudhu yang benar",
                "reference_answer": "Wudhu adalah bersuci dengan air untuk menghilangkan hadats kecil sebelum shalat. Urutan wudhu yang benar menurut hadits adalah: 1) Membaca basmalah, 2) Mencuci kedua telapak tangan tiga kali, 3) Berkumur-kumur tiga kali, 4) Membersihkan hidung (istinsyaq) tiga kali, 5) Mencuci muka tiga kali, 6) Mencuci kedua tangan hingga siku tiga kali, 7) Mengusap kepala satu kali, 8) Mengusap kedua telinga, 9) Mencuci kedua kaki hingga mata kaki tiga kali. Rasulullah SAW bersabda tentang pentingnya wudhu yang sempurna.",
                "topic": "Thaharah",
                "difficulty": "easy",
                "expected_hadits": ["wudhu", "bersuci", "thaharah"]
            },
            {
                "query": "hukum zakat fitrah",
                "reference_answer": "Zakat fitrah adalah zakat yang wajib dikeluarkan setiap Muslim pada bulan Ramadan sebelum shalat Ied. Hukumnya wajib bagi setiap Muslim yang memiliki kelebihan makanan untuk satu hari satu malam. Zakat fitrah bertujuan untuk menyucikan jiwa dari dosa-dosa kecil dan membantu fakir miskin agar dapat merayakan Ied dengan gembira. Kadarnya adalah satu sha' (sekitar 2,5 kg) makanan pokok seperti beras, gandum, atau kurma.",
                "topic": "Zakat",
                "difficulty": "medium",
                "expected_hadits": ["zakat", "fitrah", "ramadan"]
            },
            {
                "query": "keutamaan shalat berjamaah",
                "reference_answer": "Shalat berjamaah memiliki keutamaan yang sangat besar dalam Islam. Rasulullah SAW bersabda bahwa shalat berjamaah lebih utama daripada shalat sendirian dengan 27 derajat. Shalat berjamaah menimbulkan persatuan dan kesatuan umat, melatih kedisiplinan, dan memperkuat ikatan sosial antar Muslim. Allah SWT memberikan pahala yang berlipat ganda bagi mereka yang mengerjakan shalat berjamaah di masjid.",
                "topic": "Shalat",
                "difficulty": "easy",
                "expected_hadits": ["shalat", "jamaah", "masjid"]
            },
            {
                "query": "hukum riba",
                "reference_answer": "Riba adalah haram dalam Islam berdasarkan Al-Quran dan hadits. Riba adalah penambahan dalam utang piutang atau jual beli yang tidak diimbangi dengan imbalan atau nilai yang setara. Islam melarang riba karena dapat menimbulkan kezaliman, eksploitasi ekonomi, dan ketidakadilan sosial. Rasulullah SAW mengutuk pemakan riba, pemberi riba, pencatat riba, dan saksi-saksinya. Allah SWT memerintahkan untuk meninggalkan sisa riba dan bertobat.",
                "topic": "Muamalah",
                "difficulty": "hard",
                "expected_hadits": ["riba", "haram", "jual beli"]
            },
            {
                "query": "adab makan menurut rasulullah",
                "reference_answer": "Rasulullah SAW mengajarkan adab makan yang mulia, yaitu: 1) Membaca basmalah sebelum makan, 2) Makan dengan tangan kanan, 3) Makan dari yang terdekat, 4) Tidak berlebih-lebihan dalam makan, 5) Membaca hamdalah setelah selesai makan, 6) Makan sambil duduk, bukan berdiri, 7) Tidak mencela makanan, 8) Berbagi makanan dengan orang lain. Nabi SAW bersabda bahwa perut adalah seburuk-buruk tempat yang diisi manusia.",
                "topic": "Akhlak",
                "difficulty": "easy",
                "expected_hadits": ["makan", "adab", "akhlak"]
            },
            {
                "query": "hukum puasa ramadan",
                "reference_answer": "Puasa Ramadan adalah salah satu rukun Islam yang wajib dilaksanakan oleh setiap Muslim yang sudah baligh, berakal sehat, dan mampu. Puasa dilaksanakan selama sebulan penuh dari terbit fajar hingga terbenam matahari. Puasa bukan hanya menahan lapar dan dahaga, tetapi juga menahan diri dari perbuatan maksiat. Puasa memiliki hikmah untuk melatih ketakwaan, empati kepada yang kurang mampu, dan mengendalikan hawa nafsu.",
                "topic": "Ibadah",
                "difficulty": "easy",
                "expected_hadits": ["puasa", "ramadan", "rukun islam"]
            },
            {
                "query": "kewajiban berbakti kepada orang tua",
                "reference_answer": "Berbakti kepada orang tua (birrul walidain) adalah kewajiban yang sangat ditekankan dalam Islam. Allah SWT memerintahkan untuk berbakti kepada kedua orang tua setelah beribadah kepada-Nya. Rasulullah SAW bersabda bahwa ridha Allah tergantung pada ridha orang tua, dan murka Allah tergantung pada murka orang tua. Berbakti kepada orang tua meliputi berbuat baik, berkata lemah lembut, mendoakan, dan merawat mereka di masa tua.",
                "topic": "Akhlak",
                "difficulty": "medium",
                "expected_hadits": ["birr", "walidain", "orang tua"]
            },
            {
                "query": "hukum sholat jumat",
                "reference_answer": "Shalat Jumat adalah kewajiban bagi laki-laki Muslim yang sudah baligh, berakal sehat, merdeka, mukim (bukan musafir), dan tidak ada halangan syar'i. Shalat Jumat menggantikan shalat Dzuhur dan dilaksanakan secara berjamaah dengan syarat minimal 40 orang menurut sebagian ulama. Sebelum shalat didahului dengan khutbah yang merupakan bagian integral dari ibadah Jumat. Allah SWT memerintahkan untuk meninggalkan jual beli ketika panggilan shalat Jumat dikumandangkan.",
                "topic": "Shalat",
                "difficulty": "medium",
                "expected_hadits": ["jumat", "khutbah", "jamaah"]
            },
            {
                "query": "hukum jual beli dalam islam",
                "reference_answer": "Jual beli dalam Islam pada dasarnya adalah halal dan dianjurkan sebagai cara mencari rezeki yang baik. Islam mengatur prinsip-prinsip jual beli yang adil: 1) Kerelaan kedua belah pihak (antaradhin), 2) Objek yang diperjualbelikan harus halal dan bermanfaat, 3) Harga dan barang harus jelas, 4) Tidak ada unsur penipuan (gharar), 5) Tidak ada unsur riba. Rasulullah SAW bersabda bahwa pedagang yang jujur akan bersama para nabi dan syuhada di akhirat.",
                "topic": "Muamalah",
                "difficulty": "medium",
                "expected_hadits": ["jual beli", "muamalah", "perdagangan"]
            },
            {
                "query": "keutamaan membaca quran",
                "reference_answer": "Membaca Al-Quran memiliki keutamaan yang sangat besar dalam Islam. Rasulullah SAW bersabda bahwa setiap huruf yang dibaca dari Al-Quran mendapat pahala sepuluh kebaikan. Al-Quran adalah syafaat bagi pembacanya di hari kiamat. Membaca Al-Quran dengan tartil (pelan dan jelas) akan mendapat pahala berlipat ganda. Al-Quran juga merupakan obat bagi penyakit hati dan jiwa. Orang yang mahir membaca Al-Quran akan bersama para malaikat yang mulia.",
                "topic": "Tilawah",
                "difficulty": "easy",
                "expected_hadits": ["quran", "tilawah", "pahala"]
            }
        ]
        
        return [GroundTruthItem(**item) for item in ground_truth_data]
    
    async def initialize_services(self):
        """Initialize RAG and baseline services."""
        logger.info("Initializing RAG services for evaluation...")
        
        try:
            # Use evaluation-specific configuration
            from evaluation_config import create_evaluation_config
            
            rag_config, _ = create_evaluation_config()
            
            # Override some settings for evaluation
            rag_config.enable_llm_generation = True
            rag_config.fallback_to_simple = True
            rag_config.enable_query_logging = False
            rag_config.debug_mode = False
            
            self.rag_service = HadithAIService(rag_config)
            
            # Initialize baseline service (retrieval-only)
            baseline_config, _ = create_evaluation_config()
            baseline_config.enable_llm_generation = False
            baseline_config.fallback_to_simple = True
            baseline_config.enable_query_logging = False
            baseline_config.debug_mode = False
            
            self.baseline_service = HadithAIService(baseline_config)
            
            logger.info("Services initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize services: {e}")
            # Fallback to simple config if evaluation config fails
            logger.info("Attempting fallback initialization...")
            
            rag_config = ServiceConfig(
                enable_llm_generation=False,  # Disable LLM for fallback
                fallback_to_simple=True,
                max_results_display=5,
                enable_sessions=False,
                enable_analytics=False
            )
            self.rag_service = HadithAIService(rag_config)
            self.baseline_service = self.rag_service  # Use same service for both
            
            logger.info("Fallback initialization completed")
    
    def calculate_rouge_scores(self, reference: str, generated: str) -> Dict[str, float]:
        """
        Calculate ROUGE scores between reference and generated text.
        
        Args:
            reference (str): Reference text
            generated (str): Generated text
            
        Returns:
            Dict[str, float]: ROUGE scores
        """
        if not ROUGE_AVAILABLE or not reference.strip() or not generated.strip():
            return {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0}
        
        try:
            scores = self.rouge.get_scores(generated, reference, avg=True)
            return {
                "rouge-1": scores['rouge-1']['f'],
                "rouge-2": scores['rouge-2']['f'], 
                "rouge-l": scores['rouge-l']['f']
            }
        except Exception as e:
            logger.warning(f"ROUGE calculation failed: {e}")
            return {"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0}
    
    def calculate_bleu_score(self, reference: str, generated: str) -> float:
        """
        Calculate BLEU score between reference and generated text.
        
        Args:
            reference (str): Reference text
            generated (str): Generated text
            
        Returns:
            float: BLEU score
        """
        if not BLEU_AVAILABLE or not reference.strip() or not generated.strip():
            return 0.0
        
        try:
            import nltk
            
            # Try to tokenize with NLTK
            try:
                reference_tokens = nltk.word_tokenize(reference.lower())
                generated_tokens = nltk.word_tokenize(generated.lower())
            except Exception as e:
                # Fallback to simple tokenization if NLTK fails
                logger.warning(f"NLTK tokenization failed, using simple tokenization: {e}")
                reference_tokens = reference.lower().split()
                generated_tokens = generated.lower().split()
            
            # Use smoothing function to handle zero n-gram matches
            smoothing = SmoothingFunction().method1
            
            # Calculate BLEU score with weights for 1-gram to 4-gram
            score = sentence_bleu(
                [reference_tokens], 
                generated_tokens,
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smoothing
            )
            return score
        except Exception as e:
            logger.warning(f"BLEU calculation failed: {e}")
            # Fallback to simple word overlap if BLEU completely fails
            return self._calculate_simple_word_overlap(reference, generated)
    
    def _calculate_simple_word_overlap(self, reference: str, generated: str) -> float:
        """
        Fallback method: Calculate simple word overlap as BLEU substitute.
        
        Args:
            reference (str): Reference text
            generated (str): Generated text
            
        Returns:
            float: Simple overlap score (0-1)
        """
        try:
            ref_words = set(reference.lower().split())
            gen_words = set(generated.lower().split())
            
            if not ref_words:
                return 0.0
            
            overlap = len(ref_words.intersection(gen_words))
            return overlap / len(ref_words)
        except Exception:
            return 0.0
    
    def calculate_semantic_similarity(self, reference: str, generated: str) -> float:
        """
        Calculate semantic similarity using sentence embeddings.
        
        Args:
            reference (str): Reference text
            generated (str): Generated text
            
        Returns:
            float: Cosine similarity score (0-1)
        """
        if not self.semantic_model or not reference.strip() or not generated.strip():
            return 0.0
        
        try:
            # Generate embeddings
            embeddings = self.semantic_model.encode([reference, generated])
            
            # Calculate cosine similarity
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except Exception as e:
            logger.warning(f"Semantic similarity calculation failed: {e}")
            return 0.0
    
    async def evaluate_single_query(self, ground_truth_item: GroundTruthItem, 
                                  use_rag: bool = True) -> EvaluationResult:
        """
        Evaluate a single query against ground truth.
        
        Args:
            ground_truth_item (GroundTruthItem): Ground truth item
            use_rag (bool): Whether to use RAG or baseline service
            
        Returns:
            EvaluationResult: Evaluation result
        """
        service = self.rag_service if use_rag else self.baseline_service
        
        try:
            start_time = time.time()
            
            # Process query
            if use_rag:
                response = await service.process_query_async(
                    ground_truth_item.query, 
                    max_results=5
                )
            else:
                response = service.process_query(
                    ground_truth_item.query,
                    max_results=5
                )
            
            response_time = (time.time() - start_time) * 1000
            
            if not response.success:
                return EvaluationResult(
                    query=ground_truth_item.query,
                    generated_answer="",
                    reference_answer=ground_truth_item.reference_answer,
                    response_time_ms=response_time,
                    success=False,
                    error="Query processing failed"
                )
            
            # Extract generated answer
            generated_answer = response.message
            reference_answer = ground_truth_item.reference_answer
            
            # Calculate metrics
            rouge_scores = self.calculate_rouge_scores(reference_answer, generated_answer)
            bleu_score = self.calculate_bleu_score(reference_answer, generated_answer)
            semantic_similarity = self.calculate_semantic_similarity(reference_answer, generated_answer)
            
            return EvaluationResult(
                query=ground_truth_item.query,
                generated_answer=generated_answer,
                reference_answer=reference_answer,
                rouge_scores=rouge_scores,
                bleu_score=bleu_score,
                semantic_similarity=semantic_similarity,
                response_time_ms=response_time,
                num_retrieved_hadits=len(response.results),
                success=True
            )
            
        except Exception as e:
            logger.error(f"Error evaluating query '{ground_truth_item.query}': {e}")
            return EvaluationResult(
                query=ground_truth_item.query,
                generated_answer="",
                reference_answer=ground_truth_item.reference_answer,
                success=False,
                error=str(e)
            )
    
    async def evaluate_all(self, use_rag: bool = True) -> EvaluationSummary:
        """
        Evaluate all ground truth queries.
        
        Args:
            use_rag (bool): Whether to use RAG or baseline service
            
        Returns:
            EvaluationSummary: Complete evaluation summary
        """
        logger.info(f"Starting evaluation of {len(self.ground_truth)} queries...")
        logger.info(f"Using {'RAG' if use_rag else 'Baseline'} service")
        
        results = []
        
        for i, gt_item in enumerate(self.ground_truth, 1):
            logger.info(f"Evaluating query {i}/{len(self.ground_truth)}: {gt_item.query}")
            
            result = await self.evaluate_single_query(gt_item, use_rag)
            results.append(result)
            
            if result.success:
                logger.info(f"  ✅ Success - ROUGE-1: {result.rouge_scores.get('rouge-1', 0):.3f}, "
                          f"BLEU: {result.bleu_score:.3f}, Semantic: {result.semantic_similarity:.3f}")
            else:
                logger.warning(f"  ❌ Failed - {result.error}")
        
        # Calculate summary statistics
        successful_results = [r for r in results if r.success]
        total_queries = len(results)
        successful_queries = len(successful_results)
        
        if successful_results:
            avg_rouge_1 = np.mean([r.rouge_scores.get('rouge-1', 0) for r in successful_results])
            avg_rouge_2 = np.mean([r.rouge_scores.get('rouge-2', 0) for r in successful_results])
            avg_rouge_l = np.mean([r.rouge_scores.get('rouge-l', 0) for r in successful_results])
            avg_bleu_score = np.mean([r.bleu_score for r in successful_results])
            avg_semantic_similarity = np.mean([r.semantic_similarity for r in successful_results])
            avg_response_time = np.mean([r.response_time_ms for r in successful_results])
        else:
            avg_rouge_1 = avg_rouge_2 = avg_rouge_l = 0.0
            avg_bleu_score = avg_semantic_similarity = avg_response_time = 0.0
        
        summary = EvaluationSummary(
            total_queries=total_queries,
            successful_queries=successful_queries,
            avg_rouge_1=avg_rouge_1,
            avg_rouge_2=avg_rouge_2,
            avg_rouge_l=avg_rouge_l,
            avg_bleu_score=avg_bleu_score,
            avg_semantic_similarity=avg_semantic_similarity,
            avg_response_time_ms=avg_response_time,
            results=results
        )
        
        logger.info(f"Evaluation completed: {successful_queries}/{total_queries} successful")
        return summary
    
    def save_results(self, rag_summary: EvaluationSummary, 
                    baseline_summary: EvaluationSummary = None,
                    output_dir: str = "evaluation_results"):
        """
        Save evaluation results to files.
        
        Args:
            rag_summary (EvaluationSummary): RAG evaluation results
            baseline_summary (EvaluationSummary): Baseline evaluation results
            output_dir (str): Output directory for results
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        rag_results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_queries": rag_summary.total_queries,
                "successful_queries": rag_summary.successful_queries,
                "system_type": "RAG"
            },
            "summary": {
                "avg_rouge_1": rag_summary.avg_rouge_1,
                "avg_rouge_2": rag_summary.avg_rouge_2,
                "avg_rouge_l": rag_summary.avg_rouge_l,
                "avg_bleu_score": rag_summary.avg_bleu_score,
                "avg_semantic_similarity": rag_summary.avg_semantic_similarity,
                "avg_response_time_ms": rag_summary.avg_response_time_ms
            },
            "detailed_results": [
                {
                    "query": r.query,
                    "success": r.success,
                    "rouge_scores": r.rouge_scores,
                    "bleu_score": r.bleu_score,
                    "semantic_similarity": r.semantic_similarity,
                    "response_time_ms": r.response_time_ms,
                    "num_retrieved_hadits": r.num_retrieved_hadits,
                    "generated_answer_length": len(r.generated_answer),
                    "reference_answer_length": len(r.reference_answer),
                    "error": r.error
                }
                for r in rag_summary.results
            ]
        }
        
        # Save RAG results
        rag_file = output_path / f"rag_evaluation_{timestamp}.json"
        with open(rag_file, 'w', encoding='utf-8') as f:
            json.dump(rag_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"RAG results saved to: {rag_file}")
        
        # Save baseline results if provided
        if baseline_summary:
            baseline_results = {
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "total_queries": baseline_summary.total_queries,
                    "successful_queries": baseline_summary.successful_queries,
                    "system_type": "Baseline"
                },
                "summary": {
                    "avg_rouge_1": baseline_summary.avg_rouge_1,
                    "avg_rouge_2": baseline_summary.avg_rouge_2,
                    "avg_rouge_l": baseline_summary.avg_rouge_l,
                    "avg_bleu_score": baseline_summary.avg_bleu_score,
                    "avg_semantic_similarity": baseline_summary.avg_semantic_similarity,
                    "avg_response_time_ms": baseline_summary.avg_response_time_ms
                },
                "detailed_results": [
                    {
                        "query": r.query,
                        "success": r.success,
                        "rouge_scores": r.rouge_scores,
                        "bleu_score": r.bleu_score,
                        "semantic_similarity": r.semantic_similarity,
                        "response_time_ms": r.response_time_ms,
                        "num_retrieved_hadits": r.num_retrieved_hadits,
                        "generated_answer_length": len(r.generated_answer),
                        "reference_answer_length": len(r.reference_answer),
                        "error": r.error
                    }
                    for r in baseline_summary.results
                ]
            }
            
            baseline_file = output_path / f"baseline_evaluation_{timestamp}.json"
            with open(baseline_file, 'w', encoding='utf-8') as f:
                json.dump(baseline_results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Baseline results saved to: {baseline_file}")
        
        # Generate comparison report if both results available
        if baseline_summary:
            self._generate_comparison_report(rag_summary, baseline_summary, output_path, timestamp)
    
    def _generate_comparison_report(self, rag_summary: EvaluationSummary, 
                                  baseline_summary: EvaluationSummary,
                                  output_path: Path, timestamp: str):
        """Generate comparison report between RAG and baseline."""
        
        report = f"""# RAG vs Baseline Evaluation Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Summary Comparison

| Metric | RAG System | Baseline | Improvement |
|--------|------------|----------|-------------|
| Success Rate | {rag_summary.successful_queries}/{rag_summary.total_queries} ({rag_summary.successful_queries/rag_summary.total_queries*100:.1f}%) | {baseline_summary.successful_queries}/{baseline_summary.total_queries} ({baseline_summary.successful_queries/baseline_summary.total_queries*100:.1f}%) | {((rag_summary.successful_queries/rag_summary.total_queries) - (baseline_summary.successful_queries/baseline_summary.total_queries))*100:.1f}% |
| ROUGE-1 | {rag_summary.avg_rouge_1:.4f} | {baseline_summary.avg_rouge_1:.4f} | {((rag_summary.avg_rouge_1 - baseline_summary.avg_rouge_1)/baseline_summary.avg_rouge_1*100) if baseline_summary.avg_rouge_1 > 0 else 0:.1f}% |
| ROUGE-2 | {rag_summary.avg_rouge_2:.4f} | {baseline_summary.avg_rouge_2:.4f} | {((rag_summary.avg_rouge_2 - baseline_summary.avg_rouge_2)/baseline_summary.avg_rouge_2*100) if baseline_summary.avg_rouge_2 > 0 else 0:.1f}% |
| ROUGE-L | {rag_summary.avg_rouge_l:.4f} | {baseline_summary.avg_rouge_l:.4f} | {((rag_summary.avg_rouge_l - baseline_summary.avg_rouge_l)/baseline_summary.avg_rouge_l*100) if baseline_summary.avg_rouge_l > 0 else 0:.1f}% |
| BLEU Score | {rag_summary.avg_bleu_score:.4f} | {baseline_summary.avg_bleu_score:.4f} | {((rag_summary.avg_bleu_score - baseline_summary.avg_bleu_score)/baseline_summary.avg_bleu_score*100) if baseline_summary.avg_bleu_score > 0 else 0:.1f}% |
| Semantic Similarity | {rag_summary.avg_semantic_similarity:.4f} | {baseline_summary.avg_semantic_similarity:.4f} | {((rag_summary.avg_semantic_similarity - baseline_summary.avg_semantic_similarity)/baseline_summary.avg_semantic_similarity*100) if baseline_summary.avg_semantic_similarity > 0 else 0:.1f}% |
| Avg Response Time (ms) | {rag_summary.avg_response_time_ms:.1f} | {baseline_summary.avg_response_time_ms:.1f} | {((rag_summary.avg_response_time_ms - baseline_summary.avg_response_time_ms)/baseline_summary.avg_response_time_ms*100) if baseline_summary.avg_response_time_ms > 0 else 0:.1f}% |

## Detailed Analysis

### ROUGE Scores Analysis
- **ROUGE-1**: Measures unigram overlap between generated and reference text
- **ROUGE-2**: Measures bigram overlap for better semantic understanding  
- **ROUGE-L**: Measures longest common subsequence for structural similarity

### BLEU Score Analysis
- Measures n-gram precision with brevity penalty
- Higher scores indicate better alignment with reference text

### Semantic Similarity Analysis
- Uses sentence embeddings to measure semantic relatedness
- Captures meaning beyond lexical similarity

## Query-by-Query Analysis

"""
        
        for i, (rag_result, baseline_result) in enumerate(zip(rag_summary.results, baseline_summary.results)):
            report += f"""
### Query {i+1}: "{rag_result.query}"

| Metric | RAG | Baseline | Difference |
|--------|-----|----------|------------|
| ROUGE-1 | {rag_result.rouge_scores.get('rouge-1', 0):.3f} | {baseline_result.rouge_scores.get('rouge-1', 0):.3f} | {rag_result.rouge_scores.get('rouge-1', 0) - baseline_result.rouge_scores.get('rouge-1', 0):.3f} |
| ROUGE-2 | {rag_result.rouge_scores.get('rouge-2', 0):.3f} | {baseline_result.rouge_scores.get('rouge-2', 0):.3f} | {rag_result.rouge_scores.get('rouge-2', 0) - baseline_result.rouge_scores.get('rouge-2', 0):.3f} |
| ROUGE-L | {rag_result.rouge_scores.get('rouge-l', 0):.3f} | {baseline_result.rouge_scores.get('rouge-l', 0):.3f} | {rag_result.rouge_scores.get('rouge-l', 0) - baseline_result.rouge_scores.get('rouge-l', 0):.3f} |
| BLEU | {rag_result.bleu_score:.3f} | {baseline_result.bleu_score:.3f} | {rag_result.bleu_score - baseline_result.bleu_score:.3f} |
| Semantic | {rag_result.semantic_similarity:.3f} | {baseline_result.semantic_similarity:.3f} | {rag_result.semantic_similarity - baseline_result.semantic_similarity:.3f} |

"""
        
        report_file = output_path / f"comparison_report_{timestamp}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"Comparison report saved to: {report_file}")


async def main():
    """Main evaluation function."""
    print("🧪 RAG Evaluation System - Fixed V1")
    print("=" * 50)
    
    # Check dependencies
    missing_deps = []
    if not ROUGE_AVAILABLE:
        missing_deps.append("rouge")
    if not BLEU_AVAILABLE:
        missing_deps.append("nltk")
    if not SEMANTIC_AVAILABLE:
        missing_deps.append("sentence-transformers scikit-learn")
    
    if missing_deps:
        print("⚠️  Missing dependencies:")
        for dep in missing_deps:
            print(f"   pip install {dep}")
        print("\nSome evaluation metrics will be unavailable.")
        print()
    
    # Initialize evaluator
    evaluator = RAGEvaluator()
    
    try:
        # Initialize services
        await evaluator.initialize_services()
        
        print(f"📊 Ground Truth Dataset: {len(evaluator.ground_truth)} queries")
        print("🚀 Starting RAG evaluation...")
        
        # Evaluate RAG system
        rag_results = await evaluator.evaluate_all(use_rag=True)
        
        print("\n📊 RAG Evaluation Results:")
        print(f"Success Rate: {rag_results.successful_queries}/{rag_results.total_queries}")
        print(f"ROUGE-1: {rag_results.avg_rouge_1:.4f}")
        print(f"ROUGE-2: {rag_results.avg_rouge_2:.4f}")
        print(f"ROUGE-L: {rag_results.avg_rouge_l:.4f}")
        print(f"BLEU Score: {rag_results.avg_bleu_score:.4f}")
        print(f"Semantic Similarity: {rag_results.avg_semantic_similarity:.4f}")
        print(f"Avg Response Time: {rag_results.avg_response_time_ms:.1f}ms")
        
        print("\n🔄 Starting Baseline evaluation...")
        
        # Evaluate baseline system
        baseline_results = await evaluator.evaluate_all(use_rag=False)
        
        print("\n📊 Baseline Evaluation Results:")
        print(f"Success Rate: {baseline_results.successful_queries}/{baseline_results.total_queries}")
        print(f"ROUGE-1: {baseline_results.avg_rouge_1:.4f}")
        print(f"ROUGE-2: {baseline_results.avg_rouge_2:.4f}")
        print(f"ROUGE-L: {baseline_results.avg_rouge_l:.4f}")
        print(f"BLEU Score: {baseline_results.avg_bleu_score:.4f}")
        print(f"Semantic Similarity: {baseline_results.avg_semantic_similarity:.4f}")
        print(f"Avg Response Time: {baseline_results.avg_response_time_ms:.1f}ms")
        
        # Save results
        print("\n💾 Saving evaluation results...")
        evaluator.save_results(rag_results, baseline_results)
        
        print("\n✅ Evaluation completed successfully!")
        print("\nGenerated files:")
        print("- RAG evaluation results (JSON)")
        print("- Baseline evaluation results (JSON)")
        print("- Comparison report (Markdown)")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
