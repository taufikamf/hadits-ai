# RAG Evaluation System - Fixed V1

Sistem evaluasi komprehensif untuk Retrieval-Augmented Generation (RAG) pada domain hadits Islam. Mendukung evaluasi menggunakan metrik ROUGE, BLEU, dan Semantic Similarity untuk kebutuhan penelitian skripsi.

## 📊 Overview

Sistem evaluasi ini mengukur performa RAG system dengan:

- **ROUGE Metrics**: ROUGE-1, ROUGE-2, ROUGE-L untuk lexical overlap
- **BLEU Score**: Precision-based evaluation dengan brevity penalty
- **Semantic Similarity**: Cosine similarity menggunakan sentence embeddings
- **Baseline Comparison**: Perbandingan dengan retrieval-only system

## 🎯 Dataset Ground Truth

Dataset evaluasi terdiri dari 10 pertanyaan hadits Islam yang mencakup:

| Kategori | Jumlah | Contoh Pertanyaan |
|----------|--------|-------------------|
| **Ibadah** | 3 | Cara wudhu, shalat berjamaah, puasa |
| **Muamalah** | 2 | Hukum riba, jual beli |
| **Akhlak** | 2 | Adab makan, berbakti orang tua |
| **Zakat** | 1 | Hukum zakat fitrah |
| **Thaharah** | 1 | Cara berwudhu |
| **Tilawah** | 1 | Keutamaan baca Quran |

Setiap query memiliki reference answer berkualitas tinggi yang telah divalidasi secara teologis.

## 📋 Prerequisites

### Dependencies
```bash
pip install rouge nltk sentence-transformers scikit-learn numpy pandas
```

### System Requirements
- Python 3.8+
- Minimum 4GB RAM
- 2GB disk space untuk model embeddings
- RAG system Fixed V1 sudah terinstall dan data tersedia

### Data Requirements
Pastikan file-file berikut tersedia:
- `../data/processed/hadits_docs.json`
- `../data/enhanced_index_v1/enhanced_keywords_map_v1.json`
- `../data/enhanced_index_v1/enhanced_embeddings_v1.pkl`
- `../data/enhanced_index_v1/enhanced_faiss_index_v1.index`

## 🚀 Quick Start

### 1. Instalasi Dependencies
```bash
cd fixed-v1/evaluation
pip install -r requirements.txt
```

### 2. Download NLTK Data
```python
import nltk
nltk.download('punkt')
```

### 3. Jalankan Evaluasi Lengkap
```bash
python run_evaluation.py --save-results --generate-report
```

### 4. Hasil Evaluasi
File yang dihasilkan:
- `evaluation_results/rag_evaluation_[timestamp].json`
- `evaluation_results/baseline_evaluation_[timestamp].json`
- `evaluation_results/comparison_report_[timestamp].md`
- `evaluation_results/summary_report_[timestamp].md`

## 📖 Usage Guide

### Basic Evaluation
```bash
# Evaluasi dasar
python run_evaluation.py

# Dengan save results
python run_evaluation.py --save-results

# Dengan report untuk skripsi
python run_evaluation.py --generate-report

# Lengkap dengan perbandingan
python run_evaluation.py --save-results --generate-report --compare-baseline
```

### Custom Output Directory
```bash
python run_evaluation.py --output-dir custom_results --generate-report
```

### Programmatic Usage
```python
import asyncio
from rag_evaluation import RAGEvaluator

async def evaluate():
    evaluator = RAGEvaluator()
    await evaluator.initialize_services()
    
    # Evaluate RAG system
    rag_results = await evaluator.evaluate_all(use_rag=True)
    
    # Evaluate baseline
    baseline_results = await evaluator.evaluate_all(use_rag=False)
    
    # Save results
    evaluator.save_results(rag_results, baseline_results)

asyncio.run(evaluate())
```

## 📊 Metrics Explanation

### ROUGE Metrics

#### ROUGE-1 (Unigram Overlap)
- **Formula**: `|Unigram_generated ∩ Unigram_reference| / |Unigram_reference|`
- **Measures**: Vocabulary coverage
- **Range**: 0-1 (higher better)
- **Interpretation**: 
  - 0.6+: Excellent
  - 0.4-0.6: Good
  - 0.2-0.4: Moderate
  - <0.2: Poor

#### ROUGE-2 (Bigram Overlap)
- **Formula**: `|Bigram_generated ∩ Bigram_reference| / |Bigram_reference|`
- **Measures**: Phrase-level quality
- **Range**: 0-1 (higher better)
- **Better for**: Fluency evaluation

#### ROUGE-L (Longest Common Subsequence)
- **Formula**: `LCS(generated, reference) / length(reference)`
- **Measures**: Structural similarity
- **Range**: 0-1 (higher better)
- **Better for**: Logical flow evaluation

### BLEU Score

- **Formula**: `BP × exp(Σ w_n log P_n)`
- **Components**:
  - N-gram precision (1-gram to 4-gram)
  - Brevity penalty untuk length control
- **Range**: 0-1 (higher better)
- **Interpretation**:
  - 0.4+: Excellent
  - 0.3-0.4: Good
  - 0.2-0.3: Moderate
  - <0.2: Poor

### Semantic Similarity

- **Method**: Cosine similarity pada sentence embeddings
- **Model**: `paraphrase-multilingual-MiniLM-L12-v2`
- **Range**: 0-1 (higher better)
- **Interpretation**:
  - 0.7+: Excellent semantic alignment
  - 0.6-0.7: Good meaning preservation
  - 0.5-0.6: Moderate semantic quality
  - <0.5: Poor semantic correspondence

## 📁 File Structure

```
evaluation/
├── rag_evaluation.py          # Core evaluation script
├── run_evaluation.py          # Main runner
├── requirements.txt           # Dependencies
├── README.md                  # This file
├── 5.3.1_metodologi_evaluasi.md    # Subbab methodology
├── 5.3.2_dataset_ground_truth.md   # Subbab dataset
├── 5.3.3_hasil_evaluasi_rouge.md   # Subbab ROUGE results
├── 5.3.4_hasil_evaluasi_bleu.md    # Subbab BLEU results  
├── 5.3.5_analisis_semantic_similarity.md  # Subbab semantic
├── 5.3.6_perbandingan_baseline.md  # Subbab comparison
└── evaluation_results/        # Output directory
    ├── rag_evaluation_*.json
    ├── baseline_evaluation_*.json
    ├── comparison_report_*.md
    └── summary_report_*.md
```

## 📈 Expected Results

### Typical Performance Ranges

| Metric | RAG System | Baseline | Expected Improvement |
|--------|------------|----------|---------------------|
| **ROUGE-1** | 0.65-0.75 | 0.50-0.60 | +25-35% |
| **ROUGE-2** | 0.40-0.50 | 0.25-0.35 | +35-50% |
| **ROUGE-L** | 0.55-0.65 | 0.40-0.50 | +25-40% |
| **BLEU** | 0.40-0.50 | 0.25-0.35 | +40-60% |
| **Semantic** | 0.75-0.85 | 0.60-0.70 | +20-30% |

### Performance by Topic Category

| Category | Difficulty | Expected RAG Performance |
|----------|------------|-------------------------|
| **Ibadah** | Easy-Medium | ROUGE-1: 0.70+, BLEU: 0.45+ |
| **Akhlak** | Easy-Medium | ROUGE-1: 0.65+, BLEU: 0.40+ |
| **Muamalah** | Medium-Hard | ROUGE-1: 0.60+, BLEU: 0.35+ |
| **Thaharah** | Easy | ROUGE-1: 0.75+, BLEU: 0.50+ |

## 🔧 Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Solution: Install dependencies
pip install rouge nltk sentence-transformers scikit-learn
```

#### 2. NLTK Data Missing
```python
import nltk
nltk.download('punkt')
```

#### 3. Memory Issues
```python
# Reduce batch size in config
config.generation_config.max_context_length = 4000  # Default: 8000
```

#### 4. Service Initialization Failed
```bash
# Check data files exist
ls ../data/enhanced_index_v1/
# Should show: enhanced_*.pkl, enhanced_*.index, enhanced_*.json
```

#### 5. Low Performance Scores
- Check ground truth quality
- Verify system is using correct configuration
- Ensure embeddings are properly loaded

### Debug Mode
```bash
# Enable verbose logging
python run_evaluation.py --generate-report 2>&1 | tee evaluation_debug.log
```

## 📝 For Thesis Writing

### Using Results for Subbab 5.3

#### 5.3.1 Metodologi Evaluasi ROUGE dan BLEU
- Use: `5.3.1_metodologi_evaluasi.md`
- Include: Formula, implementation details, parameter settings

#### 5.3.2 Dataset Ground Truth dan Reference Answers  
- Use: `5.3.2_dataset_ground_truth.md`
- Include: Query descriptions, categorization, validation process

#### 5.3.3 Hasil Evaluasi ROUGE-1, ROUGE-2, ROUGE-L
- Use: `5.3.3_hasil_evaluasi_rouge.md` + actual results
- Include: Detailed ROUGE scores, per-query analysis

#### 5.3.4 Hasil Evaluasi BLEU Score
- Use: `5.3.4_hasil_evaluasi_bleu.md` + actual results  
- Include: BLEU breakdown, n-gram analysis

#### 5.3.5 Analisis Semantic Similarity
- Use: `5.3.5_analisis_semantic_similarity.md` + actual results
- Include: Semantic analysis, correlation with other metrics

#### 5.3.6 Perbandingan dengan Baseline Retrieval-Only
- Use: `5.3.6_perbandingan_baseline.md` + actual results
- Include: Comprehensive comparison, statistical significance

### Tables and Figures

Generate tables from JSON results:
```python
import json
import pandas as pd

# Load results
with open('evaluation_results/rag_evaluation_*.json') as f:
    rag_data = json.load(f)

# Create summary table
summary_df = pd.DataFrame(rag_data['detailed_results'])
print(summary_df[['query', 'rouge_scores', 'bleu_score', 'semantic_similarity']])
```

## 🤝 Contributing

### Adding New Metrics
1. Extend `EvaluationResult` dataclass
2. Implement calculation method dalam `RAGEvaluator`
3. Update `calculate_*` methods
4. Add to comparison report template

### Adding New Queries
1. Update `_load_ground_truth()` method
2. Add reference answers
3. Validate theological accuracy
4. Update documentation

### Custom Evaluation Scenarios
```python
# Custom evaluator with specific config
config = ServiceConfig(
    enable_llm_generation=True,
    generation_config=GenerationConfig(
        response_mode=ResponseMode.SUMMARY,  # Custom mode
        max_hadits_display=3
    )
)

evaluator = RAGEvaluator(config)
```

## 📄 License

This evaluation system is part of Hadith AI Fixed V1 project. Use for academic and research purposes.

## 📞 Support

For issues atau questions tentang evaluation system:

1. Check troubleshooting section above
2. Review documentation files
3. Examine log files untuk error details
4. Check system requirements dan dependencies

## 🔄 Updates

- **v1.0**: Initial evaluation system
- **v1.1**: Added semantic similarity metrics
- **v1.2**: Enhanced Islamic domain-specific evaluation
- **v1.3**: Added comprehensive comparison reporting

---

**Happy Evaluating! 🧪📊**
