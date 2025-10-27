# RAG vs Baseline Evaluation Report
Generated: 2025-08-28 08:15:15

## Summary Comparison

| Metric | RAG System | Baseline | Improvement |
|--------|------------|----------|-------------|
| Success Rate | 10/10 (100.0%) | 10/10 (100.0%) | 0.0% |
| ROUGE-1 | 0.1080 | 0.0874 | 23.6% |
| ROUGE-2 | 0.0058 | 0.0000 | 0.0% |
| ROUGE-L | 0.0994 | 0.0772 | 28.8% |
| BLEU Score | 0.0042 | 0.0017 | 146.1% |
| Semantic Similarity | 0.5182 | 0.4875 | 6.3% |
| Avg Response Time (ms) | 4342.0 | 4450.3 | -2.4% |

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


### Query 1: "cara berwudhu yang benar"

| Metric | RAG | Baseline | Difference |
|--------|-----|----------|------------|
| ROUGE-1 | 0.119 | 0.072 | 0.047 |
| ROUGE-2 | 0.014 | 0.000 | 0.014 |
| ROUGE-L | 0.112 | 0.072 | 0.040 |
| BLEU | 0.006 | 0.001 | 0.005 |
| Semantic | 0.592 | 0.562 | 0.029 |


### Query 2: "hukum zakat fitrah"

| Metric | RAG | Baseline | Difference |
|--------|-----|----------|------------|
| ROUGE-1 | 0.196 | 0.094 | 0.102 |
| ROUGE-2 | 0.025 | 0.000 | 0.025 |
| ROUGE-L | 0.179 | 0.082 | 0.096 |
| BLEU | 0.018 | 0.002 | 0.016 |
| Semantic | 0.451 | 0.371 | 0.080 |


### Query 3: "keutamaan shalat berjamaah"

| Metric | RAG | Baseline | Difference |
|--------|-----|----------|------------|
| ROUGE-1 | 0.138 | 0.130 | 0.008 |
| ROUGE-2 | 0.014 | 0.000 | 0.014 |
| ROUGE-L | 0.119 | 0.091 | 0.028 |
| BLEU | 0.005 | 0.002 | 0.003 |
| Semantic | 0.582 | 0.553 | 0.029 |


### Query 4: "hukum riba"

| Metric | RAG | Baseline | Difference |
|--------|-----|----------|------------|
| ROUGE-1 | 0.096 | 0.109 | -0.013 |
| ROUGE-2 | 0.000 | 0.000 | 0.000 |
| ROUGE-L | 0.096 | 0.109 | -0.013 |
| BLEU | 0.002 | 0.002 | -0.000 |
| Semantic | 0.448 | 0.412 | 0.036 |


### Query 5: "adab makan menurut rasulullah"

| Metric | RAG | Baseline | Difference |
|--------|-----|----------|------------|
| ROUGE-1 | 0.058 | 0.066 | -0.008 |
| ROUGE-2 | 0.000 | 0.000 | 0.000 |
| ROUGE-L | 0.051 | 0.053 | -0.002 |
| BLEU | 0.001 | 0.003 | -0.002 |
| Semantic | 0.562 | 0.592 | -0.031 |


### Query 6: "hukum puasa ramadan"

| Metric | RAG | Baseline | Difference |
|--------|-----|----------|------------|
| ROUGE-1 | 0.084 | 0.075 | 0.008 |
| ROUGE-2 | 0.000 | 0.000 | 0.000 |
| ROUGE-L | 0.076 | 0.075 | 0.001 |
| BLEU | 0.002 | 0.001 | 0.001 |
| Semantic | 0.397 | 0.367 | 0.030 |


### Query 7: "kewajiban berbakti kepada orang tua"

| Metric | RAG | Baseline | Difference |
|--------|-----|----------|------------|
| ROUGE-1 | 0.100 | 0.092 | 0.008 |
| ROUGE-2 | 0.000 | 0.000 | 0.000 |
| ROUGE-L | 0.093 | 0.079 | 0.014 |
| BLEU | 0.001 | 0.001 | 0.000 |
| Semantic | 0.600 | 0.446 | 0.154 |


### Query 8: "hukum sholat jumat"

| Metric | RAG | Baseline | Difference |
|--------|-----|----------|------------|
| ROUGE-1 | 0.092 | 0.061 | 0.030 |
| ROUGE-2 | 0.000 | 0.000 | 0.000 |
| ROUGE-L | 0.092 | 0.061 | 0.030 |
| BLEU | 0.002 | 0.001 | 0.000 |
| Semantic | 0.356 | 0.374 | -0.018 |


### Query 9: "hukum jual beli dalam islam"

| Metric | RAG | Baseline | Difference |
|--------|-----|----------|------------|
| ROUGE-1 | 0.090 | 0.074 | 0.016 |
| ROUGE-2 | 0.000 | 0.000 | 0.000 |
| ROUGE-L | 0.083 | 0.074 | 0.009 |
| BLEU | 0.002 | 0.001 | 0.000 |
| Semantic | 0.508 | 0.543 | -0.035 |


### Query 10: "keutamaan membaca quran"

| Metric | RAG | Baseline | Difference |
|--------|-----|----------|------------|
| ROUGE-1 | 0.107 | 0.100 | 0.007 |
| ROUGE-2 | 0.005 | 0.000 | 0.005 |
| ROUGE-L | 0.094 | 0.075 | 0.019 |
| BLEU | 0.004 | 0.002 | 0.002 |
| Semantic | 0.687 | 0.654 | 0.033 |

