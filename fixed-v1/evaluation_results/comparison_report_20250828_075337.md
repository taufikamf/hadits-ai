# RAG vs Baseline Evaluation Report
Generated: 2025-08-28 07:53:37

## Summary Comparison

| Metric | RAG System | Baseline | Improvement |
|--------|------------|----------|-------------|
| Success Rate | 10/10 (100.0%) | 10/10 (100.0%) | 0.0% |
| ROUGE-1 | 0.1000 | 0.0826 | 21.1% |
| ROUGE-2 | 0.0079 | 0.0000 | 0.0% |
| ROUGE-L | 0.0944 | 0.0723 | 30.7% |
| BLEU Score | 0.0000 | 0.0000 | 0.0% |
| Semantic Similarity | 0.5182 | 0.4875 | 6.3% |
| Avg Response Time (ms) | 4346.3 | 4362.1 | -0.4% |

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
| ROUGE-1 | 0.104 | 0.067 | 0.038 |
| ROUGE-2 | 0.010 | 0.000 | 0.010 |
| ROUGE-L | 0.099 | 0.067 | 0.032 |
| BLEU | 0.000 | 0.000 | 0.000 |
| Semantic | 0.592 | 0.562 | 0.029 |


### Query 2: "hukum zakat fitrah"

| Metric | RAG | Baseline | Difference |
|--------|-----|----------|------------|
| ROUGE-1 | 0.182 | 0.088 | 0.094 |
| ROUGE-2 | 0.026 | 0.000 | 0.026 |
| ROUGE-L | 0.166 | 0.077 | 0.089 |
| BLEU | 0.000 | 0.000 | 0.000 |
| Semantic | 0.451 | 0.371 | 0.080 |


### Query 3: "keutamaan shalat berjamaah"

| Metric | RAG | Baseline | Difference |
|--------|-----|----------|------------|
| ROUGE-1 | 0.128 | 0.129 | -0.002 |
| ROUGE-2 | 0.020 | 0.000 | 0.020 |
| ROUGE-L | 0.120 | 0.094 | 0.026 |
| BLEU | 0.000 | 0.000 | 0.000 |
| Semantic | 0.582 | 0.553 | 0.029 |


### Query 4: "hukum riba"

| Metric | RAG | Baseline | Difference |
|--------|-----|----------|------------|
| ROUGE-1 | 0.099 | 0.114 | -0.014 |
| ROUGE-2 | 0.000 | 0.000 | 0.000 |
| ROUGE-L | 0.099 | 0.114 | -0.014 |
| BLEU | 0.000 | 0.000 | 0.000 |
| Semantic | 0.448 | 0.412 | 0.036 |


### Query 5: "adab makan menurut rasulullah"

| Metric | RAG | Baseline | Difference |
|--------|-----|----------|------------|
| ROUGE-1 | 0.050 | 0.058 | -0.008 |
| ROUGE-2 | 0.000 | 0.000 | 0.000 |
| ROUGE-L | 0.044 | 0.046 | -0.002 |
| BLEU | 0.000 | 0.000 | 0.000 |
| Semantic | 0.562 | 0.592 | -0.031 |


### Query 6: "hukum puasa ramadan"

| Metric | RAG | Baseline | Difference |
|--------|-----|----------|------------|
| ROUGE-1 | 0.075 | 0.068 | 0.007 |
| ROUGE-2 | 0.000 | 0.000 | 0.000 |
| ROUGE-L | 0.070 | 0.068 | 0.002 |
| BLEU | 0.000 | 0.000 | 0.000 |
| Semantic | 0.397 | 0.367 | 0.030 |


### Query 7: "kewajiban berbakti kepada orang tua"

| Metric | RAG | Baseline | Difference |
|--------|-----|----------|------------|
| ROUGE-1 | 0.091 | 0.081 | 0.010 |
| ROUGE-2 | 0.015 | 0.000 | 0.015 |
| ROUGE-L | 0.091 | 0.069 | 0.022 |
| BLEU | 0.000 | 0.000 | 0.000 |
| Semantic | 0.600 | 0.446 | 0.154 |


### Query 8: "hukum sholat jumat"

| Metric | RAG | Baseline | Difference |
|--------|-----|----------|------------|
| ROUGE-1 | 0.098 | 0.068 | 0.030 |
| ROUGE-2 | 0.004 | 0.000 | 0.004 |
| ROUGE-L | 0.092 | 0.057 | 0.035 |
| BLEU | 0.000 | 0.000 | 0.000 |
| Semantic | 0.356 | 0.374 | -0.018 |


### Query 9: "hukum jual beli dalam islam"

| Metric | RAG | Baseline | Difference |
|--------|-----|----------|------------|
| ROUGE-1 | 0.086 | 0.063 | 0.023 |
| ROUGE-2 | 0.000 | 0.000 | 0.000 |
| ROUGE-L | 0.076 | 0.063 | 0.013 |
| BLEU | 0.000 | 0.000 | 0.000 |
| Semantic | 0.508 | 0.543 | -0.035 |


### Query 10: "keutamaan membaca quran"

| Metric | RAG | Baseline | Difference |
|--------|-----|----------|------------|
| ROUGE-1 | 0.087 | 0.090 | -0.003 |
| ROUGE-2 | 0.003 | 0.000 | 0.003 |
| ROUGE-L | 0.087 | 0.068 | 0.019 |
| BLEU | 0.000 | 0.000 | 0.000 |
| Semantic | 0.687 | 0.654 | 0.033 |

