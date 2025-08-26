# ChromaDB to FAISS Migration Guide

This document describes the migration from ChromaDB to FAISS for the Hadits AI vector database.

## Changes Made

### 1. Dependencies
- **Removed**: `chromadb==0.4.24`
- **Added**: `faiss-cpu==1.7.4`

### 2. Configuration Variables
- **Removed**: `CHROMA_DB_PATH`
- **Added**: 
  - `FAISS_INDEX_PATH` - Path to FAISS index file
  - `METADATA_PATH` - Path to metadata JSON file

### 3. Code Changes

#### `indexing/build_index.py`
- Replaced ChromaDB client with FAISS IndexFlatIP
- Added separate metadata storage as JSON file
- Maintained the same batch processing functionality
- Uses cosine similarity (via normalized vectors and inner product)

#### `retriever/query_runner.py`
- Replaced ChromaDB collection loading with FAISS index + metadata loading
- Updated embedding normalization for FAISS compatibility
- Maintained the same query interface and filtering functionality
- Added caching for index and metadata to improve performance

## Key Differences

### Storage Architecture
- **ChromaDB**: Single database with embedded metadata
- **FAISS**: Separate index file + JSON metadata file

### Similarity Search
- **ChromaDB**: Built-in cosine similarity with HNSW
- **FAISS**: IndexFlatIP (exact inner product) with L2-normalized vectors

### Metadata Handling
- **ChromaDB**: Metadata stored with vectors
- **FAISS**: Metadata stored separately and accessed by index position

## Performance Benefits

1. **Faster Search**: FAISS is optimized for vector similarity search
2. **Lower Memory Usage**: FAISS has better memory efficiency
3. **Better Scalability**: FAISS handles large datasets more efficiently
4. **Exact Results**: IndexFlatIP provides exact similarity scores

## Migration Steps

If you have existing ChromaDB data:

1. Export embeddings and documents from ChromaDB
2. Run the new FAISS indexing process:
   ```bash
   python indexing/build_index.py
   ```
3. Update environment variables in `.env` file
4. Test the new system with existing queries

## Backward Compatibility

The migration maintains full API compatibility:
- Same function signatures in `query_hadits_return()`
- Same response format with rankings and scores
- Same keyword filtering functionality
- Same batch processing capabilities

## File Structure

```
db/
├── hadits_index          # FAISS index file
└── hadits_metadata.json  # Document metadata
```

## Testing

Run the validation script to ensure the migration is working:
```bash
python validate_migration.py
```