import json

# Check hadits_docs.json format
with open('../data/processed/hadits_docs.json', 'r', encoding='utf-8') as f:
    docs = json.load(f)
    
print(f"Found {len(docs)} documents")
print(f"Sample document keys: {list(docs[0].keys())}")

# Ensure each document has 'id' and 'terjemah' fields
required_fields = ['id', 'terjemah']
valid_docs = [doc for doc in docs if all(field in doc for field in required_fields)]
print(f"Valid documents: {len(valid_docs)}")