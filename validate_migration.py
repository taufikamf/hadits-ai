#!/usr/bin/env python3
"""
Syntax and structure validation for FAISS migration.
This script checks that the code is syntactically correct and has the right structure.
"""

import os
import sys
import ast
import importlib.util

def check_syntax(file_path):
    """Check if a Python file has valid syntax"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        # Parse the AST to check syntax
        ast.parse(source, filename=file_path)
        return True, None
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error reading file: {e}"

def check_imports(file_path):
    """Check if imports in a file are structured correctly"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        tree = ast.parse(source, filename=file_path)
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports.append(name.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for name in node.names:
                    imports.append(f"{module}.{name.name}")
        
        return True, imports
    except Exception as e:
        return False, f"Error checking imports: {e}"

def check_function_definitions(file_path, expected_functions):
    """Check if expected functions are defined in a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        tree = ast.parse(source, filename=file_path)
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                functions.append(node.name)
        
        missing_functions = [f for f in expected_functions if f not in functions]
        return len(missing_functions) == 0, functions, missing_functions
    except Exception as e:
        return False, [], f"Error checking functions: {e}"

def validate_build_index():
    """Validate the build_index.py file"""
    file_path = "indexing/build_index.py"
    print(f"🔍 Validating {file_path}...")
    
    # Check syntax
    is_valid, error = check_syntax(file_path)
    if not is_valid:
        print(f"❌ Syntax error in {file_path}: {error}")
        return False
    
    # Check imports
    is_valid, imports = check_imports(file_path)
    if not is_valid:
        print(f"❌ Import error in {file_path}: {imports}")
        return False
    
    # Check that FAISS imports are present and ChromaDB imports are removed
    faiss_imported = any("faiss" in imp for imp in imports)
    chromadb_imported = any("chromadb" in imp for imp in imports)
    
    if not faiss_imported:
        print(f"❌ FAISS not imported in {file_path}")
        return False
    
    if chromadb_imported:
        print(f"❌ ChromaDB still imported in {file_path}")
        return False
    
    # Check expected functions
    expected_functions = ["load_embeddings", "index_to_faiss"]
    is_valid, functions, missing = check_function_definitions(file_path, expected_functions)
    if not is_valid:
        print(f"❌ Function check error in {file_path}: {missing}")
        return False
    
    if missing:
        print(f"❌ Missing functions in {file_path}: {missing}")
        return False
    
    print(f"✅ {file_path} validation passed")
    return True

def validate_query_runner():
    """Validate the query_runner.py file"""
    file_path = "retriever/query_runner.py"
    print(f"🔍 Validating {file_path}...")
    
    # Check syntax
    is_valid, error = check_syntax(file_path)
    if not is_valid:
        print(f"❌ Syntax error in {file_path}: {error}")
        return False
    
    # Check imports
    is_valid, imports = check_imports(file_path)
    if not is_valid:
        print(f"❌ Import error in {file_path}: {imports}")
        return False
    
    # Check that FAISS imports are present and ChromaDB imports are removed
    faiss_imported = any("faiss" in imp for imp in imports)
    chromadb_imported = any("chromadb" in imp for imp in imports)
    
    if not faiss_imported:
        print(f"❌ FAISS not imported in {file_path}")
        return False
    
    if chromadb_imported:
        print(f"❌ ChromaDB still imported in {file_path}")
        return False
    
    # Check expected functions
    expected_functions = ["load_faiss_index_and_metadata", "get_query_embedding", "query_hadits_return", "keyword_match"]
    is_valid, functions, missing = check_function_definitions(file_path, expected_functions)
    if not is_valid:
        print(f"❌ Function check error in {file_path}: {missing}")
        return False
    
    if missing:
        print(f"❌ Missing functions in {file_path}: {missing}")
        return False
    
    print(f"✅ {file_path} validation passed")
    return True

def validate_requirements():
    """Validate the requirements.txt file"""
    file_path = "requirements.txt"
    print(f"🔍 Validating {file_path}...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        has_faiss = "faiss" in content.lower()
        has_chromadb = "chromadb" in content.lower()
        
        if not has_faiss:
            print(f"❌ FAISS not found in {file_path}")
            return False
        
        if has_chromadb:
            print(f"❌ ChromaDB still present in {file_path}")
            return False
        
        print(f"✅ {file_path} validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Error validating {file_path}: {e}")
        return False

def validate_env_example():
    """Validate the .env.example file"""
    file_path = ".env.example"
    print(f"🔍 Validating {file_path}...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        has_faiss_config = "FAISS" in content
        has_metadata_config = "METADATA_PATH" in content
        has_chroma_config = "CHROMA" in content
        
        if not has_faiss_config:
            print(f"❌ FAISS configuration not found in {file_path}")
            return False
        
        if not has_metadata_config:
            print(f"❌ METADATA_PATH configuration not found in {file_path}")
            return False
        
        if has_chroma_config:
            print(f"❌ ChromaDB configuration still present in {file_path}")
            return False
        
        print(f"✅ {file_path} validation passed")
        return True
        
    except Exception as e:
        print(f"❌ Error validating {file_path}: {e}")
        return False

def main():
    """Run all validations"""
    print("🚀 Starting FAISS migration validation...\n")
    
    validations = [
        validate_requirements,
        validate_env_example,
        validate_build_index,
        validate_query_runner,
    ]
    
    passed = 0
    total = len(validations)
    
    for validation in validations:
        try:
            if validation():
                passed += 1
        except Exception as e:
            print(f"❌ Validation {validation.__name__} failed with exception: {e}")
        print()  # Add spacing between validations
    
    print(f"📊 Validation Results: {passed}/{total} validations passed")
    
    if passed == total:
        print("🎉 All validations passed! FAISS migration code is structurally correct.")
    else:
        print("❌ Some validations failed. Please check the implementation.")
    
    return passed == total

if __name__ == "__main__":
    main()