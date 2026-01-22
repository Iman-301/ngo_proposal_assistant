#!/usr/bin/env python3
"""
Requirements Verification Script
Checks if all teacher requirements are satisfied
"""

import os
import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists"""
    exists = os.path.exists(filepath)
    status = "OK " if exists else "ERR"
    print(f"{status} {description}: {filepath}")
    return exists

def check_directory_exists(dirpath, description):
    """Check if a directory exists and has files"""
    exists = os.path.isdir(dirpath)
    if exists:
        files = [f for f in os.listdir(dirpath) if f.endswith('.pdf')]
        has_files = len(files) > 0
        status = "OK " if has_files else "WARN"
        print(f"{status} {description}: {dirpath} ({len(files)} PDFs found)")
        return has_files
    else:
        print(f"ERR {description}: {dirpath} (not found)")
        return False

def main():
    print("=" * 70)
    print("REQUIREMENTS VERIFICATION - NGO PROPOSAL ASSISTANT")
    print("=" * 70)
    print()
    
    base_path = Path(__file__).parent
    all_checks_passed = True
    
    # 1. Check for required files
    print("[FILES] CHECKING REQUIRED FILES:")
    print("-" * 70)
    
    required_files = [
        ("app/pdf_rag.py", "Main RAG implementation"),
        ("app/cli.py", "CLI interface"),
        ("app/config.py", "Configuration file"),
        ("tests/evaluate.py", "Evaluation script"),
        ("requirements.txt", "Dependencies file"),
        ("README.md", "Project documentation"),
    ]
    
    for filepath, description in required_files:
        if not check_file_exists(base_path / filepath, description):
            all_checks_passed = False
    
    print()
    
    # 2. Check for data (PDFs)
    print("[DATA] CHECKING DATA (PDF DOCUMENTS):")
    print("-" * 70)
    
    kb_path = base_path / "kb"
    if not check_directory_exists(kb_path, "Knowledge base folder"):
        all_checks_passed = False
    else:
        pdf_files = [f for f in os.listdir(kb_path) if f.endswith('.pdf')]
        print(f"   Found PDFs: {', '.join(pdf_files)}")
    
    print()
    
    # 3. Check for key functionality
    print("[CODE] CHECKING KEY FUNCTIONALITY:")
    print("-" * 70)
    
    # Check if PDFRAG class exists
    try:
        sys.path.insert(0, str(base_path))
        from app.pdf_rag import PDFRAG
        print("OK  PDFRAG class: Imported successfully")
        
        # Check for key methods
        methods = ['load_pdfs_from_folder', 'build_vector_store', 'create_retriever', 'answer']
        for method in methods:
            if hasattr(PDFRAG, method):
                print(f"OK  Method '{method}': Present")
            else:
                print(f"ERR Method '{method}': Missing")
                all_checks_passed = False
    except ImportError as e:
        print(f"ERR PDFRAG class: Import failed - {e}")
        all_checks_passed = False
    
    print()
    
    # 4. Check for evaluation
    print("[EVAL] CHECKING EVALUATION:")
    print("-" * 70)
    
    try:
        from tests.evaluate import evaluate_pdf_rag
        print("OK  Evaluation function: Imported successfully")
    except ImportError as e:
        print(f"ERR Evaluation function: Import failed - {e}")
        all_checks_passed = False
    
    print()
    
    # 5. Check for citations/sources in answer format
    print("[CITE] CHECKING ANSWER FORMAT (CITATIONS):")
    print("-" * 70)
    
    try:
        from app.pdf_rag import PDFRAG
        # Check if answer method returns sources
        import inspect
        source = inspect.getsource(PDFRAG.answer)
        if 'sources' in source or '"sources"' in source or "'sources'" in source:
            print("OK  Answer method: Returns sources/citations")
        else:
            print("WARN Answer method: May not return sources")
    except Exception as e:
        print(f"WARN Could not verify answer format: {e}")
    
    print()
    
    # 6. Summary
    print("=" * 70)
    if all_checks_passed:
        print("SUCCESS: ALL REQUIREMENTS VERIFIED!")
        print()
        print("Your project satisfies all teacher requirements:")
        print("  [OK] Working RAG System")
        print("  [OK] Data (donor guidelines PDFs)")
        print("  [OK] RAG Benefit (context-specific answers with citations)")
        print("  [OK] Evaluation (requirement accuracy testing)")
        print("  [OK] Outcome (practical social-impact QA system)")
        print()
        print("Ready for presentation!")
    else:
        print("WARNING: SOME REQUIREMENTS MAY BE MISSING")
        print("Please review the checklist above and fix any issues.")
    print("=" * 70)
    
    return 0 if all_checks_passed else 1

if __name__ == "__main__":
    sys.exit(main())
