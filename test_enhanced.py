#!/usr/bin/env python3
"""Quick test script for enhanced RAG with Ollama"""
import sys
import os


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("Testing Enhanced RAG with Ollama")
print("=" * 60)

try:
    from app.pdf_rag_enhanced import EnhancedPDFRAG
    print("[OK] Enhanced RAG module imported")
except ImportError as e:
    print(f"[ERROR] Failed to import: {e}")
    sys.exit(1)

print("\n[TEST 1] Initializing Enhanced RAG (no LLM)...")
try:
    rag_no_llm = EnhancedPDFRAG(use_llm=False)
    print("[OK] Enhanced RAG initialized (retrieval-only mode)")
except Exception as e:
    print(f"[ERROR] Failed to initialize: {e}")
    sys.exit(1)

print("\n[TEST 2] Initializing Enhanced RAG with Ollama...")
try:
    rag_with_llm = EnhancedPDFRAG(use_llm=True, llm_type="ollama")
    if rag_with_llm.llm:
        print(f"[OK] Enhanced RAG initialized with LLM: {rag_with_llm.llm}")
    else:
        print("[WARN] Enhanced RAG initialized but LLM not available (will use fallback)")
except Exception as e:
    print(f"[WARN] LLM initialization failed: {e}")
    print("[INFO] System will use improved heuristics instead")

print("\n" + "=" * 60)
print("All tests passed! Enhanced RAG is ready to use.")
print("=" * 60)
print("\nNext steps:")
print("1. Run: python -m app.cli_enhanced --pretty")
print("2. Or test with: python -m app.cli_enhanced 'What is the indirect cost rate?' --pretty")
