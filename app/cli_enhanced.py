# app/cli_enhanced.py - Enhanced CLI with optional free LLM support
import argparse
import json
import os
import sys
from .config import Config

# Try to use enhanced version, fallback to original
try:
    from .pdf_rag_enhanced import EnhancedPDFRAG
    ENHANCED_AVAILABLE = True
except ImportError:
    from .pdf_rag import PDFRAG as EnhancedPDFRAG
    ENHANCED_AVAILABLE = False
    print("[INFO] Using standard RAG (enhanced version not available)")


def main():
    parser = argparse.ArgumentParser(description="NGO / Project Proposal Assistant (RAG) - Enhanced with Free LLM")
    parser.add_argument("question", type=str, nargs="*", help="Question to ask about donor/NGO requirements.")
    parser.add_argument("--kb", dest="kb", type=str, default=None, help="Path to knowledge base folder with PDFs.")
    parser.add_argument("--k", dest="topk", type=int, default=5, help="Top-K chunks to retrieve.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM (use retrieval-only mode).")
    parser.add_argument("--llm-type", dest="llm_type", type=str, default="ollama", 
                       choices=["ollama", "huggingface", "none"],
                       help="LLM type to use (default: ollama)")
    args = parser.parse_args()

    kb_path = Config.resolve_kb_path(args.kb)
    
    # Check for PDFs
    pdf_files = [f for f in os.listdir(kb_path) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"No PDF files found in {kb_path}")
        print("Please add your PDF documents (USAID, World Bank, etc.) to this folder.")
        return 1
    
    print(f"Found {len(pdf_files)} PDF documents: {', '.join(pdf_files)}")
    
    # Initialize RAG (with or without LLM)
    use_llm = not args.no_llm and ENHANCED_AVAILABLE
    if use_llm:
        print("[INIT] Initializing Enhanced RAG with optional free LLM support...")
        rag = EnhancedPDFRAG(use_llm=use_llm, llm_type=args.llm_type)
    else:
        print("[INIT] Initializing RAG (retrieval-only mode)...")
        if ENHANCED_AVAILABLE:
            rag = EnhancedPDFRAG(use_llm=False)
        else:
            from .pdf_rag import PDFRAG
            rag = PDFRAG()
    
    print("Building vector store from PDFs...")
    rag.load_pdfs_from_folder(kb_path)
    rag.build_vector_store()
    rag.create_retriever(k=args.topk)
    
    if use_llm and hasattr(rag, 'llm') and rag.llm:
        print("[OK] RAG system ready with LLM enhancement!")
    else:
        print("[OK] RAG system ready (retrieval-only mode).")
    print("Ask your questions about NGO/donor requirements.\n")
    
    if not args.question:
        # Interactive mode
        try:
            while True:
                q = input("\nQ> ").strip()
                if not q:
                    continue
                if q.lower() in ['exit', 'quit', 'bye']:
                    break
                    
                print("\n" + "="*60)
                result = rag.answer(q)
                
                # Show if LLM was used
                if result.get('llm_used'):
                    print("[LLM-Enhanced Answer]")
                else:
                    print("[Retrieval-Only Answer]")
                
                print(f"\nSUMMARY: {result['summary']}\n")
                print("SOURCES:")
                for src in result['sources']:
                    print(f"  • {src['source']} (Page {src['page']}) - {src['donor']}")
                    if src.get('snippet'):
                        print(f"    Excerpt: {src['snippet']}")
                print("="*60)
                
        except KeyboardInterrupt:
            print("\n\nExiting. Goodbye!")
            return 0
    else:
        # Single question mode
        q = " ".join(args.question)
        result = rag.answer(q)
        
        if args.pretty:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(json.dumps(result, ensure_ascii=False))
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
