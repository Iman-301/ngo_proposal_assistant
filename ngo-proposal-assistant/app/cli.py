# app/cli.py (updated)
import argparse
import json
import os
import sys
from .pdf_rag import PDFRAG
from .config import Config


def main():
    parser = argparse.ArgumentParser(description="NGO / Project Proposal Assistant (RAG)")
    parser.add_argument("question", type=str, nargs="*", help="Question to ask about donor/NGO requirements.")
    parser.add_argument("--kb", dest="kb", type=str, default=None, help="Path to knowledge base folder with PDFs.")
    parser.add_argument("--k", dest="topk", type=int, default=5, help="Top-K chunks to retrieve.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    args = parser.parse_args()

    kb_path = Config.resolve_kb_path(args.kb)
    
    # Check for PDFs
    pdf_files = [f for f in os.listdir(kb_path) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"No PDF files found in {kb_path}")
        print("Please add your PDF documents (USAID, World Bank, etc.) to this folder.")
        return 1
    
    print(f"Found {len(pdf_files)} PDF documents: {', '.join(pdf_files)}")
    
    # Initialize and load PDF RAG (offline retrieval-only)
    rag = PDFRAG()
    print("Building vector store from PDFs... (no external APIs)")
    rag.load_pdfs_from_folder(kb_path)
    rag.build_vector_store()
    rag.create_retriever(k=args.topk)
    print("RAG system ready. Ask your questions about NGO/donor requirements.\n")
    
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
                print(f"SUMMARY: {result['summary']}\n")
                print("SOURCES:")
                for src in result['sources']:
                    print(f"  • {src['source']} (Page {src['page']}) - {src['donor']} | score={src['score']}")
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