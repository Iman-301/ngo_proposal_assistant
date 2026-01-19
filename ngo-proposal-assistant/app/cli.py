import argparse
import json
from .rag import SimpleRAG
from .config import Config


def main():
    parser = argparse.ArgumentParser(description="NGO / Project Proposal Assistant (RAG)")
    parser.add_argument("question", type=str, nargs="*", help="Question to ask about donor/NGO requirements.")
    parser.add_argument("--kb", dest="kb", type=str, default=None, help="Path to knowledge base folder (md/txt).")
    parser.add_argument("--topk", dest="topk", type=int, default=Config.top_k, help="Top-K chunks to retrieve.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    args = parser.parse_args()

    kb_path = Config.resolve_kb_path(args.kb)

    rag = SimpleRAG()
    n = rag.load_kb(kb_path)
    if n == 0:
        print(f"No documents found in KB: {kb_path}")
        return 1
    rag.build_index()

    if not args.question:
        # Interactive mode
        print("RAG ready. Type your questions (Ctrl+C to exit).\n")
        try:
            while True:
                q = input("Q> ").strip()
                if not q:
                    continue
                ans = rag.answer(q, top_k=args.topk)
                if args.pretty:
                    print(json.dumps(ans, indent=2, ensure_ascii=False))
                else:
                    print(json.dumps(ans, ensure_ascii=False))
        except KeyboardInterrupt:
            print("\nBye.")
            return 0
    else:
        q = " ".join(args.question)
        ans = rag.answer(q, top_k=args.topk)
        if args.pretty:
            print(json.dumps(ans, indent=2, ensure_ascii=False))
        else:
            print(json.dumps(ans, ensure_ascii=False))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
