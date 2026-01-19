from __future__ import annotations
import os
import json
import sys

# Ensure project root is on sys.path when running this file directly
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.rag import SimpleRAG
from app.config import Config


def evaluate(kb_path: str) -> dict:
    rag = SimpleRAG()
    n = rag.load_kb(kb_path)
    assert n > 0, "Knowledge base is empty."
    rag.build_index()

    cases = [
        {
            "q": "What is the maximum total budget allowed?",
            "expect_keywords": ["$250,000", "budget"],
        },
        {
            "q": "What are the indirect cost rate limits?",
            "expect_keywords": ["Indirect", "10%"],
        },
        {
            "q": "What reports and frequency are required?",
            "expect_keywords": ["Quarterly", "narrative", "financial"],
        },
        {
            "q": "Is a LogFrame required?",
            "expect_keywords": ["LogFrame", "indicators"],
        },
        {
            "q": "What are procurement rules above $5,000?",
            "expect_keywords": ["3 quotes", "sole-source", "procurement"],
        },
    ]

    results = []
    correct = 0
    for c in cases:
        ans = rag.answer(c["q"], top_k=5)
        blob = (ans.get("summary", "") + "\n" + "\n".join(ans.get("snippets", []))).lower()
        ok = all(k.lower() in blob for k in c["expect_keywords"])
        correct += 1 if ok else 0
        results.append({"question": c["q"], "ok": ok, "citations": ans.get("citations", [])})

    return {"total": len(cases), "correct": correct, "accuracy": correct / len(cases), "details": results}


if __name__ == "__main__":
    kb = Config.resolve_kb_path(None)
    report = evaluate(kb)
    print(json.dumps(report, indent=2))
