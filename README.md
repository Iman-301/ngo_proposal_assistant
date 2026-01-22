# NGO / Project Proposal Assistant (RAG)

Objective: Answer questions about donor and NGO proposal requirements using a lightweight Retrieval-Augmented Generation (RAG) approach over local documents (templates, donor guidelines).

## Features
- TF-IDF + cosine similarity retriever over `.md`/`.txt` in `kb/`.
- Simple, citation-rich answers with top supporting snippets.
- CLI for interactive Q&A and batch questions.
- Basic evaluation script for requirement accuracy.

## Quickstart

1. Create and activate a Python 3.10+ environment, then install deps:
```bash
pip install -r requirements.txt
```

2. Run the CLI in interactive mode (defaults to `kb/`):
```bash
python -m app.cli --pretty
```
Ask something like: "What is the cost share requirement?"

3. Or pass a one-off question:
```bash
python -m app.cli "What reports are required?" --pretty
```

4. Evaluate requirement accuracy on sample questions:
```bash
python tests/evaluate.py
```

5. (Optional) PDF RAG, offline embeddings (no OpenAI key):
- Place PDFs in `kb/`.
- Run a small script that uses `PDFRAG` (see `app/pdf_rag.py`): load PDFs, build vector store, call `create_retriever()`, then `answer(question)`. This path uses local `sentence-transformers` embeddings and no LLM/API calls.

## Add Your Documents
- Put donor RFPs, guidelines, and proposal templates in `kb/` as `.md` or `.txt`.
- For a custom path: `python -m app.cli --kb path/to/your/kb --pretty`.

## How It Works
- Documents are chunked (~800 chars with overlap) to preserve context.
- TF-IDF (1-2 grams, English stop-words) builds a sparse index.
- Queries retrieve top-K chunks via cosine similarity; answer = short summary + citations + snippets.

## Extending (Optional)
- LLM summarization: Wrap retrieved snippets with an LLM (e.g., OpenAI) to draft polished answers.
- File formats: Add PDF/DOCX loaders (e.g., `pypdf`, `python-docx`).
- Better chunking: Heading-aware or semantic splitting.
- Evaluation: Use a larger QA set and exact-match/ROUGE/semantic similarity metrics.

## Notes
- This project intentionally avoids external APIs to run fully offline.
- Replace the sample KB with your actual donor materials for real accuracy.
- The PDF RAG path (`app/pdf_rag.py`) now uses local embeddings (HuggingFace). You can ignore `OPENAI_API_KEY` unless you re-enable OpenAI models.
