## NGO Proposal Assistant (Offline PDF RAG + Optional LLM)

An offline RAG system for NGO proposal guidance. It ingests PDF guidelines, builds a local vector index, retrieves relevant passages, and generates answers. It works fully offline with local embeddings. Optional LLM support is available via Ollama for more fluent summaries.

### Features
- PDF ingestion and chunking
- Local embeddings (no OpenAI key required)
- Vector search with citations (Chroma)
- Optional local LLM via Ollama
- CLI and web UI
- Evaluation scripts and reports

### Requirements
- Python 3.10+ (recommended: 3.11)
- Windows/macOS/Linux
- Optional: Ollama for LLM answers

### Setup
1. Create a virtual environment and install dependencies:
   - Windows PowerShell
	- `python -m venv .venv`
	- `.
venv\Scripts\Activate.ps1`
	- `python -m pip install -r requirements.txt`


### Run the Web UI
Start the web app:
- `python -m app.web`
Then open: `http://localhost:5001`

Upload a PDF and ask a question. The answer includes sources and page references.

### Enable LLM (Ollama)
The project works without an LLM. For better summaries, use Ollama:
- `ollama serve`
- `ollama pull llama3.2`

The system will auto-detect Ollama and use it when available.

### Evaluation
Run evaluation:
- `python tests/evaluate.py`

Outputs:
- `evaluation_report.json`
- `evaluation_report_detailed.json`

### Project Structure
- `app/pdf_rag.py`: PDF RAG pipeline (chunking, embeddings, retrieval, citations)
- `app/cli.py`: CLI entrypoint
- `app/web.py`: Flask web UI
- `app/config.py`: configuration defaults and env overrides
- `tests/evaluate.py`: evaluation runner

### Troubleshooting
- Hugging Face download timeout: re-run once; the model caches locally after first download.
- Windows symlink warning: enable Developer Mode or set `HF_HUB_DISABLE_SYMLINKS_WARNING=1`.
- Ollama model not found: run `ollama pull llama3.2`.

### Notes
- The vector store is saved in `chroma_db/` and should not be committed.
- `.env` is ignored by git; keep secrets there if needed.
