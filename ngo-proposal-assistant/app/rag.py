from __future__ import annotations
import os
import glob
from dataclasses import dataclass
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class DocumentChunk:
    text: str
    source: str
    chunk_id: int


class SimpleRAG:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 120):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None
        self.chunks: List[DocumentChunk] = []

    def _read_text_file(self, path: str) -> str:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    def _split_into_chunks(self, text: str, source: str) -> List[DocumentChunk]:
        # Prefer paragraph-aware splitting, then window with overlap
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        merged = []
        for p in paragraphs:
            if not merged:
                merged.append(p)
                continue
            # Merge small paragraphs to reach minimum useful length
            if len(merged[-1]) < self.chunk_size * 0.6:
                merged[-1] = merged[-1] + "\n\n" + p
            else:
                merged.append(p)
        text_for_window = "\n\n".join(merged) if merged else text
        chunks: List[DocumentChunk] = []
        i = 0
        chunk_id = 0
        while i < len(text_for_window):
            window = text_for_window[i : i + self.chunk_size]
            if not window.strip():
                break
            chunks.append(DocumentChunk(text=window, source=source, chunk_id=chunk_id))
            chunk_id += 1
            i += max(1, self.chunk_size - self.chunk_overlap)
        return chunks

    def load_kb(self, kb_path: str) -> int:
        patterns = [os.path.join(kb_path, "**", "*.md"), os.path.join(kb_path, "**", "*.txt")]
        files: List[str] = []
        for pat in patterns:
            files.extend(glob.glob(pat, recursive=True))
        files = sorted(set(files))
        self.chunks.clear()
        for fp in files:
            try:
                text = self._read_text_file(fp)
            except Exception:
                continue
            self.chunks.extend(self._split_into_chunks(text, source=os.path.relpath(fp, kb_path)))
        return len(self.chunks)

    def build_index(self) -> None:
        if not self.chunks:
            raise ValueError("No chunks to index. Load KB first.")
        docs = [c.text for c in self.chunks]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        self.matrix = self.vectorizer.fit_transform(docs)

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        if not self.vectorizer or self.matrix is None:
            raise ValueError("Index not built. Call build_index() after load_kb().")
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix)[0]
        idxs = sims.argsort()[::-1][:top_k]
        return [(self.chunks[i], float(sims[i])) for i in idxs]

    def answer(self, question: str, top_k: int = 5) -> dict:
        hits = self.retrieve(question, top_k=top_k)
        # Assemble a lightweight, citation-rich response
        snippets = []
        sources = []
        for chunk, score in hits:
            snippets.append(chunk.text.strip())
            sources.append({
                "source": chunk.source,
                "chunk_id": chunk.chunk_id,
                "score": round(score, 4),
            })
        # Heuristic summary: first 2 sentences from best chunk
        summary = ""
        if hits:
            best_text = hits[0][0].text.strip().replace("\n", " ")
            sentences = [s.strip() for s in best_text.split('.') if s.strip()]
            summary = '. '.join(sentences[:2])
            if summary and not summary.endswith('.'): summary += '.'
        return {
            "question": question,
            "summary": summary,
            "citations": sources,
            "snippets": snippets,
        }
