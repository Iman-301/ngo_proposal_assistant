# app/pdf_rag.py
from __future__ import annotations
import os
import glob
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

# PDF processing
from pypdf import PdfReader

# LangChain (offline) components
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

@dataclass
class PDFChunk:
    text: str
    source: str
    page: int
    chunk_id: str
    metadata: Dict[str, Any]

class PDFRAG:
    def __init__(self, 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Local embeddings (no API key required)
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vectorstore = None
        self.documents: List[PDFChunk] = []
        self.retriever = None
        self.k = 5
        self.fetch_k = 15
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def load_pdfs_from_folder(self, folder_path: str) -> int:
        """Load all PDFs from a folder"""
        pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
        
        all_docs = []
        for pdf_file in pdf_files:
            docs = self._load_single_pdf(pdf_file)
            all_docs.extend(docs)
        
        self.documents = all_docs
        return len(all_docs)
    
    def _load_single_pdf(self, pdf_path: str) -> List[PDFChunk]:
        """Load and chunk a single PDF"""
        chunks = []
        reader = PdfReader(pdf_path)
        filename = os.path.basename(pdf_path)
        
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            
            if not text.strip():
                continue
            
            # Split text into chunks
            text_chunks = self.text_splitter.split_text(text)
            
            for chunk_num, chunk_text in enumerate(text_chunks):
                chunk_id = f"{filename}_{page_num}_{chunk_num}"
                donor = self._detect_donor_from_text(chunk_text, self._infer_donor(filename))
                
                pdf_chunk = PDFChunk(
                    text=chunk_text,
                    source=filename,
                    page=page_num + 1,  # 1-indexed for humans
                    chunk_id=chunk_id,
                    metadata={
                        "source": filename,
                        "page": page_num + 1,
                        "type": "guidelines",
                        "donor": donor
                    }
                )
                chunks.append(pdf_chunk)
        
        return chunks
    
    def _infer_donor(self, filename: str) -> str:
        """Infer donor from filename"""
        filename_lower = filename.lower()
        if "usaid" in filename_lower:
            return "USAID"
        if "standard-provisions" in filename_lower or "non-us" in filename_lower:
            return "USAID"
        elif "worldbank" in filename_lower or "gpsa" in filename_lower:
            return "World Bank"
        elif "gates" in filename_lower:
            return "Gates Foundation"
        elif "eu" in filename_lower or "european" in filename_lower:
            return "European Union"
        else:
            return "Unknown"

    def _detect_donor_from_text(self, text: str, default: str) -> str:
        """Infer donor from chunk text as a fallback."""
        blob = text.lower()
        if "usaid" in blob:
            return "USAID"
        if "world bank" in blob or "gpsa" in blob:
            return "World Bank"
        if "gates foundation" in blob:
            return "Gates Foundation"
        if "european union" in blob or "eu " in blob:
            return "European Union"
        return default

    def _extract_terms(self, question: str) -> List[str]:
        """Pull out salient keywords from the question for keyword-augmented recall."""
        raw = [w.strip("?.,;:!()[]{}\"'\n\t ") for w in question.lower().split()]
        terms = []
        keywords = {
            "usaid", "gpsa", "world", "bank", "indirect", "cost", "rate", "%", "percent",
            "procurement", "competitive", "quotes", "quote", "eligible", "eligibility",
            "report", "reports", "progress", "financial",
        }
        for word in raw:
            if word in keywords:
                terms.append(word)
        if "%" in question:
            terms.append("%")
        # de-dup while preserving order
        seen = set()
        uniq = []
        for t in terms:
            if t in seen:
                continue
            seen.add(t)
            uniq.append(t)
        return uniq
    
    def build_vector_store(self, persist_directory: str | None = "./chroma_db", reset: bool = True):
        """Build vector store from loaded documents"""
        if not self.documents:
            raise ValueError("No documents loaded. Call load_pdfs_from_folder first.")

        if persist_directory and reset and os.path.exists(persist_directory):
            # Remove stale cache to keep metadata in sync across runs
            import shutil
            shutil.rmtree(persist_directory, ignore_errors=True)
        
        # Convert to LangChain Documents
        lc_documents = []
        for chunk in self.documents:
            lc_doc = Document(
                page_content=chunk.text,
                metadata={
                    "source": chunk.source,
                    "page": chunk.page,
                    "chunk_id": chunk.chunk_id,
                    "donor": chunk.metadata["donor"]
                }
            )
            lc_documents.append(lc_doc)
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=lc_documents,
            embedding=self.embeddings,
            persist_directory=persist_directory
        )
        
        return self.vectorstore
    
    def create_retriever(self, k: int = 10, fetch_k: int | None = None):
        """Create a retriever (no LLM) using MMR for diversity."""
        if not self.vectorstore:
            raise ValueError("Vector store not built. Call build_vector_store first.")
        self.k = k
        self.fetch_k = fetch_k or max(k * 3, k + 2)
        # Store retriever for API symmetry even though we call vectorstore directly for scores
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": self.k, "fetch_k": self.fetch_k}
        )
        return self.retriever
    
    def answer(self, question: str) -> Dict[str, Any]:
        """Answer a question using retrieval only (no LLM/API)."""
        if not self.retriever:
            raise ValueError("Retriever not created. Call create_retriever first.")

        # Use MMR with scores to avoid duplicate near-identical chunks and surface broader coverage
        docs = self.vectorstore.max_marginal_relevance_search(
            question,
            k=self.k,
            fetch_k=self.fetch_k,
        )

        # Keyword augmentation: ensure chunks that directly contain key terms are available
        must_terms = self._extract_terms(question)
        if must_terms:
            added = 0
            for chunk in self.documents:
                blob = chunk.text.lower()
                if any(term == "%" and "%" in chunk.text for term in must_terms) or any(term in blob for term in must_terms if term != "%"):
                    doc = Document(
                        page_content=chunk.text,
                        metadata={
                            "source": chunk.source,
                            "page": chunk.page,
                            "chunk_id": chunk.chunk_id,
                            "donor": chunk.metadata.get("donor", "Unknown"),
                        },
                    )
                    docs.append(doc)
                    added += 1
                    if added >= 3:
                        break

        # Drop accidental duplicate chunks while preserving order
        seen_ids = set()
        results: List[Tuple[Document, float]] = []
        for doc in docs:
            chunk_id = doc.metadata.get("chunk_id")
            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)
            results.append((doc, None))

        # Improved summary: find best sentence matching keywords
        summary = ""
        if results:
            import re
            
            # Extract keywords from question (ignore stop words roughly)
            q_words = set(re.findall(r"\w+", question.lower()))
            stop_words = {"what", "is", "the", "for", "of", "to", "in", "a", "an", "and", "or", "by", "does", "do", "required", "requirements"}
            q_keywords = q_words - stop_words
            
            candidates = []
            # Collect sentences from top 3 docs
            for doc_tuple in results[:3]: 
                doc_obj = doc_tuple[0] # results is list of (doc, score) or (doc, None)
                # split on punctuation
                sentences = re.split(r'(?<=[.!?])\s+', doc_obj.page_content.replace('\n', ' '))
                for sent in sentences:
                    if len(sent.strip()) > 30: 
                        candidates.append(sent.strip())

            best_sent = ""
            best_score = -1
            
            for sent in candidates:
                s_lower = sent.lower()
                # Score 1: keyword matches
                score = sum(1 for k in q_keywords if k in s_lower)
                # Score 2: high information density indicators
                if any(x in s_lower for x in ["must", "shall", "eligible", "rate", "%", "percent", "guidelines"]):
                    score += 1
                
                if score > best_score:
                    best_score = score
                    best_sent = sent
            
            summary = best_sent if best_sent else (results[0][0].page_content[:200] + "...")
            if summary and not summary.endswith(('.', '!', '?')):
                summary += "."

        sources = []
        for doc, score in results:
            snippet_text = doc.page_content
            if must_terms:
                lower = snippet_text.lower()
                positions = [lower.find(term) for term in must_terms if lower.find(term) != -1]
                if positions:
                    pos = min(positions)
                    start = max(0, pos - 80)
                    end = min(len(snippet_text), pos + 160)
                    snippet_text = snippet_text[start:end]
            snippet = snippet_text
            if len(snippet) > 220:
                snippet = snippet[:220] + "..."
            sources.append({
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "Unknown"),
                "donor": doc.metadata.get("donor", "Unknown"),
                "score": round(score, 4) if isinstance(score, (float, int)) else score,
                "snippet": snippet
            })

        return {
            "question": question,
            "summary": summary,
            "sources": sources,
        }

# Simple test
if __name__ == "__main__":
    rag = PDFRAG()
    rag.load_pdfs_from_folder("./kb")
    rag.build_vector_store()
    rag.create_retriever()
    
    test_question = "What is the indirect cost rate for non-US NGOs according to USAID?"
    answer = rag.answer(test_question)
    
    print("Question:", answer["question"])
    print("\nSummary:", answer["summary"])
    print("\nSources:")
    for src in answer["sources"]:
        print(f"  - {src['source']} (Page {src['page']}) - {src['donor']}")