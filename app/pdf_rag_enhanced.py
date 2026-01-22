# app/pdf_rag_enhanced.py
# Enhanced RAG with optional free LLM support (Ollama)
from __future__ import annotations
import os
import glob
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

# PDF processing
from pypdf import PdfReader

# LangChain (offline) components
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# Optional LLM support (free via Ollama)
try:
    # Try new langchain-ollama package first
    try:
        from langchain_ollama import OllamaLLM
        OLLAMA_AVAILABLE = True
        OLLAMA_NEW = True
    except ImportError:
        # Fallback to old langchain-community
        try:
            from langchain_community.llms import Ollama
            OLLAMA_AVAILABLE = True
            OLLAMA_NEW = False
        except ImportError:
            OLLAMA_AVAILABLE = False
            OLLAMA_NEW = False
except ImportError:
    OLLAMA_AVAILABLE = False
    OLLAMA_NEW = False

try:
    from langchain_huggingface import HuggingFacePipeline
    from transformers import pipeline
    HF_PIPELINE_AVAILABLE = True
except ImportError:
    HF_PIPELINE_AVAILABLE = False


@dataclass
class PDFChunk:
    text: str
    source: str
    page: int
    chunk_id: str
    metadata: Dict[str, Any]


class EnhancedPDFRAG:
    def __init__(self, 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 use_llm: bool = True,
                 llm_type: str = "ollama"):  # "ollama", "huggingface", or "none"
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Local embeddings (no API key required)
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vectorstore = None
        self.documents: List[PDFChunk] = []
        self.retriever = None
        self.k = 5
        self.fetch_k = 15
        
        # LLM setup (free options)
        self.use_llm = use_llm
        self.llm_type = llm_type
        self.llm = None
        
        if use_llm:
            self._setup_llm()
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def _setup_llm(self):
        """Setup free LLM (Ollama or HuggingFace)"""
        if self.llm_type == "ollama" and OLLAMA_AVAILABLE:
            try:
                # Try common Ollama models (free, local)
                models_to_try = ["llama3.2", "mistral", "llama3.1", "phi3"]
                for model in models_to_try:
                    try:
                        if OLLAMA_NEW:
                            self.llm = OllamaLLM(model=model, temperature=0.1)
                        else:
                            self.llm = Ollama(model=model, temperature=0.1)
                        print(f"[OK] Using Ollama with model: {model}")
                        return
                    except:
                        continue
                print("[WARN] Ollama installed but no models found. Run: ollama pull llama3.2")
            except Exception as e:
                print(f"[WARN] Ollama not available: {e}")
        
        elif self.llm_type == "huggingface" and HF_PIPELINE_AVAILABLE:
            try:
                # Use a small free model from HuggingFace
                pipe = pipeline(
                    "text-generation",
                    model="microsoft/Phi-3-mini-4k-instruct",
                    device_map="auto",
                    model_kwargs={"torch_dtype": "float16"}
                )
                self.llm = HuggingFacePipeline(pipeline=pipe)
                print("[OK] Using HuggingFace local model")
                return
            except Exception as e:
                print(f"[WARN] HuggingFace pipeline not available: {e}")
        
        # Fallback: no LLM
        print("[INFO] Using retrieval-only mode (no LLM)")
        self.use_llm = False
    
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
                    page=page_num + 1,
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
        if "usaid" in filename_lower or "standard-provisions" in filename_lower or "non-us" in filename_lower:
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
        """Pull out salient keywords from the question"""
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
            import shutil
            shutil.rmtree(persist_directory, ignore_errors=True)
        
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
        
        self.vectorstore = Chroma.from_documents(
            documents=lc_documents,
            embedding=self.embeddings,
            persist_directory=persist_directory
        )
        
        return self.vectorstore
    
    def create_retriever(self, k: int = 10, fetch_k: int | None = None):
        """Create a retriever using MMR for diversity."""
        if not self.vectorstore:
            raise ValueError("Vector store not built. Call build_vector_store first.")
        self.k = k
        self.fetch_k = fetch_k or max(k * 3, k + 2)
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": self.k, "fetch_k": self.fetch_k}
        )
        return self.retriever
    
    def _generate_llm_summary(self, question: str, retrieved_texts: List[str]) -> str:
        """Generate summary using free LLM"""
        if not self.llm:
            return ""
        
        # Combine retrieved texts - use more context (up to 10 sources, 1200 chars each)
        context = "\n\n".join([f"[Source {i+1}]: {text[:1200]}" for i, text in enumerate(retrieved_texts[:10])])
        
        prompt = f"""You are an expert assistant helping NGOs understand donor proposal requirements. Answer the question based ONLY on the provided document excerpts.

Question: {question}

Relevant document excerpts:
{context}

Instructions:
- CAREFULLY search through ALL excerpts for specific numbers, rates, percentages, amounts, deadlines, and requirements
- IMPORTANT: If the question asks about a "rate", look for explicit rate statements like "X percent rate", "X% rate", "de minimis rate of X%", NOT percentages mentioned in other contexts (like "20 percent or more" which refers to cost changes, not the rate itself)
- If you find exact rate numbers (e.g., "15% rate", "15 percent de minimis rate", "up to 15 percent"), you MUST include them in your answer
- Look for phrases like "up to X percent", "X% rate", "de minimis rate", "NICRA", "2 CFR 200.414"
- DO NOT confuse percentages mentioned in conditional statements (like "if costs change by 20%") with the actual rate being asked about
- If the information is not in the excerpts, state that clearly - do not guess or infer rates from unrelated percentages
- Be precise and cite specific details when available
- Keep the answer concise but complete (2-4 sentences)

Answer:"""
        
        try:
            response = self.llm.invoke(prompt)
            # Clean up response
            response = response.strip()
            # Remove common LLM artifacts
            if response.startswith("Answer:"):
                response = response[7:].strip()
            # Remove any trailing "Answer:" or similar artifacts
            if "\nAnswer:" in response:
                response = response.split("\nAnswer:")[0].strip()
            return response
        except Exception as e:
            print(f"⚠️ LLM generation failed: {e}")
            return ""
    
    def _improved_heuristic_summary(self, question: str, results: List[Tuple[Document, Any]]) -> str:
        """Improved heuristic-based summary (no LLM needed)"""
        if not results:
            return ""
        
        import re
        
        # Extract keywords from question
        q_words = set(re.findall(r"\w+", question.lower()))
        stop_words = {"what", "is", "the", "for", "of", "to", "in", "a", "an", "and", "or", "by", "does", "do", "required", "requirements", "are", "can", "how"}
        q_keywords = q_words - stop_words
        
        # Collect and score sentences
        candidates = []
        for doc_tuple in results[:5]:  # Check top 5 docs
            doc_obj = doc_tuple[0]
            text = doc_obj.page_content.replace('\n', ' ')
            sentences = re.split(r'(?<=[.!?])\s+', text)
            
            for sent in sentences:
                sent = sent.strip()
                if len(sent) < 30 or len(sent) > 300:
                    continue
                
                s_lower = sent.lower()
                # Score based on keyword matches
                score = sum(2 for k in q_keywords if k in s_lower)
                
                # Bonus for important indicators
                if any(x in s_lower for x in ["must", "shall", "required", "eligible", "rate", "%", "percent", "guidelines", "policy"]):
                    score += 3
                
                # Bonus for numbers (often indicate requirements)
                if re.search(r'\d+', s_lower):
                    score += 1
                
                candidates.append((sent, score))
        
        # Get top 2-3 sentences and combine them intelligently
        candidates.sort(key=lambda x: x[1], reverse=True)
        
        if not candidates:
            return results[0][0].page_content[:200] + "..."
        
        # Take top sentences, avoiding duplicates
        selected = []
        seen_words = set()
        for sent, score in candidates[:5]:
            words = set(re.findall(r'\w+', sent.lower()))
            # Check overlap with already selected sentences
            overlap = len(words & seen_words) / max(len(words), 1)
            if overlap < 0.7:  # Not too similar
                selected.append(sent)
                seen_words.update(words)
                if len(selected) >= 2:
                    break
        
        summary = " ".join(selected)
        
        # Clean up
        summary = re.sub(r'\s+', ' ', summary).strip()
        if summary and not summary.endswith(('.', '!', '?')):
            summary += "."
        
        return summary[:400]  # Limit length
    
    def answer(self, question: str) -> Dict[str, Any]:
        """Answer a question with optional LLM enhancement"""
        if not self.retriever:
            raise ValueError("Retriever not created. Call create_retriever first.")

        # Retrieve relevant documents - increase k for better coverage
        docs = self.vectorstore.max_marginal_relevance_search(
            question,
            k=min(self.k + 3, 10),  # Get more initial results
            fetch_k=self.fetch_k,
        )

        # Enhanced keyword augmentation - search for percentage/rate related chunks
        must_terms = self._extract_terms(question)
        added_chunk_ids = set()  # Track added chunks to avoid duplicates
        if must_terms:
            added = 0
            # Also search for percentage-related terms - prioritize "de minimis" chunks
            rate_keywords = ["de minimis", "percent", "%", "rate", "nicra", "15", "fifteen", "2 cfr 200.414"]
            # Check if question is about rates
            is_rate_question = any(t in ["rate", "percent", "%", "cost rate"] for t in must_terms) or "rate" in question.lower()
            
            # First pass: prioritize chunks with "de minimis" if it's a rate question
            if is_rate_question:
                for chunk in self.documents:
                    blob = chunk.text.lower()
                    if ("de minimis" in blob or "minimis" in blob) and chunk.chunk_id not in added_chunk_ids:
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
                        added_chunk_ids.add(chunk.chunk_id)
                        added += 1
                        if added >= 3:
                            break
            
            # Second pass: general rate/percentage keywords
            for chunk in self.documents:
                if chunk.chunk_id in added_chunk_ids:
                    continue
                blob = chunk.text.lower()
                # Check for direct term matches
                term_match = any(term == "%" and "%" in chunk.text for term in must_terms) or any(term in blob for term in must_terms if term != "%")
                # Also check for rate/percentage keywords if question is about rates
                rate_match = False
                if is_rate_question:
                    rate_match = any(kw in blob for kw in rate_keywords)
                
                if term_match or rate_match:
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
                    added_chunk_ids.add(chunk.chunk_id)
                    added += 1
                    if added >= 8:  # Increase total to 8
                        break

        # Remove duplicates
        seen_ids = set()
        results: List[Tuple[Document, float]] = []
        for doc in docs:
            chunk_id = doc.metadata.get("chunk_id")
            if chunk_id in seen_ids:
                continue
            seen_ids.add(chunk_id)
            results.append((doc, None))

        # Generate summary (LLM or improved heuristic)
        summary = ""
        if results:
            # Use more chunks for LLM (up to 10) to get better context
            retrieved_texts = [doc.page_content for doc, _ in results[:10]]
            
            if self.use_llm and self.llm:
                summary = self._generate_llm_summary(question, retrieved_texts)
            
            # Fallback to improved heuristic if LLM failed or not available
            if not summary:
                summary = self._improved_heuristic_summary(question, results)
        
        # Format sources
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
            "llm_used": self.use_llm and self.llm is not None
        }
