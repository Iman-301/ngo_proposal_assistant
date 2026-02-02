"""
Improved PDF RAG system with better accuracy and query classification
"""
from __future__ import annotations
import os
import re
import glob
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field

# PDF processing
from pypdf import PdfReader

# LangChain components
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# LLM support
try:
    from langchain_ollama import OllamaLLM
    OLLAMA_AVAILABLE = True
except ImportError:
    try:
        from langchain_community.llms import Ollama
        OLLAMA_AVAILABLE = True
    except ImportError:
        OLLAMA_AVAILABLE = False


@dataclass
class PDFChunk:
    """Enhanced chunk with better metadata"""
    text: str
    source: str
    page: int
    chunk_id: str
    metadata: Dict[str, Any]
    section_type: str = "general"
    keywords: List[str] = field(default_factory=list)
    has_requirements: bool = False


class NGOProposalRAG:
    """Unified RAG system for NGO proposal assistance"""
    
    def __init__(self, 
                 chunk_size: int = 800,
                 chunk_overlap: int = 150,
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 use_llm: bool = False,
                 llm_model: str = "llama3.2"):
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.vectorstore = None
        self.chunks: List[PDFChunk] = []
        self.retriever = None
        
        # LLM
        self.use_llm = use_llm
        self.llm_model = llm_model
        self.llm = None
        
        if use_llm and OLLAMA_AVAILABLE:
            self._init_llm()
        
        # Text splitter with better settings for structured documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""],
            keep_separator=True
        )
    
    def _init_llm(self):
        """Initialize LLM if available"""
        try:
            # Try Ollama first
            if "llama3.2" in self.llm_model:
                try:
                    from langchain_ollama import OllamaLLM
                    self.llm = OllamaLLM(model=self.llm_model, temperature=0.1)
                except:
                    from langchain_community.llms import Ollama
                    self.llm = Ollama(model=self.llm_model, temperature=0.1)
                print(f"[OK] Using Ollama with model: {self.llm_model}")
                return
        except Exception as e:
            print(f"[WARN] LLM initialization failed: {e}")
            self.llm = None
            self.use_llm = False
    
    def _extract_section_info(self, text: str) -> Tuple[str, List[str]]:
        """Extract section type and keywords from text"""
        text_lower = text.lower()
        section_type = "general"
        
        # Detect section type
        if any(term in text_lower for term in ["eligible", "eligibility", "who can apply", "qualify"]):
            section_type = "eligibility"
        elif any(term in text_lower for term in ["rate", "%", "percent", "cost", "funding", "budget", "amount"]):
            section_type = "financial"
        elif any(term in text_lower for term in ["procurement", "purchase", "buy", "contract"]):
            section_type = "procurement"
        elif any(term in text_lower for term in ["report", "submit", "deadline", "due", "timeline"]):
            section_type = "reporting"
        elif any(term in text_lower for term in ["requirement", "must", "shall", "should", "required"]):
            section_type = "requirements"
        
        # Extract keywords
        keywords = []
        ngo_keywords = {
            "grant", "funding", "proposal", "budget", "cso", "ngo", "organization",
            "eligible", "requirement", "procurement", "report", "deadline", "timeline",
            "amount", "rate", "percent", "cost", "financial", "compliance", "audit",
            "subgrant", "partner", "mentor", "application", "submit", "guideline"
        }
        
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text_lower)
        for word in words:
            if word in ngo_keywords and word not in keywords:
                keywords.append(word)
        
        return section_type, keywords
    
    def load_pdfs_from_folder(self, folder_path: str) -> int:
        """Load all PDFs from a folder with enhanced metadata"""
        pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
        all_chunks = []
        
        for pdf_file in pdf_files:
            chunks = self._process_pdf(pdf_file)
            all_chunks.extend(chunks)
        
        self.chunks = all_chunks
        return len(all_chunks)
    
    def _process_pdf(self, pdf_path: str) -> List[PDFChunk]:
        """Process a single PDF with enhanced metadata"""
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
                
                # Extract section info
                section_type, keywords = self._extract_section_info(chunk_text)
                
                # Detect donor
                donor = self._detect_donor(filename, chunk_text)
                
                # Check if this contains requirements
                has_requirements = any(marker in chunk_text.lower() 
                                     for marker in ["must", "shall", "required", "should"])
                
                pdf_chunk = PDFChunk(
                    text=chunk_text,
                    source=filename,
                    page=page_num + 1,
                    chunk_id=chunk_id,
                    metadata={
                        "source": filename,
                        "page": page_num + 1,
                        "donor": donor,
                        "section_type": section_type,
                        "keywords": ", ".join(keywords),
                        "has_requirements": has_requirements
                    },
                    section_type=section_type,
                    keywords=keywords,
                    has_requirements=has_requirements
                )
                chunks.append(pdf_chunk)
        
        return chunks
    
    def _detect_donor(self, filename: str, text: str) -> str:
        """Detect donor organization from filename and text"""
        filename_lower = filename.lower()
        text_lower = text.lower()
        
        # Check filename first
        if "usaid" in filename_lower or "standard-provisions" in filename_lower:
            return "USAID"
        elif "gpsa" in filename_lower or "worldbank" in filename_lower:
            return "World Bank"
        elif "eu" in filename_lower or "european" in filename_lower:
            return "European Union"
        
        # Check text content
        if "usaid" in text_lower:
            return "USAID"
        elif "world bank" in text_lower or "gpsa" in text_lower:
            return "World Bank"
        
        return "Unknown"
    
    def build_vector_store(self, persist_directory: str = "./chroma_db"):
        """Build vector store from chunks"""
        if not self.chunks:
            raise ValueError("No documents loaded")
        
        # Clean old vector store if it exists
        if os.path.exists(persist_directory):
            import shutil
            shutil.rmtree(persist_directory, ignore_errors=True)
        
        # Convert to LangChain documents
        documents = []
        for chunk in self.chunks:
            doc = Document(
                page_content=chunk.text,
                metadata=chunk.metadata
            )
            documents.append(doc)
        
        # Create vector store
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=persist_directory,
            collection_metadata={"hnsw:space": "cosine"}
        )
        
        return self.vectorstore
    
    def create_retriever(self, k: int = 8):
        """Create a retriever with MMR for diversity"""
        if not self.vectorstore:
            raise ValueError("Vector store not built")
        
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": k,
                "fetch_k": k * 2,
                "lambda_mult": 0.7  # Balance relevance vs diversity
            }
        )
        return self.retriever
    
    def _classify_question(self, question: str) -> str:
        """Classify question type for better retrieval"""
        question_lower = question.lower()
        
        classification_rules = {
            "eligibility": ["eligible", "who can apply", "qualify", "who is eligible", "qualification"],
            "financial": ["rate", "%", "percent", "cost", "budget", "amount", "funding", "grant amount", "maximum", "minimum"],
            "procurement": ["procure", "purchase", "buy", "equipment", "contract", "bidding"],
            "reporting": ["report", "submit", "deadline", "due date", "timeline", "when to submit"],
            "requirements": ["requirement", "must", "shall", "should", "need to", "required to"],
            "application": ["how to apply", "application process", "submit proposal", "apply for"],
            "compliance": ["comply", "compliance", "audit", "review", "check"]
        }
        
        for category, keywords in classification_rules.items():
            if any(keyword in question_lower for keyword in keywords):
                return category
        
        return "general"
    
    def _enhance_search(self, question: str, k: int = 10) -> List[Document]:
        """Enhanced search with query classification and filtering"""
        question_type = self._classify_question(question)
        
        # Get initial semantic search results
        try:
            docs_with_scores = self.vectorstore.similarity_search_with_relevance_scores(
                question,
                k=k * 2
            )
        except:
            docs_with_scores = [(doc, 0.5) for doc in self.vectorstore.similarity_search(question, k=k * 2)]
        
        # Re-rank based on question type
        enhanced_results = []
        for doc, score in docs_with_scores:
            enhanced_score = score
            
            # Boost for matching question type
            doc_section_type = doc.metadata.get("section_type", "general")
            if question_type == doc_section_type:
                enhanced_score *= 1.5
            
            # Boost for requirement markers in eligibility questions
            if question_type == "eligibility" and doc.metadata.get("has_requirements", False):
                enhanced_score *= 1.3
            
            # Boost for donor relevance
            if "gpsa" in question.lower() and "World Bank" in doc.metadata.get("donor", ""):
                enhanced_score *= 1.2
            elif "usaid" in question.lower() and "USAID" in doc.metadata.get("donor", ""):
                enhanced_score *= 1.2
            
            enhanced_results.append((doc, enhanced_score))
        
        # Sort by enhanced score
        enhanced_results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top k documents
        return [doc for doc, score in enhanced_results[:k]]
    
    def _extract_key_sentences(self, text: str, question: str, max_sentences: int = 3) -> str:
        """Extract the most relevant sentences for the question"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Score each sentence
        scored_sentences = []
        question_words = set(re.findall(r'\w+', question.lower()))
        stop_words = {"what", "is", "the", "for", "of", "to", "in", "a", "an", "and", "or"}
        keywords = [w for w in question_words if w not in stop_words and len(w) > 2]
        
        for sentence in sentences:
            if len(sentence.strip()) < 20:
                continue
            
            s_lower = sentence.lower()
            score = 0
            
            # Score keyword matches
            for keyword in keywords:
                if keyword in s_lower:
                    score += 2
            
            # Boost for requirement indicators
            if any(marker in s_lower for marker in ["must", "shall", "required", "should"]):
                score += 3
            
            # Boost for numerical information
            if re.search(r'\d+', s_lower):
                score += 1
            
            if score > 0:
                scored_sentences.append((sentence.strip(), score))
        
        # Sort by score and take top sentences
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = [s for s, _ in scored_sentences[:max_sentences]]
        
        # Combine sentences intelligently
        if top_sentences:
            return " ".join(top_sentences)
        else:
            # Fallback: return first meaningful sentences
            meaningful = [s.strip() for s in sentences if len(s.strip()) > 40]
            return " ".join(meaningful[:2]) if meaningful else sentences[0]
    
    def _generate_answer_with_llm(self, question: str, context_docs: List[Document]) -> str:
        """Generate answer using LLM with improved prompt"""
        if not self.llm:
            return self._generate_heuristic_answer(question, context_docs)
        
        # Prepare context
        context_parts = []
        for i, doc in enumerate(context_docs[:5], 1):
            source_info = f"Source {i} (Page {doc.metadata.get('page', 'N/A')} - {doc.metadata.get('donor', 'Unknown')}):"
            context_parts.append(source_info)
            context_parts.append(doc.page_content[:800])  # Limit context length
            context_parts.append("")
        
        context = "\n".join(context_parts)
        
        # Create focused prompt based on question type
        question_type = self._classify_question(question)
        
        if question_type == "eligibility":
            prompt = f"""You are an expert NGO grant advisor. Answer the question based ONLY on the provided sources.

QUESTION: {question}

SOURCES:
{context}

INSTRUCTIONS FOR ELIGIBILITY QUESTIONS:
1. List ALL eligibility requirements mentioned in the sources
2. Be specific about who qualifies and who doesn't
3. Mention any required documentation or criteria
4. If there are different types of eligibility (e.g., for different organization types), explain each
5. If certain organizations are explicitly excluded, mention that
6. Use bullet points if there are multiple requirements
7. Only use information from the provided sources

ANSWER:"""
        
        else:
            prompt = f"""You are an expert NGO grant advisor. Answer the question based ONLY on the provided sources.

QUESTION: {question}

SOURCES:
{context}

INSTRUCTIONS:
1. Answer concisely (2-4 sentences)
2. Use ONLY information from the provided sources
3. Be specific with numbers, dates, and requirements
4. If the information is not in the sources, say: "Based on the available documents, this information is not specified"
5. Mention the relevant donor/organization when applicable

ANSWER:"""
        
        try:
            response = self.llm.invoke(prompt)
            response = response.strip()
            
            # Clean up
            if response.startswith("Answer:"):
                response = response[7:].strip()
            
            return response
        except Exception as e:
            print(f"[LLM Error] {e}")
            return self._generate_heuristic_answer(question, context_docs)
    
    def _generate_heuristic_answer(self, question: str, context_docs: List[Document]) -> str:
        """Generate answer without LLM using intelligent extraction"""
        if not context_docs:
            return "No relevant information found in the documents."
        
        question_type = self._classify_question(question)
        
        # For eligibility questions, look for requirement sentences
        if question_type == "eligibility":
            eligibility_sentences = []
            for doc in context_docs[:3]:
                text = doc.page_content
                sentences = re.split(r'(?<=[.!?])\s+', text)
                
                for sentence in sentences:
                    s_lower = sentence.lower()
                    # Look for eligibility-related sentences
                    if (("eligible" in s_lower or "qualif" in s_lower or 
                         "must" in s_lower or "required" in s_lower) and
                        len(sentence.strip()) > 30):
                        eligibility_sentences.append(sentence.strip())
            
            if eligibility_sentences:
                # Deduplicate and combine
                unique_sentences = []
                seen_words = set()
                for sentence in eligibility_sentences:
                    words = set(re.findall(r'\w+', sentence.lower()))
                    if len(words & seen_words) / max(len(words), 1) < 0.5:
                        unique_sentences.append(sentence)
                        seen_words.update(words)
                        if len(unique_sentences) >= 3:
                            break
                
                if unique_sentences:
                    donor = context_docs[0].metadata.get("donor", "Unknown")
                    return f"According to {donor} guidelines: " + " ".join(unique_sentences)
        
        # General fallback: extract key sentences from top document
        top_doc = context_docs[0]
        answer = self._extract_key_sentences(top_doc.page_content, question)
        donor = top_doc.metadata.get("donor", "Unknown")
        
        return f"According to {donor} guidelines: {answer}"
    
    def answer(self, question: str) -> Dict[str, Any]:
        """Main answer method with improved accuracy"""
        if not self.vectorstore:
            raise ValueError("Vector store not built. Call build_vector_store() first.")
        
        # Enhanced search with classification
        relevant_docs = self._enhance_search(question, k=8)
        
        # Generate answer
        if self.use_llm and self.llm:
            summary = self._generate_answer_with_llm(question, relevant_docs)
            llm_used = True
        else:
            summary = self._generate_heuristic_answer(question, relevant_docs)
            llm_used = False
        
        # Format sources
        sources = []
        for i, doc in enumerate(relevant_docs[:5], 1):
            # Extract relevant snippet
            snippet = self._extract_relevant_snippet(doc.page_content, question)
            
            sources.append({
                "source": doc.metadata.get("source", "Unknown"),
                "page": doc.metadata.get("page", "Unknown"),
                "donor": doc.metadata.get("donor", "Unknown"),
                "section_type": doc.metadata.get("section_type", "general"),
                "snippet": snippet
            })
        
        return {
            "question": question,
            "summary": summary,
            "sources": sources,
            "llm_used": llm_used,
            "question_type": self._classify_question(question)
        }
    
    def _extract_relevant_snippet(self, text: str, question: str, max_length: int = 200) -> str:
        """Extract the most relevant part of the text for the question"""
        # Find where question keywords appear
        question_words = set(re.findall(r'\w+', question.lower()))
        stop_words = {"what", "is", "the", "for", "of", "to", "in", "a", "an", "and", "or"}
        keywords = [w for w in question_words if w not in stop_words and len(w) > 2]
        
        if not keywords:
            return text[:max_length] + ("..." if len(text) > max_length else "")
        
        text_lower = text.lower()
        positions = []
        
        for keyword in keywords:
            pos = text_lower.find(keyword)
            if pos != -1:
                positions.append(pos)
        
        if not positions:
            return text[:max_length] + ("..." if len(text) > max_length else "")
        
        # Extract around the first keyword
        start_pos = min(positions)
        start = max(0, start_pos - 80)
        end = min(len(text), start_pos + 120)
        
        snippet = text[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet = snippet + "..."
        
        return snippet


# Simple test
if __name__ == "__main__":
    rag = NGOProposalRAG(use_llm=False)
    
    # Test with sample PDFs
    rag.load_pdfs_from_folder("./kb")
    rag.build_vector_store()
    rag.create_retriever()
    
    test_questions = [
        "Who is eligible for GPSA grants?",
        "What is the maximum grant amount?",
        "What are the procurement requirements?"
    ]
    
    for q in test_questions:
        print(f"\nQuestion: {q}")
        result = rag.answer(q)
        print(f"Answer: {result['summary'][:200]}...")
        print(f"Question type: {result['question_type']}")
        print(f"Sources: {len(result['sources'])}")