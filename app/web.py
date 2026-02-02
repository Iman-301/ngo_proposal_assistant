"""
Simplified web interface for NGO Proposal Assistant
"""
import os
import tempfile
from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from werkzeug.utils import secure_filename

from .config import Config
from .pdf_rag import NGOProposalRAG

app = Flask(__name__)
app.secret_key = "ngo_assistant_secret_key"
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB

# Global RAG instance
rag = None
current_pdfs = []


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in {"pdf"}


def init_rag_with_pdfs(pdf_paths):
    """Initialize RAG with given PDFs"""
    global rag
    
    # Create new RAG instance
    rag = NGOProposalRAG(
        chunk_size=800,
        chunk_overlap=150,
        use_llm=True,  # Enable LLM if available
        llm_model="llama3.2"
    )
    
    # Load PDFs
    total_chunks = 0
    for pdf_path in pdf_paths:
        if os.path.exists(pdf_path):
            # Load single PDF
            rag.load_pdfs_from_folder(os.path.dirname(pdf_path))
            total_chunks = len(rag.chunks)
    
    if total_chunks > 0:
        # Build vector store
        rag.build_vector_store()
        rag.create_retriever(k=8)
        return True, f"Loaded {total_chunks} text chunks from {len(pdf_paths)} PDF(s)"
    else:
        return False, "No PDF content could be loaded"


@app.route("/", methods=["GET", "POST"])
def index():
    global rag, current_pdfs
    
    if request.method == "POST":
        # Check if this is a PDF upload
        if "pdf_file" in request.files:
            file = request.files["pdf_file"]
            
            if file and file.filename and allowed_file(file.filename):
                # Save uploaded file
                filename = secure_filename(file.filename)
                upload_dir = tempfile.mkdtemp(prefix="ngo_uploads_")
                os.makedirs(upload_dir, exist_ok=True)
                filepath = os.path.join(upload_dir, filename)
                file.save(filepath)
                
                # Initialize RAG with this PDF
                success, message = init_rag_with_pdfs([filepath])
                
                if success:
                    flash(f"✅ {message}")
                    current_pdfs = [filepath]
                else:
                    flash(f"❌ {message}")
            
            else:
                flash("❌ Please upload a valid PDF file")
        
        # Check if this is a question
        elif "question" in request.form:
            question = request.form.get("question", "").strip()
            
            if not question:
                flash("❌ Please enter a question")
            elif rag is None:
                flash("❌ Please upload a PDF first")
            else:
                try:
                    # Get answer
                    result = rag.answer(question)
                    
                    return render_template(
                        "index.html",
                        question=question,
                        result=result,
                        has_pdf=rag is not None,
                        pdf_count=len(current_pdfs)
                    )
                    
                except Exception as e:
                    flash(f"❌ Error processing question: {str(e)}")
    
    return render_template(
        "index.html",
        has_pdf=rag is not None,
        pdf_count=len(current_pdfs)
    )


@app.route("/clear", methods=["POST"])
def clear_session():
    """Clear current session"""
    global rag, current_pdfs
    rag = None
    current_pdfs = []
    flash("✅ Session cleared. Ready for new uploads.")
    return redirect(url_for("index"))


@app.route("/api/ask", methods=["POST"])
def api_ask():
    """API endpoint for asking questions"""
    global rag
    
    if rag is None:
        return jsonify({
            "success": False,
            "error": "No PDF loaded. Please upload a PDF first."
        })
    
    data = request.get_json()
    question = data.get("question", "").strip()
    
    if not question:
        return jsonify({"success": False, "error": "No question provided"})
    
    try:
        result = rag.answer(question)
        return jsonify({
            "success": True,
            "result": result
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })


if __name__ == "__main__":
    # Ensure templates directory exists
    os.makedirs("templates", exist_ok=True)
    
    print("Starting NGO Proposal Assistant Web UI...")
    print("Access at: http://localhost:5001")
    print("\nMake sure Ollama is running for LLM support:")
    print("  ollama serve")
    print("  ollama pull llama3.2")
    
    app.run(debug=True, port=5001)