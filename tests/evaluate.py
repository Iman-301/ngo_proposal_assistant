import json
import os
import sys
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.pdf_rag import NGOProposalRAG
from app.config import Config


def evaluate_pdf_rag():
    """Evaluate the RAG system with comprehensive test cases"""
    print("🚀 Evaluating Unified NGO Proposal Assistant RAG")
    
    # Initialize RAG with LLM support if available
    try:
        rag = NGOProposalRAG(
            use_llm=True,  # Try to use LLM if available
            llm_model="llama3.2",
            chunk_size=800,
            chunk_overlap=150
        )
        print("[INFO] Initialized RAG with LLM support")
    except Exception as e:
        print(f"[WARN] LLM initialization failed, using retrieval-only: {e}")
        rag = NGOProposalRAG(use_llm=False)
    
    kb_path = Config.resolve_kb_path(None)
    
    print("=" * 60)
    print("NGO PROPOSAL ASSISTANT - COMPREHENSIVE EVALUATION")
    print("=" * 60)
    print(f"Knowledge base: {kb_path}")
    
    # Load and build the system
    print("\n📚 Loading PDF documents...")
    doc_count = rag.load_pdfs_from_folder(kb_path)
    
    if doc_count == 0:
        print("❌ No documents found. Please add PDFs to the 'kb/' folder.")
        return None
    
    print(f"✅ Loaded {doc_count} text chunks from PDFs")
    
    print("\n🔨 Building vector database...")
    rag.build_vector_store()
    
    print("\n🔍 Creating retriever...")
    rag.create_retriever()
    
    print("\n" + "✅" * 20)
    print("SYSTEM READY FOR EVALUATION")
    print("✅" * 20 + "\n")
    
    # Comprehensive test cases (updated to be more accurate)
    test_cases = [
        {
            "question": "What is the indirect cost rate for non-US NGOs?",
            "expected_donor": "USAID",
            "expected_keywords": ["indirect", "cost", "rate", "percent", "%"],
            "category": "Financial Requirements",
            "difficulty": "Easy"
        },
        {
            "question": "Who is eligible for GPSA grants?",
            "expected_donor": "World Bank", 
            "expected_keywords": ["eligible", "organizations", "civil", "society", "cso"],
            "category": "Eligibility",
            "difficulty": "Easy"
        },
        {
            "question": "What are procurement rules for USAID grants?",
            "expected_donor": "USAID",
            "expected_keywords": ["procurement", "competitive", "quotes", "purchase"],
            "category": "Procurement",
            "difficulty": "Medium"
        },
        {
            "question": "What reports are required by World Bank GPSA?",
            "expected_donor": "World Bank",
            "expected_keywords": ["reports", "progress", "financial", "submit"],
            "category": "Reporting",
            "difficulty": "Easy"
        },
        {
            "question": "What is the maximum grant amount for GPSA projects?",
            "expected_donor": "World Bank",
            "expected_keywords": ["maximum", "grant", "amount", "$", "funding", "400000", "800000"],
            "category": "Financial Requirements",
            "difficulty": "Medium"
        },
        {
            "question": "What are financial reporting requirements?",
            "expected_donor": "USAID",  # Both have this, but USAID is more detailed
            "expected_keywords": ["financial", "report", "quarterly", "annual", "audit"],
            "category": "Reporting",
            "difficulty": "Medium"
        },
        {
            "question": "What are partnership arrangements for GPSA grants?",
            "expected_donor": "World Bank",
            "expected_keywords": ["partnership", "mentor", "implementing", "partner"],
            "category": "Administrative",
            "difficulty": "Medium"
        },
        {
            "question": "What are the selection criteria for GPSA grants?",
            "expected_donor": "World Bank",
            "expected_keywords": ["selection", "criteria", "evaluation", "review"],
            "category": "Application Process",
            "difficulty": "Medium"
        },
        {
            "question": "What types of organizations are eligible for GPSA grants?",
            "expected_donor": "World Bank",
            "expected_keywords": ["ngo", "cso", "organization", "non-profit", "foundation"],
            "category": "Eligibility",
            "difficulty": "Easy"
        },
        {
            "question": "What is the application process for GPSA grants?",
            "expected_donor": "World Bank",
            "expected_keywords": ["application", "process", "submit", "online", "deadline"],
            "category": "Application Process",
            "difficulty": "Medium"
        },
    ]

    print(f"Running {len(test_cases)} test cases...\n")
    
    results = []
    for i, test in enumerate(test_cases, 1):
        print(f"Test {i}/{len(test_cases)}: {test['question']}")
        print(f"  Category: {test['category']} | Difficulty: {test['difficulty']}")
        
        # Get answer from RAG system
        try:
            result = rag.answer(test["question"])
        except Exception as e:
            print(f"  ❌ Error answering question: {e}")
            result = {"summary": "", "sources": []}
        
        # Extract expected donors
        expected_donor = test["expected_donor"]
        
        # Check donor match
        found_donors = [src.get("donor", "") for src in result.get("sources", [])]
        donor_match = False
        for found in found_donors:
            if expected_donor.lower() in found.lower():
                donor_match = True
                break
        
        # Check keyword match
        blob = (result.get("summary", "") + "\n" + 
                "\n".join(src.get("snippet", "") for src in result.get("sources", []))).lower()
        
        hits = []
        misses = []
        for kw in test["expected_keywords"]:
            if kw.lower() in blob:
                hits.append(kw)
            else:
                misses.append(kw)
        
        # More lenient scoring
        keyword_match_ratio = len(hits) / len(test["expected_keywords"]) if test["expected_keywords"] else 1
        keyword_match = keyword_match_ratio >= 0.4
        
        # Combined pass/fail
        passed = donor_match and keyword_match
        
        # Store detailed results
        test_result = {
            "test_id": i,
            "question": test["question"],
            "category": test["category"],
            "difficulty": test["difficulty"],
            "passed": passed,
            "donor_match": donor_match,
            "keyword_match": keyword_match,
            "keyword_match_ratio": keyword_match_ratio,
            "matched_keywords": hits,
            "missing_keywords": misses,
            "expected_donor": expected_donor,
            "found_donors": list(set(found_donors)),
            "summary": result.get("summary", ""),
            "source_count": len(result.get("sources", [])),
            "sources_preview": [
                {
                    "donor": src.get("donor", "Unknown"),
                    "page": src.get("page", "Unknown"),
                    "snippet_preview": src.get("snippet", "")[:100] + "..."
                }
                for src in result.get("sources", [])[:2]
            ],
            "llm_used": result.get("llm_used", False)
        }
        
        results.append(test_result)
        
        # Print test result
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  Result: {status}")
        print(f"  Donor match: {'✅' if donor_match else '❌'} (Expected: {expected_donor})")
        print(f"  Keywords: {len(hits)}/{len(test['expected_keywords'])} matched ({keyword_match_ratio:.0%})")
        if result.get('summary'):
            summary_preview = result['summary'][:80] + "..." if len(result['summary']) > 80 else result['summary']
            print(f"  Summary: {summary_preview}")
        print()

    # Calculate comprehensive statistics
    passed_count = sum(1 for r in results if r["passed"])
    accuracy = passed_count / len(results)
    
    # Calculate accuracy by category
    categories = {}
    for result in results:
        category = result["category"]
        if category not in categories:
            categories[category] = {"total": 0, "passed": 0}
        categories[category]["total"] += 1
        if result["passed"]:
            categories[category]["passed"] += 1
    
    # Calculate accuracy by difficulty
    difficulties = {}
    for result in results:
        difficulty = result["difficulty"]
        if difficulty not in difficulties:
            difficulties[difficulty] = {"total": 0, "passed": 0}
        difficulties[difficulty]["total"] += 1
        if result["passed"]:
            difficulties[difficulty]["passed"] += 1
    
    # Additional metrics
    avg_sources = sum(r["source_count"] for r in results) / len(results)
    keyword_accuracy = sum(1 for r in results if r["keyword_match"]) / len(results)
    donor_accuracy = sum(1 for r in results if r["donor_match"]) / len(results)
    llm_used_count = sum(1 for r in results if r.get("llm_used", False))
    
    # Create comprehensive report
    report = {
        "metadata": {
            "evaluation_date": datetime.now().isoformat(),
            "system_version": "NGO Proposal Assistant RAG v2.0",
            "documents_used": [f for f in os.listdir(kb_path) if f.endswith('.pdf')],
            "total_chunks_loaded": doc_count,
            "llm_used_in_tests": llm_used_count
        },
        "summary": {
            "total_tests": len(results),
            "passed": passed_count,
            "failed": len(results) - passed_count,
            "overall_accuracy": accuracy,
            "keyword_match_rate": keyword_accuracy,
            "donor_match_rate": donor_accuracy,
            "average_sources_per_query": avg_sources,
            "llm_usage_rate": llm_used_count / len(results) if results else 0
        },
        "performance_by_category": {
            category: {
                "total": data["total"],
                "passed": data["passed"],
                "accuracy": data["passed"] / data["total"] if data["total"] > 0 else 0
            }
            for category, data in categories.items()
        },
        "performance_by_difficulty": {
            difficulty: {
                "total": data["total"],
                "passed": data["passed"],
                "accuracy": data["passed"] / data["total"] if data["total"] > 0 else 0
            }
            for difficulty, data in difficulties.items()
        },
        "detailed_results": results,
    }

    # Save detailed report
    detailed_filename = "evaluation_report_detailed.json"
    with open(detailed_filename, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Save simplified report
    simple_report = {
        "evaluation_summary": {
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "total_tests": len(results),
            "passed": passed_count,
            "accuracy": f"{accuracy:.1%}",
            "categories_tested": list(categories.keys()),
        },
        "category_performance": [
            {
                "category": category,
                "accuracy": f"{data['passed']/data['total']:.1%}",
                "score": f"{data['passed']}/{data['total']}"
            }
            for category, data in categories.items()
        ]
    }
    
    simple_filename = "evaluation_report.json"
    with open(simple_filename, "w", encoding="utf-8") as f:
        json.dump(simple_report, f, indent=2, ensure_ascii=False)

    # Print comprehensive summary
    print("\n" + "=" * 60)
    print("📊 EVALUATION COMPLETE - SUMMARY")
    print("=" * 60)
    print(f"Overall Accuracy: {accuracy:.1%} ({passed_count}/{len(results)})")
    print(f"Keyword Match Rate: {keyword_accuracy:.1%}")
    print(f"Donor Match Rate: {donor_accuracy:.1%}")
    print(f"Average sources per query: {avg_sources:.1f}")
    print(f"LLM used in {llm_used_count}/{len(results)} tests ({llm_used_count/len(results):.0%})")
    
    print("\n📈 Performance by Category:")
    for category, data in categories.items():
        cat_accuracy = data["passed"] / data["total"]
        print(f"  {category}: {cat_accuracy:.1%} ({data['passed']}/{data['total']})")
    
    print("\n📈 Performance by Difficulty:")
    for difficulty, data in difficulties.items():
        diff_accuracy = data["passed"] / data["total"]
        print(f"  {difficulty}: {diff_accuracy:.1%} ({data['passed']}/{data['total']})")
    
    print(f"\n💾 Reports saved:")
    print(f"  - Detailed: {detailed_filename}")
    print(f"  - Simple: {simple_filename}")
    print("=" * 60)

    return report


def run_interactive_demo():
    """Run an interactive demo"""
    print("\n" + "=" * 60)
    print("🎮 INTERACTIVE DEMO")
    print("=" * 60)
    
    # Initialize RAG
    rag = NGOProposalRAG(use_llm=True, llm_model="llama3.2")
    kb_path = Config.resolve_kb_path(None)
    
    print("Loading documents...")
    rag.load_pdfs_from_folder(kb_path)
    rag.build_vector_store()
    rag.create_retriever()
    
    print("✅ System ready!")
    print("\nType your questions about NGO/donor requirements.")
    print("Type 'exit', 'quit', or press Ctrl+C to end.\n")
    
    try:
        while True:
            question = input("\n❓ Your question: ").strip()
            
            if not question:
                continue
                
            if question.lower() in ['exit', 'quit', 'bye', 'q']:
                print("\n👋 Demo ended. Goodbye!")
                break
            
            print("\n" + "─" * 50)
            print(f"📋 Question: {question}")
            print("⏳ Searching...")
            
            result = rag.answer(question)
            
            print(f"\n📝 Answer ({'LLM-enhanced' if result.get('llm_used') else 'Retrieval-only'}):")
            print(f"   {result['summary']}")
            
            if result.get("sources"):
                print(f"\n📚 Top sources ({len(result['sources'])} found):")
                for i, src in enumerate(result["sources"][:3], 1):
                    print(f"   {i}. {src.get('donor', 'Unknown')} - {src.get('source', 'Unknown')} (Page {src.get('page', 'N/A')})")
                    if src.get('snippet'):
                        print(f"      Excerpt: {src['snippet'][:100]}...")
            
            print("─" * 50)
            
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted. Goodbye!")


if __name__ == "__main__":
    # Run evaluation
    print("NGO Proposal Assistant - Evaluation")
    print("1. Run comprehensive evaluation")
    print("2. Run interactive demo")
    print("3. Exit")
    
    choice = input("\nChoose option (1-3): ").strip()
    
    if choice == "1":
        report = evaluate_pdf_rag()
        
        # Ask if user wants to run demo
        demo = input("\nRun interactive demo? (y/n): ").strip().lower()
        if demo == 'y':
            run_interactive_demo()
    
    elif choice == "2":
        run_interactive_demo()
    
    else:
        print("Exiting.")