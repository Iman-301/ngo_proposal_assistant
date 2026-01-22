import json
import os
import sys
from datetime import datetime

# Ensure repo root is on path when running as a script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.pdf_rag import PDFRAG
from app.config import Config


def evaluate_pdf_rag():
    """Evaluate the RAG system with comprehensive test cases"""
    # Initialize RAG
    rag = PDFRAG()
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
    
    # Comprehensive test cases
    test_cases = [
        # ======================
        # EXISTING TESTS (from your current evaluation)
        # ======================
        {
            "question": "What is the indirect cost rate for non-US NGOs?",
            "expected_donor": "USAID",
            "expected_keywords": ["indirect", "cost", "rate", "%"],
            "category": "Financial Requirements",
            "difficulty": "Easy"
        },
        {
            "question": "Who is eligible for GPSA grants?",
            "expected_donor": "World Bank", 
            "expected_keywords": ["eligible", "organizations", "civil", "society"],
            "category": "Eligibility",
            "difficulty": "Easy"
        },
        {
            "question": "What procurement rules does USAID require?",
            "expected_donor": "USAID",
            "expected_keywords": ["procurement", "competitive", "quotes"],
            "category": "Procurement",
            "difficulty": "Medium"
        },
        {
            "question": "What reports are required by World Bank GPSA?",
            "expected_donor": "World Bank",
            "expected_keywords": ["reports", "progress", "financial"],
            "category": "Reporting",
            "difficulty": "Easy"
        },
        
        # ======================
        # NEW TESTS - Added for comprehensive evaluation
        # ======================
        {
            "question": "What is the maximum grant amount for GPSA projects?",
            "expected_donor": "World Bank",
            "expected_keywords": ["maximum", "grant", "amount", "$", "funding"],
            "category": "Financial Requirements",
            "difficulty": "Medium"
        },
        {
            "question": "What are the financial reporting requirements?",
            "expected_donor": "USAID",  # Both have this, but USAID is more detailed
            "expected_keywords": ["financial", "report", "quarterly", "annual"],
            "category": "Reporting",
            "difficulty": "Medium"
        },
        {
            "question": "What are the environmental safeguards required?",
            "expected_donor": "USAID",
            "expected_keywords": ["environmental", "safeguard", "assessment"],
            "category": "Compliance",
            "difficulty": "Hard"
        },
        {
            "question": "How are sub-grants and sub-contracts handled?",
            "expected_donors": ["USAID", "World Bank"],  # Both should mention this
            "expected_keywords": ["sub-grant", "subrecipient", "contract", "subcontract"],
            "category": "Administrative",
            "difficulty": "Medium"
        },
        {
            "question": "What are the deadlines for grant applications?",
            "expected_donor": "World Bank",
            "expected_keywords": ["deadline", "application", "submit", "date", "cycle"],
            "category": "Application Process",
            "difficulty": "Easy"
        },
        {
            "question": "What audit requirements exist for grants?",
            "expected_donors": ["USAID", "World Bank"],
            "expected_keywords": ["audit", "requirement", "financial", "review"],
            "category": "Compliance",
            "difficulty": "Medium"
        },
        {
            "question": "What are the rules for equipment purchases?",
            "expected_donor": "USAID",
            "expected_keywords": ["equipment", "purchase", "depreciation", "cost"],
            "category": "Procurement",
            "difficulty": "Hard"
        },
        {
            "question": "Which donors require co-financing or cost sharing?",
            "expected_donors": ["USAID", "World Bank"],
            "expected_keywords": ["cost", "sharing", "co-financing", "match"],
            "category": "Financial Requirements",
            "difficulty": "Medium"
        },
        {
            "question": "What are the requirements for travel expenses?",
            "expected_donor": "USAID",
            "expected_keywords": ["travel", "expenses", "per diem", "transportation"],
            "category": "Administrative",
            "difficulty": "Medium"
        },
        {
            "question": "How is intellectual property handled in grants?",
            "expected_donor": "USAID",
            "expected_keywords": ["intellectual", "property", "patent", "rights"],
            "category": "Legal",
            "difficulty": "Hard"
        },
    ]

    print(f"Running {len(test_cases)} test cases...\n")
    
    results = []
    for i, test in enumerate(test_cases, 1):
        print(f"Test {i}/{len(test_cases)}: {test['question']}")
        print(f"  Category: {test['category']} | Difficulty: {test['difficulty']}")
        
        # Get answer from RAG system
        result = rag.answer(test["question"])
        
        # Extract expected donors (handle both single donor and list)
        expected_donors = test.get("expected_donors", [test.get("expected_donor")])
        if isinstance(expected_donors, str):
            expected_donors = [expected_donors]
        
        # Check donor match
        found_donors = [src.get("donor", "") for src in result.get("sources", [])]
        donor_match = False
        matching_donors = []
        
        for expected in expected_donors:
            for found in found_donors:
                if expected and expected.lower() in found.lower():
                    donor_match = True
                    matching_donors.append(expected)
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
        
        # More lenient scoring: pass if at least 40% of keywords match
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
            "expected_donors": expected_donors,
            "matching_donors": matching_donors,
            "found_donors": list(set(found_donors)),  # Unique donors found
            "summary": result.get("summary", ""),
            "source_count": len(result.get("sources", [])),
            "sources_preview": [
                {
                    "donor": src.get("donor", "Unknown"),
                    "page": src.get("page", "Unknown"),
                    "snippet_preview": src.get("snippet", "")[:100] + "..."
                }
                for src in result.get("sources", [])[:2]  # Preview first 2 sources
            ]
        }
        
        results.append(test_result)
        
        # Print test result
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  Result: {status}")
        print(f"  Donor match: {'✅' if donor_match else '❌'} (Expected: {', '.join(expected_donors)})")
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
    
    # Create comprehensive report
    report = {
        "metadata": {
            "evaluation_date": datetime.now().isoformat(),
            "system_version": "NGO Proposal Assistant RAG v1.0",
            "documents_used": [f for f in os.listdir(kb_path) if f.endswith('.pdf')],
            "total_chunks_loaded": doc_count,
        },
        "summary": {
            "total_tests": len(results),
            "passed": passed_count,
            "failed": len(results) - passed_count,
            "overall_accuracy": accuracy,
            "keyword_match_rate": keyword_accuracy,
            "donor_match_rate": donor_accuracy,
            "average_sources_per_query": avg_sources,
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
    
    # Save simplified report (for quick viewing)
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
    
    print("\n📈 Performance by Category:")
    for category, data in categories.items():
        accuracy = data["passed"] / data["total"]
        print(f"  {category}: {accuracy:.1%} ({data['passed']}/{data['total']})")
    
    print("\n📈 Performance by Difficulty:")
    for difficulty, data in difficulties.items():
        accuracy = data["passed"] / data["total"]
        print(f"  {difficulty}: {accuracy:.1%} ({data['passed']}/{data['total']})")
    
    print(f"\n💾 Reports saved:")
    print(f"  - Detailed: {detailed_filename}")
    print(f"  - Simple: {simple_filename}")
    print("=" * 60)

    return report


def run_quick_demo():
    """Run a quick interactive demo after evaluation"""
    print("\n" + "=" * 60)
    print("🎮 QUICK INTERACTIVE DEMO")
    print("=" * 60)
    print("Try asking questions about NGO/donor requirements!")
    print("Type 'exit' to quit\n")
    
    rag = PDFRAG()
    kb_path = Config.resolve_kb_path(None)
    rag.load_pdfs_from_folder(kb_path)
    rag.build_vector_store()
    rag.create_retriever()
    
    sample_questions = [
        "What's the indirect cost rate?",
        "Who can apply for GPSA grants?",
        "What reports are required?",
        "What are procurement rules?",
    ]
    
    print("Sample questions you can try:")
    for i, q in enumerate(sample_questions, 1):
        print(f"  {i}. {q}")
    print()
    
    try:
        while True:
            question = input("❓ Your question: ").strip()
            
            if not question:
                continue
                
            if question.lower() in ['exit', 'quit', 'bye', 'q']:
                print("\n👋 Demo ended. Goodbye!")
                break
            
            print("\n" + "-" * 40)
            print(f"Q: {question}")
            
            result = rag.answer(question)
            
            if result.get("summary"):
                print(f"\n📝 Summary: {result['summary']}")
            
            if result.get("sources"):
                print(f"\n📚 Top sources found ({len(result['sources'])} total):")
                for i, src in enumerate(result["sources"][:3], 1):  # Show top 3
                    print(f"  {i}. {src.get('donor', 'Unknown')} - Page {src.get('page', 'N/A')}")
                    if src.get('snippet'):
                        print(f"     Excerpt: {src['snippet'][:120]}...")
            
            print("-" * 40 + "\n")
            
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted. Goodbye!")


if __name__ == "__main__":
    # Run comprehensive evaluation
    report = evaluate_pdf_rag()
    
    # Optional: Run interactive demo
    demo = input("\nRun interactive demo? (y/n): ").strip().lower()
    if demo == 'y':
        run_quick_demo()
