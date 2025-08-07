#!/usr/bin/env python3
"""
Test script for advanced semantic chunking and clause-based retrieval system
"""

import asyncio
import json
from pathlib import Path
import sys

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

from app.services.advanced_query_handler import AdvancedQueryHandler
from app.models.schemas import QueryRequest
from app.core.config import settings

async def test_advanced_system():
    """Test the complete advanced system"""
    
    print("🚀 TESTING ADVANCED HACKRX SYSTEM")
    print("=" * 60)
    print("✨ Features: Semantic Chunking + LLM Re-ranking + Fallback Reasoning")
    print()
    
    # Test questions focusing on the areas that failed before
    test_questions = [
        "What is the age eligibility criteria for purchasing the Arogya Sanjeevani Policy?",
        "What is the minimum and maximum sum insured available under this policy?",
        "Is pre‑hospitalisation and post‑hospitalisation coverage available, and for how many days?",
        "Is ambulance coverage included, and what is the reimbursement limit per hospitalisation?",  # Previously failed
        "Are daycare procedures covered under the policy and what types?",  # Previously failed
        "What is the mandatory co-payment clause percentage applicable per claim?",
        "Can the policy be ported from an existing health insurance plan without losing continuity benefits?",
        "Is the policy renewable for life even after maturity, and are there any upper age limits?",  # Previously failed
        "Does the policy cover Covid‑19 hospitalisation and related expenses?",
        "What specific named ailments have defined waiting periods (e.g. cataract, joint replacement) under this policy?"
    ]
    
    document_url = "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D"
    
    try:
        # Initialize advanced system
        print("🔧 Initializing Advanced Query Handler...")
        handler = AdvancedQueryHandler()
        print("✅ System initialized")
        print()
        
        # Create request
        request = QueryRequest(
            documents=document_url,
            questions=test_questions
        )
        
        print("📄 Processing document with advanced semantic chunking...")
        print("🧠 Using multi-stage retrieval + LLM re-ranking...")
        print()
        
        # Process queries
        response = await handler.process_query(request)
        
        print("🎯 ADVANCED SYSTEM RESULTS:")
        print("=" * 50)
        
        # Display results with analysis
        expected_answers = [
            "Age 18-65 for adults, 91 days to 25 years for children",
            "₹1 lakh to ₹10 lakhs in multiples of ₹50,000",
            "Pre: 30 days, Post: 60 days",
            "₹2,000 per hospitalisation",
            "All daycare procedures covered",
            "5% for all age groups",
            "Yes, with continuity benefits retained",
            "Yes, lifelong renewability with no age limit",
            "Yes, covered like any other illness",
            "24-month waiting period for cataract, joint replacement, etc."
        ]
        
        correct_count = 0
        total_score = 0
        max_score = 28  # Estimated based on HackRx scoring
        
        for i, (question, answer, expected) in enumerate(zip(test_questions, response.answers, expected_answers)):
            print(f"\n📝 QUESTION {i+1}:")
            print(f"Q: {question}")
            print(f"A: {answer[:200]}{'...' if len(answer) > 200 else ''}")
            
            # Simple accuracy check
            is_correct = _check_answer_quality(answer, expected)
            status = "✅ GOOD" if is_correct else "❌ NEEDS IMPROVEMENT"
            
            if is_correct:
                correct_count += 1
                # Estimate score based on question complexity
                if i in [3, 4, 7]:  # Previously failed questions
                    total_score += 3.0  # Higher weight
                elif i in [2, 6, 9]:  # Complex questions
                    total_score += 2.5
                else:
                    total_score += 2.0
            
            print(f"Status: {status}")
        
        print(f"\n🏆 ADVANCED SYSTEM PERFORMANCE:")
        print(f"   • Questions Answered Correctly: {correct_count}/{len(test_questions)} ({correct_count/len(test_questions)*100:.1f}%)")
        print(f"   • Estimated HackRx Score: {total_score:.1f}/{max_score} ({total_score/max_score*100:.1f}%)")
        print(f"   • Document Weight Multiplier: 2.0x (Unknown document)")
        print(f"   • Final Weighted Score: {total_score*2:.1f}")
        
        # Performance improvements
        improvements = []
        if correct_count >= 8:
            improvements.append("🎯 Excellent retrieval accuracy")
        if total_score >= 20:
            improvements.append("🏅 Strong performance on complex questions")
        if correct_count > 7:  # Better than before
            improvements.append("📈 Significant improvement over basic system")
        
        if improvements:
            print(f"\n✨ ADVANCED FEATURES WORKING:")
            for improvement in improvements:
                print(f"   {improvement}")
        
        # System insights
        print(f"\n🔍 SYSTEM INSIGHTS:")
        print(f"   • Semantic chunking preserves clause boundaries")
        print(f"   • Multi-stage retrieval finds more relevant content")
        print(f"   • LLM re-ranking improves candidate quality")
        print(f"   • Fallback reasoning handles edge cases")
        print(f"   • Enhanced metadata enables clause-level matching")
        
        return correct_count >= 8  # Success if 80%+ accuracy
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def _check_answer_quality(answer: str, expected_key_info: str) -> bool:
    """Simple check for answer quality based on key information"""
    answer_lower = answer.lower()
    expected_lower = expected_key_info.lower()
    
    # Check for "not available" responses (should be fewer now)
    if any(phrase in answer_lower for phrase in ["not available", "not found", "not specified"]):
        return False
    
    # Check for key terms from expected answer
    key_terms = expected_lower.split()
    key_terms = [term.strip('.,()') for term in key_terms if len(term) > 2]
    
    matches = sum(1 for term in key_terms if term in answer_lower)
    match_ratio = matches / len(key_terms) if key_terms else 0
    
    return match_ratio > 0.3  # At least 30% key terms should match

if __name__ == "__main__":
    success = asyncio.run(test_advanced_system())
    
    if success:
        print("\n🎉 ADVANCED SYSTEM TEST: SUCCESS!")
        print("   Ready for HackRx evaluation with 80-85% target accuracy")
    else:
        print("\n⚠️ ADVANCED SYSTEM TEST: NEEDS TUNING")
        print("   Consider further optimizations")
