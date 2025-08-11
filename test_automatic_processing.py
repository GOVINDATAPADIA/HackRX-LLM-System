#!/usr/bin/env python3
"""
Test script to verify automatic embedding generation works
"""

import requests
import json

# API configuration
API_BASE_URL = "http://localhost:8000"
API_TOKEN = "cd1d783d335b1b00dbe6e50b828060c5475a425013da0d6d7cbf1092b61d32a0"

# Test data - same as your test_request.json
test_request = {
    "documents": "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D",
    "questions": [
        "What is the age eligibility criteria for purchasing the Arogya Sanjeevani Policy?",
        "What is the minimum and maximum sum insured available under this policy?",
        "Is ambulance coverage included, and what is the reimbursement limit per hospitalisation?"
    ]
}

def test_automatic_processing():
    """Test the automatic embedding generation"""
    print("🧪 TESTING AUTOMATIC EMBEDDING GENERATION")
    print("=" * 50)
    
    # Headers
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    print(f"📡 Making request to: {API_BASE_URL}/api/v1/hackrx/run")
    print(f"📄 Document: {test_request['documents'][:80]}...")
    print(f"❓ Questions: {len(test_request['questions'])}")
    print("\n⏳ Processing... (This may take a few minutes for first-time document processing)")
    
    try:
        # Make the request
        response = requests.post(
            f"{API_BASE_URL}/api/v1/hackrx/run",
            headers=headers,
            json=test_request,
            timeout=300  # 5 minutes timeout for processing
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ SUCCESS! Automatic processing worked!")
            print("=" * 50)
            
            # Display answers
            for i, answer in enumerate(result.get("answers", []), 1):
                print(f"\n📝 Answer {i}:")
                print(f"Q: {test_request['questions'][i-1]}")
                print(f"A: {answer}")
                print("-" * 30)
                
        else:
            print(f"\n❌ ERROR: Status {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.Timeout:
        print("\n⏰ Request timed out (document processing takes time)")
        print("💡 Try running again - document should be cached for faster processing")
        
    except requests.exceptions.ConnectionError:
        print("\n🚫 Connection error - make sure FastAPI server is running")
        print("💡 Start server with: python main.py")
        
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")

if __name__ == "__main__":
    test_automatic_processing()
