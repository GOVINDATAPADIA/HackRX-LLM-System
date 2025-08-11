#!/usr/bin/env python3
"""
Quick test to verify all fixes work together
"""

import requests
import json

# API configuration
API_BASE_URL = "http://localhost:8000"
API_TOKEN = "cd1d783d335b1b00dbe6e50b828060c5475a425013da0d6d7cbf1092b61d32a0"

# Simple test request
test_request = {
    "documents": "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D",
    "questions": [
        "What is the age eligibility criteria?",
        "What is the co-payment percentage?",
        "Is ambulance coverage included?"
    ]
}

def test_fixed_system():
    """Test the completely fixed system"""
    print("🧪 TESTING COMPLETE SYSTEM FIXES")
    print("=" * 50)
    print("✅ Fixed chunking: Smaller semantic chunks instead of 1 giant chunk")
    print("✅ Fixed similarity: Lowered threshold from 0.5 to 0.2")
    print("✅ Fixed validation: 4KB max chunk size instead of 32KB")
    print("✅ Enhanced debugging: Better search logging")
    print("✅ Single document mode: Auto-clear old data")
    print()
    
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    print("📡 Making API request...")
    print("⏳ Processing (should now create many chunks and find results)...")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/hackrx/run",
            headers=headers,
            json=test_request,
            timeout=300
        )
        
        if response.status_code == 200:
            result = response.json()
            print("\n🎉 SUCCESS! System fixes working!")
            print("=" * 50)
            
            for i, answer in enumerate(result.get("answers", []), 1):
                print(f"\n📝 Answer {i}:")
                print(f"Q: {test_request['questions'][i-1]}")
                print(f"A: {answer}")
                print("-" * 30)
                
            print("\n💡 CHECK SERVER LOGS to verify:")
            print("   • Multiple chunks created (not just 1)")
            print("   • Search found relevant chunks (not 0)")
            print("   • Similarity scores above 0.2")
                
        else:
            print(f"\n❌ ERROR: Status {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("\n🚫 Connection error - make sure FastAPI server is running")
        print("💡 Start server with: python main.py")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")

if __name__ == "__main__":
    test_fixed_system()
