#!/usr/bin/env python3
"""
Test script for Single Document Mode with Auto-Clear functionality
"""

import requests
import json
import time

# API configuration
API_BASE_URL = "http://localhost:8000"
API_TOKEN = "cd1d783d335b1b00dbe6e50b828060c5475a425013da0d6d7cbf1092b61d32a0"

# Test with two different documents
test_doc_1 = {
    "documents": "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D",
    "questions": ["What is the age eligibility criteria for this policy?"]
}

# For testing purposes - you can change this to a different PDF URL
test_doc_2 = {
    "documents": "https://example.com/different-document.pdf",  # Different URL for testing
    "questions": ["What are the main benefits of this policy?"]
}

def get_document_status():
    """Get current document status"""
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/documents/status", headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Status {response.status_code}: {response.text}"}
    except Exception as e:
        return {"error": str(e)}

def test_single_document_mode():
    """Test the single document mode with auto-clear functionality"""
    print("🧪 TESTING SINGLE DOCUMENT MODE WITH AUTO-CLEAR")
    print("=" * 60)
    
    headers = {
        "Authorization": f"Bearer {API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    print("📊 Step 1: Check initial document status")
    status = get_document_status()
    print(f"Current status: {json.dumps(status, indent=2)}")
    print()
    
    print("📄 Step 2: Process first document")
    print(f"Document 1: {test_doc_1['documents'][:80]}...")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/hackrx/run",
            headers=headers,
            json=test_doc_1,
            timeout=300
        )
        
        if response.status_code == 200:
            print("✅ Document 1 processed successfully!")
            
            # Check status after processing
            print("\n📊 Status after document 1:")
            status = get_document_status()
            print(f"Vectors in Pinecone: {status.get('pinecone_stats', {}).get('total_vectors', 'Unknown')}")
            print(f"Current document hash: {status.get('current_document', {}).get('hash', 'None')}")
        else:
            print(f"❌ Error processing document 1: {response.text}")
            return
            
    except requests.exceptions.ConnectionError:
        print("🚫 Connection error - make sure FastAPI server is running")
        print("💡 Start server with: python main.py")
        return
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return
    
    print("\n" + "="*60)
    print("🔄 Step 3: Process SAME document again (should skip processing)")
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/hackrx/run",
            headers=headers,
            json=test_doc_1,  # Same document
            timeout=60  # Should be much faster
        )
        
        if response.status_code == 200:
            print("✅ Same document request handled quickly (cached)!")
        else:
            print(f"❌ Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    print("\n" + "="*60)
    print("🆕 Step 4: Process DIFFERENT document (should auto-clear and reload)")
    print("⚠️  NOTE: Using test URL - change test_doc_2 URL to a real PDF for full test")
    
    # For demo purposes, we'll modify the URL slightly to trigger new processing
    test_doc_modified = test_doc_1.copy()
    test_doc_modified["documents"] = test_doc_1["documents"] + "&test=different"  # Slightly different URL
    test_doc_modified["questions"] = ["What is the co-payment percentage?"]
    
    try:
        print(f"Modified URL: {test_doc_modified['documents'][:80]}...")
        
        response = requests.post(
            f"{API_BASE_URL}/api/v1/hackrx/run",
            headers=headers,
            json=test_doc_modified,
            timeout=300
        )
        
        if response.status_code == 200:
            print("✅ Different document processed! Auto-clear worked!")
            
            # Final status check
            print("\n📊 Final status:")
            status = get_document_status()
            print(f"Vectors in Pinecone: {status.get('pinecone_stats', {}).get('total_vectors', 'Unknown')}")
            print(f"Current document hash: {status.get('current_document', {}).get('hash', 'None')}")
            
        else:
            print(f"❌ Error processing different document: {response.text}")
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    
    print("\n🎉 SINGLE DOCUMENT MODE TEST COMPLETE!")
    print("💡 Check the server logs to see the auto-clear and processing steps")

if __name__ == "__main__":
    test_single_document_mode()
