import pytest
import asyncio
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# Test data
TEST_TOKEN = "cd1d783d335b1b00dbe6e50b828060c5475a425013da0d6d7cbf1092b61d32a0"
TEST_HEADERS = {
    "Authorization": f"Bearer {TEST_TOKEN}",
    "Content-Type": "application/json"
}

def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert "LLM-Powered Query Retrieval System" in response.json()["message"]

def test_unauthorized_access():
    """Test API without proper authentication"""
    test_payload = {
        "documents": "https://example.com/test.pdf",
        "questions": ["What is this document about?"]
    }
    
    response = client.post("/api/v1/hackrx/run", json=test_payload)
    assert response.status_code == 401

def test_invalid_token():
    """Test API with invalid token"""
    test_payload = {
        "documents": "https://example.com/test.pdf", 
        "questions": ["What is this document about?"]
    }
    
    headers = {
        "Authorization": "Bearer invalid_token",
        "Content-Type": "application/json"
    }
    
    response = client.post("/api/v1/hackrx/run", json=test_payload, headers=headers)
    assert response.status_code == 401

def test_document_status_endpoint():
    """Test document status endpoint"""
    response = client.get("/api/v1/documents/status", headers=TEST_HEADERS)
    assert response.status_code == 200
    assert "status" in response.json()

# Note: Full integration tests would require valid OpenAI API keys
# and actual document URLs. These are basic API structure tests.
