from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import List, Dict, Any, Optional
import os
from dotenv import load_dotenv
import logging
from contextlib import asynccontextmanager

from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.services.advanced_query_handler import AdvancedQueryHandler
from app.models.schemas import QueryRequest, QueryResponse, HealthResponse
from app.core.config import settings
from app.core.logging_config import setup_logging

load_dotenv()

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Global services
document_processor = None
embedding_service = None
llm_service = None
query_handler = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    global document_processor, embedding_service, llm_service, query_handler
    
    logger.info("Initializing services...")
    
    document_processor = DocumentProcessor()
    embedding_service = EmbeddingService()
    llm_service = LLMService()
    query_handler = AdvancedQueryHandler()
    
    logger.info("Services initialized successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down services...")

app = FastAPI(
    title="LLM-Powered Intelligent Query-Retrieval System",
    description="HackRx Solution: Process large documents and make contextual decisions for insurance, legal, HR, and compliance domains.",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify the API token"""
    token = credentials.credentials
    if token != settings.API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return token

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        message="LLM-Powered Query Retrieval System is running",
        version="1.0.0"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    return HealthResponse(
        status="healthy",
        message="All services are operational",
        version="1.0.0"
    )

@app.post(f"/api/{settings.API_VERSION}/hackrx/run", response_model=QueryResponse)
async def run_hackrx_query(
    request: QueryRequest,
    token: str = Depends(verify_token)
):
    """
    Main endpoint for processing documents and answering queries
    
    This endpoint automatically:
    1. Downloads and processes the document from the provided URL (if not already processed)
    2. Creates embeddings using Google Gemini and stores them in Pinecone vector database
    3. For each question, performs semantic search to find relevant document sections
    4. Uses LLM to generate contextual answers based on retrieved content
    5. Returns structured JSON response with explainable decisions
    
    No manual preprocessing required - just send URL + questions and get answers!
    """
    try:
        logger.info(f"🚀 NEW QUERY REQUEST RECEIVED")
        logger.info(f"📄 Document URL: {request.documents}")
        logger.info(f"❓ Number of questions: {len(request.questions)}")
        logger.info("=" * 60)
        
        # Show current document status
        doc_info = query_handler.get_current_document_info()
        if doc_info["has_document_loaded"]:
            logger.info(f"📋 Current document: {doc_info['current_document_hash']}")
        else:
            logger.info("📋 No document currently loaded")
        
        # Process the document and get answers (automatic embedding generation with single-doc mode)
        response = await query_handler.process_query(request)
        
        logger.info(f"✅ Successfully processed {len(request.questions)} questions")
        logger.info("🎉 Query processing complete!")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Error processing query: {str(e)}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing query: {str(e)}"
        )

@app.get(f"/api/{settings.API_VERSION}/documents/status")
async def document_status(token: str = Depends(verify_token)):
    """Get status of document processing and current loaded document"""
    try:
        # Get current document info
        doc_info = query_handler.get_current_document_info()
        
        # Get Pinecone stats
        vector_count = embedding_service.get_vector_count() if embedding_service else 0
        
        return {
            "status": "operational",
            "mode": "single_document_mode",
            "current_document": {
                "url": doc_info["current_document_url"],
                "hash": doc_info["current_document_hash"],
                "is_loaded": doc_info["has_document_loaded"]
            },
            "pinecone_stats": {
                "total_vectors": vector_count,
                "index_name": settings.PINECONE_INDEX_NAME
            },
            "cache_info": {
                "cached_documents": doc_info["cache_size"]
            }
        }
    except Exception as e:
        logger.error(f"Error getting document status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        log_level=settings.LOG_LEVEL.lower()
    )
