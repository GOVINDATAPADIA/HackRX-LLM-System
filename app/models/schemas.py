from pydantic import BaseModel, HttpUrl, Field
from typing import List, Dict, Any, Optional


class QueryRequest(BaseModel):
    """Request model for document query processing"""
    documents: HttpUrl = Field(
        ..., 
        description="URL to the document (PDF, DOCX, etc.) to be processed"
    )
    questions: List[str] = Field(
        ..., 
        description="List of questions to be answered based on the document",
        min_items=1
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
                "questions": [
                    "What is the grace period for premium payment?",
                    "What is the waiting period for pre-existing diseases?"
                ]
            }
        }


class QueryResponse(BaseModel):
    """Response model for document query results"""
    answers: List[str] = Field(
        ..., 
        description="List of answers corresponding to the input questions"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "answers": [
                    "A grace period of thirty days is provided for premium payment after the due date.",
                    "There is a waiting period of thirty-six (36) months for pre-existing diseases."
                ]
            }
        }


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Service status")
    message: str = Field(..., description="Status message")
    version: str = Field(..., description="API version")


class DocumentChunk(BaseModel):
    """Model for document chunks with metadata"""
    content: str = Field(..., description="Text content of the chunk")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding of the chunk")
    chunk_id: str = Field(..., description="Unique identifier for the chunk")


class SearchResult(BaseModel):
    """Model for search results from vector database"""
    chunk: DocumentChunk = Field(..., description="The matching document chunk")
    similarity_score: float = Field(..., description="Similarity score (0-1)")
    rank: int = Field(..., description="Rank in search results")


class LLMResponse(BaseModel):
    """Model for LLM response with metadata"""
    answer: str = Field(..., description="Generated answer")
    confidence: float = Field(..., description="Confidence score (0-1)")
    sources: List[str] = Field(default_factory=list, description="Source chunks used")
    reasoning: str = Field(..., description="Explanation of the decision process")
    tokens_used: int = Field(..., description="Number of tokens consumed")
