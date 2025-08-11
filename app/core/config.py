from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # API Configuration
    API_HOST: str = "localhost"
    API_PORT: int = 8000
    API_VERSION: str = "v1"
    API_TOKEN: str = "cd1d783d335b1b00dbe6e50b828060c5475a425013da0d6d7cbf1092b61d32a0"
    
    # Gemini Configuration
    GOOGLE_API_KEY: str = "AIzaSyCtLUANrZVOXueA2qAPmDe3BO7dkzTFMIY"
    GEMINI_MODEL: str = "gemini-1.5-pro"
    MAX_TOKENS: int = 120
    TEMPERATURE: float = 0.1
    
    # Pinecone Configuration
    PINECONE_API_KEY: str = "pcsk_37asXz_2UaWucJqSrwNc2APoWFCVA4LXcoKhucsoAqTmLxV1y7XFYgk9GAJmcm4TZ5TRWe"
    PINECONE_ENVIRONMENT: str = "us-east-1"
    PINECONE_INDEX_NAME: str = "index-hackrx"
    
    # Document Processing
    MAX_FILE_SIZE_MB: int = 50
    SUPPORTED_FORMATS: str = "pdf,docx,txt,eml"
    CHUNK_SIZE: int = 1000  # Optimal size for semantic chunks
    CHUNK_OVERLAP: int = 150  # Good overlap to preserve context
    
    # Vector Database Configuration - Optimized for speed
    VECTOR_DIMENSION: int = 1536  # Match existing Pinecone index dimension
    TOP_K_RESULTS: int = 8  # Reduced for speed (was 12)
    SIMILARITY_THRESHOLD: float = 0.2  # Keep low threshold for recall
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/app.log"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Global settings instance
settings = Settings()
