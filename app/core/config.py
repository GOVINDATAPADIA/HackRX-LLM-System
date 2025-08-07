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
    CHUNK_SIZE: int = 800  # Reduced to handle API limits (was 1200)
    CHUNK_OVERLAP: int = 100  # Reduced overlap for smaller chunks (was 200)
    
    # Vector Database Configuration
    VECTOR_DIMENSION: int = 1536  # Match existing Pinecone index dimension
    TOP_K_RESULTS: int = 12  # Increased for better retrieval (was 8)
    SIMILARITY_THRESHOLD: float = 0.5  # Lowered for better recall (was 0.7)
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/app.log"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Global settings instance
settings = Settings()
