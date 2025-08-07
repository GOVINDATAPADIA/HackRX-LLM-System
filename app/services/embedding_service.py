import numpy as np
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai
import logging
from typing import List, Dict, Any, Optional
import time
import hashlib

from app.models.schemas import DocumentChunk, SearchResult
from app.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for creating embeddings and managing Pinecone vector database"""
    
    def __init__(self):
        # Configure Gemini API
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=settings.PINECONE_API_KEY)
        
        self.index_name = settings.PINECONE_INDEX_NAME
        self.dimension = settings.VECTOR_DIMENSION
        self.top_k = settings.TOP_K_RESULTS
        self.similarity_threshold = settings.SIMILARITY_THRESHOLD
        
        # Initialize or connect to index
        self.index = None
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize or connect to Pinecone index"""
        try:
            # Check if index exists
            existing_indexes = [index.name for index in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                logger.info(f"Creating new Pinecone index: {self.index_name}")
                
                # Create index with serverless configuration
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                
                # Wait for index to be ready
                time.sleep(10)
            else:
                logger.info(f"Connecting to existing Pinecone index: {self.index_name}")
                
                # Check if dimensions match
                index_info = self.pc.describe_index(self.index_name)
                existing_dimension = index_info.dimension
                
                if existing_dimension != self.dimension:
                    logger.warning(f"Dimension mismatch: Index has {existing_dimension}, expected {self.dimension}")
                    logger.info(f"Deleting existing index and creating new one...")
                    
                    # Delete existing index
                    self.pc.delete_index(self.index_name)
                    time.sleep(5)
                    
                    # Create new index with correct dimensions
                    self.pc.create_index(
                        name=self.index_name,
                        dimension=self.dimension,
                        metric="cosine",
                        spec=ServerlessSpec(
                            cloud="aws",
                            region="us-east-1"
                        )
                    )
                    time.sleep(10)
            
            # Connect to index
            self.index = self.pc.Index(self.index_name)
            
            # Get index stats
            stats = self.index.describe_index_stats()
            logger.info(f"Connected to Pinecone index with {stats['total_vector_count']} vectors")
            
        except Exception as e:
            logger.error(f"Error initializing Pinecone index: {str(e)}")
            raise Exception(f"Failed to initialize Pinecone: {str(e)}")
    
    async def create_embeddings(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Create embeddings for document chunks using Gemini"""
        try:
            logger.info(f"Creating embeddings for {len(chunks)} chunks using Gemini...")
            
            valid_chunks = []
            
            for i, chunk in enumerate(chunks):
                # Create embedding using Gemini API
                try:
                    # Skip empty or very short chunks
                    if not chunk.content.strip() or len(chunk.content.strip()) < 10:
                        logger.warning(f"Skipping chunk {i} - too short or empty")
                        continue
                    
                    result = genai.embed_content(
                        model="models/embedding-001",
                        content=chunk.content,
                        task_type="retrieval_document"
                    )
                    
                    # Get the embedding vector (768 dimensions from Google)
                    embedding = result['embedding']
                    
                    # Validate embedding is not all zeros
                    if not embedding or all(v == 0.0 for v in embedding):
                        logger.warning(f"Skipping chunk {i} - received zero embedding")
                        continue
                    
                    # Pad to 1536 dimensions to match Pinecone index
                    if len(embedding) < self.dimension:
                        # Pad with small random values instead of zeros to avoid all-zero vectors
                        padding_size = self.dimension - len(embedding)
                        import random
                        padding = [random.uniform(-0.001, 0.001) for _ in range(padding_size)]
                        embedding.extend(padding)
                    elif len(embedding) > self.dimension:
                        # Truncate if somehow larger
                        embedding = embedding[:self.dimension]
                    
                    # Final validation - ensure no all-zero vectors
                    if all(abs(v) < 1e-10 for v in embedding):
                        logger.warning(f"Skipping chunk {i} - embedding is essentially zero")
                        continue
                    
                    chunk.embedding = embedding
                    valid_chunks.append(chunk)
                    
                    # Small delay to respect rate limits
                    time.sleep(0.1)
                    
                except Exception as e:
                    logger.error(f"Error creating embedding for chunk {i}: {str(e)}")
                    # Skip this chunk instead of creating zero vector
                    continue
                
                if (len(valid_chunks)) % 10 == 0:
                    logger.info(f"Created embeddings for {len(valid_chunks)} valid chunks (processed {i + 1}/{len(chunks)})")
            
            logger.info(f"Successfully created {len(valid_chunks)} valid embeddings from {len(chunks)} chunks")
            return valid_chunks
            
        except Exception as e:
            logger.error(f"Error in create_embeddings: {str(e)}")
            raise
            
            logger.info("All embeddings created successfully")
            return chunks
            
        except Exception as e:
            logger.error(f"Error creating embeddings: {str(e)}")
            raise Exception(f"Failed to create embeddings: {str(e)}")
    
    async def add_to_index(self, chunks: List[DocumentChunk]):
        """Add chunks to Pinecone index"""
        try:
            logger.info(f"Adding {len(chunks)} chunks to Pinecone index...")
            
            # Filter out chunks with invalid embeddings
            valid_chunks = []
            for chunk in chunks:
                if chunk.embedding is None:
                    logger.warning(f"Skipping chunk {chunk.chunk_id} - no embedding")
                    continue
                
                # Validate embedding is not all zeros or near-zero
                if all(abs(v) < 1e-10 for v in chunk.embedding):
                    logger.warning(f"Skipping chunk {chunk.chunk_id} - embedding is zero or near-zero")
                    continue
                
                # Validate embedding has correct dimension
                if len(chunk.embedding) != self.dimension:
                    logger.warning(f"Skipping chunk {chunk.chunk_id} - wrong dimension: {len(chunk.embedding)} vs {self.dimension}")
                    continue
                
                valid_chunks.append(chunk)
            
            if not valid_chunks:
                logger.warning("No valid chunks to add to Pinecone index")
                return
            
            logger.info(f"Uploading {len(valid_chunks)} valid chunks (filtered from {len(chunks)} total)")
            
            # Prepare vectors for upsert
            vectors = []
            for chunk in valid_chunks:
                vector_data = {
                    "id": chunk.chunk_id,
                    "values": chunk.embedding,
                    "metadata": {
                        "content": chunk.content,
                        "source": chunk.metadata.get("source", "unknown"),
                        "chunk_index": chunk.metadata.get("chunk_index", 0),
                        "character_count": chunk.metadata.get("character_count", len(chunk.content)),
                        "section_title": chunk.metadata.get("section_title", "Unknown"),
                        "clause_type": chunk.metadata.get("clause_type", "general"),
                        "is_special_clause": chunk.metadata.get("is_special_clause", False)
                    }
                }
                vectors.append(vector_data)
            
            # Upsert in batches of 100
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                
                # Final validation of batch
                valid_batch = []
                for vector in batch:
                    if all(abs(v) < 1e-10 for v in vector["values"]):
                        logger.warning(f"Skipping vector {vector['id']} in batch - still has zero values")
                        continue
                    valid_batch.append(vector)
                
                if valid_batch:
                    self.index.upsert(vectors=valid_batch)
                    logger.info(f"Upserted batch of {len(valid_batch)} vectors")
                
                if (i + batch_size) % 500 == 0:
                    logger.info(f"Progress: {min(i + batch_size, len(vectors))}/{len(vectors)} vectors processed")
            
            logger.info(f"Successfully added {len(valid_chunks)} valid chunks to Pinecone index")
            
        except Exception as e:
            logger.error(f"Error adding to Pinecone index: {str(e)}")
            raise Exception(f"Failed to add to Pinecone index: {str(e)}")
    
    async def search(self, query: str, k: Optional[int] = None) -> List[SearchResult]:
        """Search for similar chunks using the query"""
        try:
            k = k or self.top_k
            
            # Create embedding for query using Gemini
            result = genai.embed_content(
                model="models/embedding-001",
                content=query,
                task_type="retrieval_query"
            )
            
            # Get the embedding vector (768 dimensions from Google)
            query_embedding = result['embedding']
            
            # Pad to 1536 dimensions to match Pinecone index
            if len(query_embedding) < self.dimension:
                # Pad with zeros to reach target dimension
                padding_size = self.dimension - len(query_embedding)
                query_embedding.extend([0.0] * padding_size)
            elif len(query_embedding) > self.dimension:
                # Truncate if somehow larger
                query_embedding = query_embedding[:self.dimension]
            
            # Search in Pinecone index
            search_results = self.index.query(
                vector=query_embedding,
                top_k=k,
                include_metadata=True
            )
            
            # Build results
            results = []
            for i, match in enumerate(search_results.matches):
                # Filter by similarity threshold
                if match.score < self.similarity_threshold:
                    continue
                
                chunk = DocumentChunk(
                    content=match.metadata["content"],
                    chunk_id=match.id,
                    metadata={
                        "source": match.metadata.get("source", "unknown"),
                        "chunk_index": match.metadata.get("chunk_index", 0),
                        "character_count": match.metadata.get("character_count", 0)
                    }
                )
                
                result = SearchResult(
                    chunk=chunk,
                    similarity_score=float(match.score),
                    rank=i + 1
                )
                results.append(result)
            
            logger.info(f"Search found {len(results)} relevant chunks for query: '{query[:50]}...'")
            return results
            
        except Exception as e:
            logger.error(f"Error searching Pinecone: {str(e)}")
            raise Exception(f"Failed to search: {str(e)}")
    
    def get_document_count(self) -> int:
        """Get number of processed documents (estimated from unique sources)"""
        try:
            # This is an approximation - query a sample and count unique sources
            sample_results = self.index.query(
                vector=[0.0] * self.dimension,
                top_k=1000,
                include_metadata=True
            )
            
            sources = set()
            for match in sample_results.matches:
                if "source" in match.metadata:
                    sources.add(match.metadata["source"])
            
            return len(sources)
        except:
            return 0
    
    def get_vector_count(self) -> int:
        """Get total number of vectors in Pinecone index"""
        try:
            stats = self.index.describe_index_stats()
            return stats['total_vector_count']
        except:
            return 0
    
    def clear_index(self):
        """Clear the entire Pinecone index"""
        logger.info("Clearing Pinecone index...")
        try:
            # Delete all vectors (this deletes the entire index content)
            self.index.delete(delete_all=True)
            logger.info("Pinecone index cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing Pinecone index: {str(e)}")
    
    def get_index_info(self) -> Dict[str, Any]:
        """Get detailed index information"""
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vectors": stats['total_vector_count'],
                "dimension": stats['dimension'],
                "index_fullness": stats.get('index_fullness', 0),
                "namespaces": stats.get('namespaces', {})
            }
        except Exception as e:
            logger.error(f"Error getting index info: {str(e)}")
            return {}
