import logging
from typing import List, Dict, Any
import asyncio

from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.models.schemas import SearchResult, LLMResponse

logger = logging.getLogger(__name__)


class QueryHandler:
    """Main orchestrator for handling document queries"""
    
    def __init__(
        self,
        document_processor: DocumentProcessor,
        embedding_service: EmbeddingService,
        llm_service: LLMService
    ):
        self.document_processor = document_processor
        self.embedding_service = embedding_service
        self.llm_service = llm_service
        
        # Cache for processed documents to avoid reprocessing
        self.document_cache = {}
    
    async def process_query(self, document_url: str, questions: List[str]) -> List[str]:
        """
        Main method to process a document and answer questions
        
        Workflow:
        1. Download and process document
        2. Create embeddings and store in vector database
        3. For each question:
           a. Perform semantic search
           b. Generate LLM response with context
           c. Return structured answer
        """
        try:
            logger.info(f"Starting query processing for {len(questions)} questions")
            
            # Convert HttpUrl to string if needed
            if hasattr(document_url, '__str__'):
                document_url = str(document_url)
            
            # Step 1: Process document if not in cache
            document_hash = self._hash_url(document_url)
            
            if document_hash not in self.document_cache:
                logger.info("Document not in cache, processing...")
                
                # Download document
                file_path = await self.document_processor.download_document(document_url)
                
                # Process document into chunks
                chunks = await self.document_processor.process_document(file_path)
                
                # Create embeddings
                chunks_with_embeddings = await self.embedding_service.create_embeddings(chunks)
                
                # Add to vector database
                await self.embedding_service.add_to_index(chunks_with_embeddings)
                
                # Cache the fact that document is processed
                self.document_cache[document_hash] = {
                    'url': document_url,
                    'chunks_count': len(chunks),
                    'processed': True
                }
                
                logger.info(f"Document processed and cached: {len(chunks)} chunks created")
            else:
                logger.info("Document found in cache, using existing data")
            
            # Step 2: Answer questions
            answers = []
            
            for i, question in enumerate(questions, 1):
                logger.info(f"Processing question {i}/{len(questions)}: {question[:50]}...")
                
                try:
                    # Perform semantic search
                    search_results = await self.embedding_service.search(question)
                    
                    if not search_results:
                        logger.warning(f"No relevant chunks found for question: {question}")
                        answers.append("The information is not available in the provided document.")
                        continue
                    
                    # Generate answer using LLM
                    llm_response = await self.llm_service.answer_question(
                        question=question,
                        search_results=search_results
                    )
                    
                    answers.append(llm_response.answer)
                    
                    logger.info(f"Question {i} processed successfully (confidence: {llm_response.confidence:.2f})")
                    
                except Exception as e:
                    logger.error(f"Error processing question {i}: {str(e)}")
                    answers.append(f"Error processing question: {str(e)}")
            
            logger.info(f"Query processing completed. Generated {len(answers)} answers.")
            return answers
            
        except Exception as e:
            logger.error(f"Error in query processing: {str(e)}")
            raise Exception(f"Query processing failed: {str(e)}")
    
    async def process_single_question(
        self, 
        question: str, 
        document_url: str = None,
        use_existing_index: bool = True
    ) -> Dict[str, Any]:
        """
        Process a single question with detailed response including metadata
        """
        try:
            # If document URL provided and not using existing index, process it
            if document_url and not use_existing_index:
                await self.process_query(document_url, [])  # Process without questions
            
            # Search for relevant chunks
            search_results = await self.embedding_service.search(question)
            
            if not search_results:
                return {
                    "answer": "The information is not available in the provided document.",
                    "confidence": 0.1,
                    "sources": [],
                    "reasoning": "No relevant document sections found for this question",
                    "search_results": []
                }
            
            # Generate detailed response
            llm_response = await self.llm_service.answer_question(
                question=question,
                search_results=search_results
            )
            
            # Format search results for response
            formatted_results = []
            for result in search_results[:3]:  # Top 3 results
                formatted_results.append({
                    "content_preview": result.chunk.content[:200] + "...",
                    "similarity_score": result.similarity_score,
                    "chunk_id": result.chunk.chunk_id,
                    "rank": result.rank
                })
            
            return {
                "answer": llm_response.answer,
                "confidence": llm_response.confidence,
                "sources": llm_response.sources,
                "reasoning": llm_response.reasoning,
                "tokens_used": llm_response.tokens_used,
                "search_results": formatted_results
            }
            
        except Exception as e:
            logger.error(f"Error processing single question: {str(e)}")
            return {
                "answer": f"Error processing question: {str(e)}",
                "confidence": 0.0,
                "sources": [],
                "reasoning": f"Processing error: {str(e)}",
                "search_results": []
            }
    
    def get_cache_status(self) -> Dict[str, Any]:
        """Get current cache status"""
        return {
            "cached_documents": len(self.document_cache),
            "vector_database_size": self.embedding_service.get_vector_count(),
            "processed_documents": self.embedding_service.get_document_count()
        }
    
    def clear_cache(self):
        """Clear document cache and vector database"""
        logger.info("Clearing cache and vector database...")
        self.document_cache.clear()
        self.embedding_service.clear_index()
        logger.info("Cache and vector database cleared")
    
    def _hash_url(self, url: str) -> str:
        """Generate hash for URL to use as cache key"""
        import hashlib
        return hashlib.md5(url.encode()).hexdigest()
