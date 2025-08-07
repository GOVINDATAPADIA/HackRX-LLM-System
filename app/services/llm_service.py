import google.generativeai as genai
import logging
from typing import List, Dict, Any, Optional
import json

from app.models.schemas import SearchResult, LLMResponse
from app.core.config import settings

logger = logging.getLogger(__name__)


class LLMService:
    """Service for interacting with LLM (Gemini) for question answering"""
    
    def __init__(self):
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        self.model = genai.GenerativeModel(settings.GEMINI_MODEL)
        self.max_tokens = settings.MAX_TOKENS
        self.temperature = settings.TEMPERATURE
    
    async def answer_question(
        self, 
        question: str, 
        search_results: List[SearchResult], 
        context: str = ""
    ) -> LLMResponse:
        """Generate answer using LLM based on search results"""
        try:
            logger.info(f"Generating answer for question: '{question[:50]}...'")
            
            # Prepare context from search results
            context_chunks = []
            sources = []
            
            for result in search_results:
                context_chunks.append({
                    "content": result.chunk.content,
                    "similarity": result.similarity_score,
                    "source": result.chunk.chunk_id
                })
                sources.append(result.chunk.chunk_id)
            
            # Build system prompt
            system_prompt = self._build_system_prompt()
            
            # Build user prompt with context
            user_prompt = self._build_user_prompt(question, context_chunks, context)
            
            # Combine system and user prompts for Gemini
            full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            # Count tokens (rough estimation)
            prompt_tokens = self._count_tokens(full_prompt)
            
            if prompt_tokens > self.max_tokens * 0.7:  # Leave room for response
                logger.warning(f"Prompt too long ({prompt_tokens} tokens), truncating context...")
                context_chunks = context_chunks[:3]  # Keep top 3 results
                user_prompt = self._build_user_prompt(question, context_chunks, context)
                full_prompt = f"{system_prompt}\n\n{user_prompt}"
            
            # Call Gemini API
            generation_config = genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=min(1000, self.max_tokens - prompt_tokens) if prompt_tokens < self.max_tokens else 1000,
            )
            
            response = self.model.generate_content(
                full_prompt,
                generation_config=generation_config
            )
            
            answer = response.text if response.text else "No answer generated"
            
            # Gemini doesn't provide token usage, so we estimate
            tokens_used = prompt_tokens + self._count_tokens(answer)
            
            # Calculate confidence based on search results quality
            confidence = self._calculate_confidence(search_results, answer)
            
            # Generate reasoning
            reasoning = self._generate_reasoning(question, search_results, answer)
            
            llm_response = LLMResponse(
                answer=answer,
                confidence=confidence,
                sources=sources[:3],  # Top 3 sources
                reasoning=reasoning,
                tokens_used=tokens_used
            )
            
            logger.info(f"Generated answer with confidence: {confidence:.2f}")
            return llm_response
            
        except Exception as e:
            logger.error(f"Error generating LLM response: {str(e)}")
            raise Exception(f"Failed to generate answer: {str(e)}")
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for the LLM"""
        return """You are an expert document analysis assistant specializing in insurance, legal, HR, and compliance documents. Your task is to answer questions based on provided document excerpts with high accuracy and clear explanations.

Guidelines:
1. Answer questions directly and precisely based on the provided context
2. When specific information is not explicitly stated, provide the most reasonable interpretation based on standard insurance practices
3. Provide specific details like numbers, dates, conditions, and exceptions when available
4. For policy-related questions, include relevant conditions, limitations, or exceptions
5. Be concise but comprehensive
6. Use exact terminology from the document when possible
7. If there are multiple interpretations, provide the most likely one based on context

Focus areas:
- Policy terms and conditions
- Coverage details and limitations
- Waiting periods and grace periods
- Exclusions and exceptions
- Claim procedures
- Legal compliance requirements

Provide confident, factual answers based on the document content and standard insurance policy practices."""
    
    def _build_user_prompt(self, question: str, context_chunks: List[Dict], additional_context: str = "") -> str:
        """Build user prompt with question and context"""
        prompt = f"Question: {question}\n\n"
        
        if additional_context:
            prompt += f"Additional Context: {additional_context}\n\n"
        
        prompt += "Relevant Document Excerpts:\n"
        
        for i, chunk in enumerate(context_chunks, 1):
            prompt += f"\n[Excerpt {i}] (Relevance: {chunk['similarity']:.2f})\n"
            prompt += f"{chunk['content']}\n"
            prompt += f"Source: {chunk['source']}\n"
        
        prompt += "\nPlease provide a comprehensive answer based on the above excerpts:"
        
        return prompt
    
    def _calculate_confidence(self, search_results: List[SearchResult], answer: str) -> float:
        """Calculate confidence score based on search results quality and answer"""
        if not search_results:
            return 0.1
        
        # Base confidence on similarity scores
        avg_similarity = sum(result.similarity_score for result in search_results[:3]) / min(3, len(search_results))
        
        # Adjust based on number of relevant results
        result_bonus = min(len(search_results) / 5.0, 0.2)
        
        # Adjust based on answer length and specificity
        answer_quality = 0.0
        if len(answer) > 50:  # Substantial answer
            answer_quality += 0.1
        if any(keyword in answer.lower() for keyword in ['specific', 'period', 'condition', 'limit', 'coverage']):
            answer_quality += 0.1
        if "not available" in answer.lower():
            answer_quality -= 0.2
        
        confidence = avg_similarity + result_bonus + answer_quality
        return max(0.0, min(1.0, confidence))
    
    def _generate_reasoning(self, question: str, search_results: List[SearchResult], answer: str) -> str:
        """Generate explanation of reasoning process"""
        reasoning_parts = []
        
        reasoning_parts.append(f"Question analysis: The question asks about '{question[:50]}...'")
        
        if search_results:
            reasoning_parts.append(f"Document search: Found {len(search_results)} relevant excerpts with similarity scores ranging from {search_results[-1].similarity_score:.2f} to {search_results[0].similarity_score:.2f}")
            
            top_sources = [result.chunk.chunk_id for result in search_results[:2]]
            reasoning_parts.append(f"Primary sources used: {', '.join(top_sources)}")
        else:
            reasoning_parts.append("Document search: No relevant excerpts found in the document")
        
        if "not available" in answer.lower():
            reasoning_parts.append("Conclusion: The specific information requested was not found in the available document excerpts")
        else:
            reasoning_parts.append("Conclusion: Answer derived from matching document content with high confidence")
        
        return " | ".join(reasoning_parts)
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text (rough estimation for Gemini)"""
        try:
            # Simple estimation: roughly 4 characters per token for English text
            return len(text) // 4
        except:
            # Fallback estimation based on words
            return int(len(text.split()) * 1.3)
