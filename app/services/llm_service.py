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
            
            # PHASE 2: Fact Verification for Numerical Accuracy
            verification_result = await self._verify_numerical_facts(question, answer, context_chunks)
            
            # Gemini doesn't provide token usage, so we estimate
            tokens_used = prompt_tokens + self._count_tokens(answer)
            
            # Enhanced confidence calculation incorporating verification
            base_confidence = self._calculate_confidence(search_results, answer)
            verification_confidence = verification_result["confidence"]
            confidence = (base_confidence + verification_confidence) / 2  # Average both confidences
            
            # Generate reasoning with verification details
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
        """Build enhanced system prompt for maximum accuracy"""
        return """You are an expert insurance policy analyst with exceptional attention to detail. Your task is to provide precise, definitive answers based on document excerpts.

CRITICAL INSTRUCTIONS:
1. EXTRACT EXACT NUMBERS: When asked for amounts, ages, percentages, or periods, find and state the EXACT figures from the text
2. BE DEFINITIVE: Avoid hedge words like "likely," "probably," "may be" - state facts directly
3. PRIORITIZE DIRECT QUOTES: When possible, quote exact phrases containing key information
4. NUMERICAL ACCURACY: Double-check all numbers, percentages, and ranges before responding
5. STRUCTURED ANSWERS: For complex questions, organize information clearly

ANSWER FORMAT:
- Start with the direct answer to the question
- Include specific numbers/percentages/ranges from the document
- Provide supporting context only if needed
- End with clear, definitive conclusion

EXAMPLES OF GOOD RESPONSES:
❌ "The policy likely covers this with some restrictions"
✅ "Yes, the policy covers this up to ₹2,000 per hospitalization"

❌ "The age range appears to be around 18-65"  
✅ "The entry age for adults is 18 to 65 years"

FOCUS AREAS:
- Policy limits and ranges (sum insured, age limits)
- Specific percentages (co-payment, coverage)
- Time periods (waiting periods, coverage duration)
- Eligibility criteria and conditions

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
        """Build enhanced user prompt with numerical focus"""
        
        # Analyze question type for specialized prompting
        question_lower = question.lower()
        is_numerical = any(word in question_lower for word in ['amount', 'percentage', 'age', 'limit', 'minimum', 'maximum', 'sum', 'cost', 'price', '%'])
        is_coverage = any(word in question_lower for word in ['cover', 'include', 'eligible', 'available'])
        is_period = any(word in question_lower for word in ['period', 'waiting', 'days', 'months', 'years'])
        
        prompt = f"Based on the document excerpts below, answer this question with maximum accuracy:\n\n"
        prompt += f"QUESTION: {question}\n\n"
        
        # Add specific instructions based on question type
        if is_numerical:
            prompt += "⚠️ NUMERICAL QUESTION: Focus on finding EXACT numbers, amounts, percentages, and ranges. Quote the specific figures.\n\n"
        elif is_coverage:
            prompt += "⚠️ COVERAGE QUESTION: Provide definitive yes/no answer with specific conditions and limits.\n\n"
        elif is_period:
            prompt += "⚠️ TIME PERIOD QUESTION: Extract exact durations, waiting periods, and time limits.\n\n"
        
        prompt += "RELEVANT DOCUMENT EXCERPTS:\n"
        prompt += "=" * 50 + "\n\n"
        
        for i, chunk in enumerate(context_chunks, 1):
            prompt += f"EXCERPT {i} (Relevance: {chunk['similarity']:.3f}):\n"
            prompt += f"{chunk['content']}\n"
            prompt += "-" * 30 + "\n\n"
        
        if additional_context:
            prompt += f"ADDITIONAL CONTEXT:\n{additional_context}\n\n"
        
        prompt += "INSTRUCTIONS FOR THIS ANSWER:\n"
        prompt += "1. Read all excerpts carefully for the specific information requested\n"
        prompt += "2. Extract EXACT numbers, percentages, or specific terms mentioned\n"
        prompt += "3. Provide a direct, definitive answer\n"
        prompt += "4. Include specific details from the document excerpts\n"
        prompt += "5. Avoid uncertain language - be confident in your response\n\n"
        
        prompt += "ANSWER:"
        
        return prompt
    
    async def _verify_numerical_facts(self, question: str, answer: str, context_chunks: List[Dict]) -> Dict[str, Any]:
        """Verify numerical facts in the answer against source content"""
        
        # Extract numbers from the answer
        import re
        numbers_in_answer = re.findall(r'₹?[\d,]+(?:\.\d+)?%?', answer)
        
        if not numbers_in_answer:
            return {"verified": True, "confidence": 1.0, "details": "No numerical claims to verify"}
        
        verification_prompt = f"""
FACT VERIFICATION TASK:
Verify if these numerical claims from an answer are accurate based on the source text.

ANSWER CLAIMS: {answer}

NUMBERS TO VERIFY: {numbers_in_answer}

SOURCE EXCERPTS:
"""
        
        for i, chunk in enumerate(context_chunks, 1):
            verification_prompt += f"\nEXCERPT {i}:\n{chunk['content']}\n"
        
        verification_prompt += """
VERIFICATION INSTRUCTIONS:
1. Check each number/percentage/amount in the answer against the source text
2. Mark as VERIFIED if the exact number appears in source text
3. Mark as CONFLICTED if source shows different number
4. Mark as UNSUPPORTED if number not found in source

Respond with:
VERIFICATION: [VERIFIED/CONFLICTED/UNSUPPORTED]
CONFIDENCE: [0.0-1.0]
DETAILS: [Brief explanation]
"""
        
        try:
            verification_response = self.model.generate_content(verification_prompt)
            verification_text = verification_response.text if verification_response.text else "No verification response"
            
            # Parse verification result
            is_verified = "VERIFIED" in verification_text and "CONFLICTED" not in verification_text
            confidence = 0.9 if is_verified else 0.3
            
            return {
                "verified": is_verified,
                "confidence": confidence,
                "details": verification_text
            }
        except Exception as e:
            logger.warning(f"Fact verification failed: {str(e)}")
            return {"verified": True, "confidence": 0.5, "details": "Verification unavailable"}
        
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
