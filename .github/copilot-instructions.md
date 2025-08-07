<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# HackRx LLM-Powered Query Retrieval System

This is a FastAPI-based intelligent document processing system for the Bajaj HackRx hackathon. The system processes PDFs, DOCX, and email documents to answer contextual questions using LLM technology.

## Key Components:
- **FastAPI backend** with RESTful API endpoints
- **Document processing** for PDF, DOCX, and email formats
- **FAISS vector database** for semantic search
- **OpenAI GPT-4 integration** for intelligent question answering
- **Embeddings service** using OpenAI's text-embedding-ada-002

## Architecture:
1. Document Processing → Extract and chunk text
2. Embedding Creation → Generate vector representations
3. Vector Database → Store in FAISS for fast retrieval
4. Semantic Search → Find relevant document sections
5. LLM Processing → Generate contextual answers
6. JSON Response → Return structured results

## Development Guidelines:
- Follow the existing service pattern architecture
- Use proper error handling and logging
- Implement async/await patterns for I/O operations
- Maintain type hints and Pydantic models
- Focus on token efficiency and response latency
- Ensure explainable decision reasoning

## API Authentication:
Use Bearer token: `cd1d783d335b1b00dbe6e50b828060c5475a425013da0d6d7cbf1092b61d32a0`

## Target Domains:
- Insurance policy analysis
- Legal document processing  
- HR compliance checking
- Contract clause matching
