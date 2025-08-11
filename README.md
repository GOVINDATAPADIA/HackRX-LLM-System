# LLM-Powered Intelligent Query-Retrieval System
## Bajaj HackRx 2025 Submission

A sophisticated document processing and question-answering system that leverages Large Language Models (LLMs) to analyze insurance, legal, HR, and compliance documents with **automatic embedding generation** and high accuracy responses.

## ⚡ Key Features

- **🚀 Automatic Processing**: No manual steps required - just send URL + questions and get answers!
- **� Single Document Mode**: Automatically clears old data when new PDF is provided
- **�📄 Multi-format Support**: Handles PDFs, DOCX, and email documents
- **🧠 Smart Embeddings**: Uses Google Gemini for high-quality vector representations
- **💾 Vector Database**: Pinecone for fast semantic search
- **🎯 Intelligent QA**: Context-aware answers with explainable reasoning
- **🔒 Secure API**: Bearer token authentication
- **⚡ Smart Caching**: Same URL = reuse embeddings, different URL = auto-clear and reload

## 🎯 Problem Statement Solution

This system addresses the HackRx challenge by implementing:

1. **Multi-format Document Processing**: Handles PDFs, DOCX, and email documents
2. **Semantic Search**: Uses Pinecone vector database with Google embeddings
3. **Intelligent Question Answering**: Gemini-1.5-pro powered responses with context
4. **Explainable AI**: Provides reasoning and source traceability
5. **RESTful API**: FastAPI-based backend with authentication
6. **⭐ NEW: Automatic Embedding Generation**: Documents are processed automatically when API is called

## 🏗️ System Architecture

```
📡 API Request (URL + Questions)
    ↓
🔍 Same document as before?
    ↓ (if different)
🧹 Auto-clear old Pinecone data
    ↓
📥 Document Processor (Download & Chunk)
    ↓  
🧠 Embedding Service (Google Gemini embedding-001)
    ↓
💾 Vector Database (Pinecone Storage - Single Document Mode)
    ↓
🎯 Query Handler (Semantic Search + LLM)
    ↓
✅ JSON Response (Structured Answers)
```

## ⚡ Quick Start (No Manual Processing Required!)

### Prerequisites
- Python 3.8+
- Google Gemini API key
- Pinecone API key

### Installation

1. **Clone and setup environment:**
```bash
cd HackRX-LLM-System
python -m venv venv
venv\Scripts\activate  # Windows
# or source venv/bin/activate  # Linux/Mac
```

2. **Install dependencies (choose one method):**

   **Method A - Automatic (Recommended):**
   ```bash
   python install.py
   ```

   **Method B - Use setup script:**
   ```bash
   setup.bat  # Windows
   # or ./setup.sh  # Linux/Mac
   ```

   **Method C - Manual installation:**
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -r requirements-minimal.txt
   ```

3. **Configure environment:**
```bash
copy .env.example .env  # Windows
# or cp .env.example .env  # Linux/Mac
# Edit .env file with your API keys:
# - GOOGLE_API_KEY (Gemini API key)
# - PINECONE_API_KEY  
# - PINECONE_ENVIRONMENT
```

4. **Run the application:**
```bash
python main.py
```

The API will be available at `http://localhost:8000`

> **⚠️ Installation Issues?** Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for Python 3.12+ compatibility solutions.

## 📡 API Usage

### Main Endpoint
```http
POST /api/v1/hackrx/run
Content-Type: application/json
Authorization: Bearer cd1d783d335b1b00dbe6e50b828060c5475a425013da0d6d7cbf1092b61d32a0

{
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the grace period for premium payment?",
        "What is the waiting period for pre-existing diseases?",
        "Does this policy cover maternity expenses?"
    ]
}
```

### Response Format
```json
{
    "answers": [
        "A grace period of thirty days is provided for premium payment after the due date.",
        "There is a waiting period of thirty-six (36) months for pre-existing diseases.",
        "Yes, the policy covers maternity expenses with 24 months continuous coverage requirement."
    ]
}
```

## 🔧 Technical Features

### Document Processing
- **PDF**: PyPDF2 for text extraction
- **DOCX**: python-docx for Word document processing  
- **Email**: Built-in email library for .eml files
- **Chunking**: Intelligent text segmentation with overlap

### Vector Search
- **FAISS**: Facebook AI Similarity Search for fast retrieval
- **Embeddings**: OpenAI text-embedding-ada-002 (1536 dimensions)
- **Similarity**: Cosine similarity with configurable threshold

### LLM Integration
- **Model**: GPT-4 for high-quality responses
- **Context**: Intelligent prompt engineering with relevant chunks
- **Token Optimization**: Efficient token usage tracking
- **Temperature**: Low temperature (0.1) for factual accuracy

### Performance Optimization
- **Caching**: Document processing cache to avoid reprocessing
- **Async**: Asynchronous operations for better performance
- **Batching**: Efficient embedding creation in batches
- **Indexing**: Persistent FAISS index storage

## 📊 Evaluation Metrics

The system is optimized for the HackRx scoring criteria:

1. **Accuracy**: Precise clause matching and context understanding
2. **Token Efficiency**: Optimized prompts and response generation
3. **Latency**: Fast vector search and cached processing
4. **Reusability**: Modular service architecture
5. **Explainability**: Clear reasoning and source attribution

## 📁 Project Structure

```
HackRX-LLM-System/
├── app/
│   ├── core/
│   │   ├── config.py          # Application settings
│   │   └── logging_config.py  # Logging configuration
│   ├── models/
│   │   └── schemas.py         # Pydantic models
│   ├── services/
│   │   ├── document_processor.py  # Document handling
│   │   ├── embedding_service.py   # Vector operations
│   │   ├── llm_service.py         # GPT-4 integration
│   │   └── query_handler.py       # Main orchestrator
│   └── __init__.py
├── data/                      # FAISS index storage
├── logs/                      # Application logs
├── main.py                    # FastAPI application
├── requirements.txt           # Dependencies
├── .env.example              # Environment template
└── README.md                 # This file
```

## 🔐 Security

- **Authentication**: Bearer token validation
- **CORS**: Configurable cross-origin resource sharing
- **Input Validation**: Pydantic model validation
- **Error Handling**: Comprehensive exception management

## 🧪 Testing

Run the health check endpoint:
```bash
curl -X GET "http://localhost:8000/health"
```

Test with the provided sample data:
```bash
curl -X POST "http://localhost:8000/api/v1/hackrx/run" \
  -H "Authorization: Bearer cd1d783d335b1b00dbe6e50b828060c5475a425013da0d6d7cbf1092b61d32a0" \
  -H "Content-Type: application/json" \
  -d @test_request.json
```

## 🤖 AI Integration

### OpenAI Services Used
- **GPT-4**: Main reasoning and answer generation
- **text-embedding-ada-002**: Vector embeddings for semantic search

### Prompt Engineering
- Specialized system prompts for document analysis
- Context-aware user prompts with relevant excerpts
- Confidence scoring and reasoning generation

## 📈 Performance

- **Document Processing**: ~2-5 seconds per document
- **Embedding Creation**: ~100ms per chunk
- **Vector Search**: <50ms for similarity search
- **LLM Response**: ~1-3 seconds per question

## 🛠️ Development

### Adding New Document Types
1. Extend `DocumentProcessor` class
2. Add format detection in `_get_file_extension`
3. Implement extraction method
4. Update `SUPPORTED_FORMATS` in config

### Customizing LLM Behavior
- Modify system prompts in `LLMService._build_system_prompt`
- Adjust confidence calculation in `_calculate_confidence`
- Update reasoning generation in `_generate_reasoning`

## 📝 License

This project is created for the Bajaj HackRx 2025 hackathon.

## 👥 Contributors

Team HackRx LLM System
- Developed for Bajaj HackRx 2025
- Focus: Insurance & Legal Document AI

---

**Ready for deployment and testing!** 🚀
