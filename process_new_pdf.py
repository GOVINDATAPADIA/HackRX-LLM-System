#!/usr/bin/env python3
"""
Direct PDF Processing Script
- Downloads and processes new PDF
- Generates embeddings
- Stores in Pinecone
"""

import asyncio
import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import EmbeddingService

async def process_new_pdf():
    """Process the new Arogya Sanjeevani PDF document"""
    
    # Your new PDF URL
    pdf_url = "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D"
    
    print("🚀 PROCESSING NEW AROGYA SANJEEVANI POLICY PDF")
    print("=" * 60)
    print(f"📄 PDF URL: {pdf_url[:80]}...")
    print()
    
    try:
        # Initialize services
        document_processor = DocumentProcessor()
        embedding_service = EmbeddingService()
        
        print("✅ Services initialized")
        
        # Step 1: Download the PDF
        print("📥 Step 1: Downloading PDF...")
        file_path = await document_processor.download_document(pdf_url)
        print(f"✅ PDF downloaded to: {file_path}")
        
        # Step 2: Process document into chunks using new chunking model
        print("🔧 Step 2: Processing document with new chunking model...")
        chunks = await document_processor.process_document(file_path)
        print(f"✅ Created {len(chunks)} chunks")
        
        # Show chunk details
        if chunks:
            sample_chunk = chunks[0]
            print(f"📊 Sample chunk details:")
            print(f"   • Length: {len(sample_chunk.content)} characters")
            print(f"   • Words: {sample_chunk.metadata.get('word_count', 'N/A')}")
            print(f"   • Content preview: {sample_chunk.content[:100]}...")
        
        # Step 3: Create embeddings
        print("🧠 Step 3: Creating embeddings with Google AI...")
        chunks_with_embeddings = await embedding_service.create_embeddings(chunks)
        print(f"✅ Generated {len(chunks_with_embeddings)} embeddings")
        
        # Show embedding details
        if chunks_with_embeddings:
            sample_embedding = chunks_with_embeddings[0].embedding
            print(f"📐 Embedding dimension: {len(sample_embedding)}")
            print(f"🎯 Sample embedding values: {sample_embedding[:5]}...")
        
        # Step 4: Store in Pinecone
        print("💾 Step 4: Storing embeddings in Pinecone...")
        await embedding_service.add_to_index(chunks_with_embeddings)
        print("✅ All embeddings stored successfully!")
        
        # Step 5: Verify storage
        print("🔍 Step 5: Verifying storage...")
        stats = embedding_service.index.describe_index_stats()
        print(f"📊 Index now contains {stats.get('total_vector_count', 0)} vectors")
        
        # Test search
        print("🧪 Step 6: Testing search functionality...")
        test_results = await embedding_service.search("policy coverage", k=3)
        print(f"🎯 Search test found {len(test_results)} relevant chunks")
        
        if test_results:
            print("📝 Top search result:")
            top_result = test_results[0]
            print(f"   • Similarity: {top_result.similarity_score:.4f}")
            print(f"   • Content: {top_result.chunk.content[:150]}...")
        
        print("\n🎉 SUCCESS! New PDF processed and ready for Q&A!")
        print("💡 You can now start your FastAPI server and ask questions about the Arogya Sanjeevani Policy")
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(process_new_pdf())
