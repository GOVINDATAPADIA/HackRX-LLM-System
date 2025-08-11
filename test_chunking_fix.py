#!/usr/bin/env python3
"""
Test the fixed chunking system
"""

import asyncio
import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

from app.services.document_processor import DocumentProcessor
from app.services.embedding_service import EmbeddingService

async def test_fixed_chunking():
    """Test the new chunking system with the Arogya Sanjeevani PDF"""
    
    pdf_url = "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf?sv=2023-01-03&st=2025-07-21T08%3A29%3A02Z&se=2025-09-22T08%3A29%3A00Z&sr=b&sp=r&sig=nzrz1K9Iurt%2BBXom%2FB%2BMPTFMFP3PRnIvEsipAX10Ig4%3D"
    
    print("🔧 TESTING FIXED CHUNKING SYSTEM")
    print("=" * 50)
    
    try:
        # Initialize services
        document_processor = DocumentProcessor()
        embedding_service = EmbeddingService()
        
        print("✅ Services initialized")
        print(f"📊 Configuration:")
        print(f"   • Chunk size: {document_processor.chunk_size} characters")
        print(f"   • Chunk overlap: {document_processor.chunk_overlap} characters")
        print(f"   • Max chunk size: {document_processor.max_chunk_size} bytes")
        print(f"   • Similarity threshold: {embedding_service.similarity_threshold}")
        print()
        
        # Step 1: Download and process with new chunking
        print("📥 Step 1: Downloading and processing PDF...")
        file_path = await document_processor.download_document(pdf_url)
        chunks = await document_processor.process_document(file_path)
        
        print(f"🎉 SUCCESS! Created {len(chunks)} chunks (was 1 before)")
        print()
        
        # Show chunk analysis
        if chunks:
            print("📊 Chunk Analysis:")
            total_chars = sum(len(chunk.content) for chunk in chunks)
            avg_size = total_chars // len(chunks)
            
            print(f"   • Total chunks: {len(chunks)}")
            print(f"   • Average chunk size: {avg_size} characters")
            print(f"   • Size range: {min(len(c.content) for c in chunks)} - {max(len(c.content) for c in chunks)} chars")
            print()
            
            print("📝 Sample chunks:")
            for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
                print(f"\n   Chunk {i+1}:")
                print(f"   • Size: {len(chunk.content)} characters")
                print(f"   • ID: {chunk.chunk_id}")
                print(f"   • Preview: {chunk.content[:150]}...")
                print("   " + "-" * 50)
        
        # Step 2: Test embeddings creation
        print("\n🧠 Step 2: Creating embeddings...")
        chunks_with_embeddings = await embedding_service.create_embeddings(chunks[:5])  # Test with first 5
        print(f"✅ Created {len(chunks_with_embeddings)} embeddings")
        
        # Step 3: Test search functionality
        print("\n🔍 Step 3: Testing search with new chunks...")
        
        # Clear old data and add new chunks
        embedding_service.clear_index()
        await embedding_service.add_to_index(chunks_with_embeddings)
        
        # Test searches
        test_queries = [
            "age eligibility criteria",
            "co-payment percentage",
            "ambulance coverage",
            "pre-existing conditions"
        ]
        
        for query in test_queries:
            print(f"\n   🔎 Testing: '{query}'")
            results = await embedding_service.search(query, k=3)
            print(f"      Found: {len(results)} relevant chunks")
            
            if results:
                top_result = results[0]
                print(f"      Top result similarity: {top_result.similarity_score:.3f}")
                print(f"      Content preview: {top_result.chunk.content[:100]}...")
        
        print(f"\n🎉 CHUNKING FIX COMPLETE!")
        print(f"✅ System now creates {len(chunks)} semantic chunks instead of 1 giant chunk")
        print(f"✅ Search should now find relevant content with similarity threshold: {embedding_service.similarity_threshold}")
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_fixed_chunking())
