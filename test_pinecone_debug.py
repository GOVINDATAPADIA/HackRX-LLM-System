#!/usr/bin/env python3
"""
Debug script to test Pinecone index creation and embedding storage
"""

import os
import sys
import asyncio
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

from app.core.config import settings
from app.services.embedding_service import EmbeddingService
from app.models.schemas import DocumentChunk

async def test_pinecone_debug():
    """Test Pinecone connection and embedding storage"""
    print("🔧 Debugging Pinecone Index Issues...")
    print(f"📊 Config - Vector Dimension: {settings.VECTOR_DIMENSION}")
    print(f"🏷️ Config - Index Name: {settings.PINECONE_INDEX_NAME}")
    
    try:
        # Initialize embedding service
        embedding_service = EmbeddingService()
        print("✅ EmbeddingService initialized")
        
        # Check index info
        index_info = embedding_service.pc.describe_index(settings.PINECONE_INDEX_NAME)
        print(f"📋 Existing Index Dimension: {index_info.dimension}")
        print(f"📋 Config Dimension: {embedding_service.dimension}")
        
        # Create a test chunk
        test_chunk = DocumentChunk(
            content="This is a test document chunk for Pinecone debugging.",
            chunk_id="test_chunk_001",
            metadata={
                "source": "debug_test",
                "chunk_index": 0,
                "character_count": 50
            }
        )
        
        print("🧪 Creating test embedding...")
        
        # Test embedding creation
        chunks_with_embeddings = await embedding_service.create_embeddings([test_chunk])
        test_embedding = chunks_with_embeddings[0].embedding
        
        print(f"📐 Test Embedding Dimension: {len(test_embedding)}")
        print(f"🎯 First 5 values: {test_embedding[:5]}")
        print(f"🎯 Last 5 values: {test_embedding[-5:]}")
        
        # Test adding to index
        print("💾 Testing Pinecone upsert...")
        await embedding_service.add_to_index(chunks_with_embeddings)
        
        print("✅ SUCCESS: Data saved to Pinecone!")
        
        # Test search
        print("🔍 Testing search...")
        results = await embedding_service.search("test document", k=1)
        print(f"🎯 Search Results: {len(results)} found")
        
        if results:
            print(f"📄 Result Content: {results[0].chunk.content[:100]}...")
            print(f"⭐ Similarity Score: {results[0].similarity_score}")
            print(f"🏆 Rank: {results[0].rank}")
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_pinecone_debug())
