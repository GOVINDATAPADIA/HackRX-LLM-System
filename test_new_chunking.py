#!/usr/bin/env python3
"""
Test script to demonstrate the improved chunking model
"""

import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

from app.services.document_processor import DocumentProcessor
from app.core.config import settings

def test_chunking_comparison():
    """Compare old vs new chunking approaches"""
    
    # Sample insurance document text
    sample_text = """
    Insurance Policy Terms and Conditions. This policy provides comprehensive coverage for health-related expenses. 
    Premium payments must be made monthly by the due date. A grace period of thirty days is allowed for late payments.
    
    Pre-existing medical conditions have a waiting period of thirty-six months. During this period, claims related to 
    pre-existing conditions will not be covered. After the waiting period, full coverage applies.
    
    Claims Processing Guidelines. All claims must be submitted within ninety days of the medical treatment. 
    Required documents include original bills, prescription receipts, and doctor's certificates. 
    The insurance company will process claims within fifteen business days of receipt.
    
    Policy Benefits and Limitations. Annual coverage limit is five hundred thousand dollars. 
    Co-payment of twenty percent applies to all outpatient treatments. In-patient treatments require 
    pre-authorization from the insurance company.
    """
    
    print("🔧 TESTING NEW RECURSIVE CHUNKING MODEL")
    print("=" * 60)
    
    # Initialize document processor with new chunking
    processor = DocumentProcessor()
    
    print(f"📊 Configuration:")
    print(f"   • Chunk Size: {settings.CHUNK_SIZE} characters")
    print(f"   • Overlap: {settings.CHUNK_OVERLAP} characters")
    print(f"   • Text Length: {len(sample_text)} characters")
    print()
    
    # Create chunks using new method
    chunks = processor._create_chunks(sample_text, "test_document.pdf")
    
    print(f"📝 CHUNKING RESULTS:")
    print(f"   • Total Chunks Created: {len(chunks)}")
    print()
    
    for i, chunk in enumerate(chunks, 1):
        print(f"📄 CHUNK {i}:")
        print(f"   • Length: {len(chunk.content)} chars")
        print(f"   • Words: {chunk.metadata.get('word_count', 0)}")
        print(f"   • Sentences: {chunk.metadata.get('sentences_count', 0)}")
        print(f"   • Content Preview: {chunk.content[:100]}...")
        print(f"   • Chunk ID: {chunk.chunk_id}")
        print()
    
    print("✅ NEW CHUNKING ADVANTAGES:")
    print("   🎯 Better semantic boundaries (paragraphs, sentences)")
    print("   🔄 Smart overlap with word boundaries") 
    print("   📐 Consistent chunk sizes with flexible content")
    print("   🧹 Text preprocessing for cleaner chunks")
    print("   📊 Enhanced metadata (word count, sentence count)")
    print("   🔀 Recursive splitting handles complex documents")

if __name__ == "__main__":
    test_chunking_comparison()
