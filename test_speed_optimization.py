#!/usr/bin/env python3
"""
Speed test for the optimized system - target under 30 seconds
"""

import asyncio
import time
import json
from app.services.advanced_query_handler import AdvancedQueryHandler

async def test_speed():
    """Test response time with sample question"""
    
    # Sample question
    test_question = "What is the ambulance coverage limit?"
    
    print("🚀 Testing Speed Optimizations...")
    print(f"📋 Question: {test_question}")
    print("⏱️  Starting timer...")
    
    start_time = time.time()
    
    try:
        # Initialize handler
        handler = AdvancedQueryHandler()
        
        # Run query
        result = await handler.run_hackrx_query(test_question)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\n✅ SPEED TEST RESULTS:")
        print(f"⏱️  Total Time: {total_time:.2f} seconds")
        print(f"🎯 Target: Under 30 seconds")
        print(f"📊 Status: {'✅ PASS' if total_time < 30 else '❌ FAIL'}")
        print(f"\n📝 Answer Preview: {result.answer[:100]}...")
        print(f"🔍 Sources: {len(result.sources)} chunks used")
        print(f"📈 Confidence: {result.confidence}")
        
        return total_time < 30
        
    except Exception as e:
        end_time = time.time()
        total_time = end_time - start_time
        print(f"❌ Test failed after {total_time:.2f}s: {str(e)}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_speed())
    print(f"\n{'🎉 Speed optimization successful!' if success else '⚠️ Speed optimization needed.'}")
