import os
import asyncio
import time
import sys

# Path setup
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(backend_dir)

from retrieval.engine import HybridEngine
from retrieval.services.orchestrator import ToolPlan

async def test_performance():
    print("Initializing HybridEngine...")
    engine = HybridEngine()
    
    query = "What are the main themes in the documents and how do they relate to current renewable energy trends?"
    
    print(f"\nTesting Query: {query}")
    print("-" * 50)
    
    # Test Fast Mode (should be parallel and skip reflection)
    start_time = time.time()
    response = await engine.chat_async(query, mode="fast")
    duration = time.time() - start_time
    
    print(f"FAST MODE RESULTS:")
    print(f"Duration: {duration:.2f}s")
    print(f"Response snippet: {response['response'][:100]}...")
    print(f"Tools used: {response['tools_used']}")
    print(f"Sub-queries: {response['sub_queries']}")
    print(f"Reflection loops: {response['reflection_loops']} (Expected 0 for fast mode)")
    print("-" * 50)
    
    # Test Normal Mode (should still be parallel but include reflection)
    # print("Testing Normal Mode (with reflection)...")
    # start_time = time.time()
    # response = await engine.chat_async(query, mode="accurate")
    # duration = time.time() - start_time
    # print(f"ACCURATE MODE RESULTS:")
    # print(f"Duration: {duration:.2f}s")
    # print(f"Reflection loops: {response['reflection_loops']}")

if __name__ == "__main__":
    asyncio.run(test_performance())
