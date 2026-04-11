import asyncio
import os
import sys

# Add backend to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from retrieval.engine import HybridEngine
from retrieval.services.router import QueryType

async def test_generic():
    engine = HybridEngine()
    
    test_queries = [
        "hi there!",
        "what's up?",
        "who are you?",
        "What is the main theme of the recent reports?",
        "Explain the impact of AI on job markets based on the documents.",
        "Tell me something"
    ]
    
    print("\n" + "="*50)
    print("RUNNING GENERIC DETECTION TESTS")
    print("="*50 + "\n")
    
    for query in test_queries:
        print(f"\nQUERY: {query}")
        plan = await engine.orchestrator.plan_tools(query)
        print(f"RESULT: is_generic={plan.is_generic}, tools={plan.tools}, type={plan.fallback_query_type}")
        
        # Test engine integration
        context = await engine.get_context_async(query)
        print(f"ENGINE SUMMARY: {len(context['vector_context'])} vector results, {len(context['graph_context'])} graph results")
        
    print("\n" + "="*50)

if __name__ == "__main__":
    asyncio.run(test_generic())
