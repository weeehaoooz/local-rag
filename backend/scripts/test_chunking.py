import sys
import os
import asyncio

# Add backend to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import setup_models
from indexing.graph_extractor import _small_to_big_parse, _count_tokens
from llama_index.core.schema import Document

async def test_chunking():
    print("Setting up models and Settings...")
    setup_models()
    
    # Create a dummy large document
    text = (
        "# Introduction\n"
        "This is a test document with multiple sections. " * 50 + 
        "\n\n# Methodology\n" + 
        "The experiments were conducted in a controlled environment. " * 100 +
        "\n\n# Results\n" + 
        "We found that semantic splitting works better than fixed splitting. " * 150
    )
    doc = Document(text=text, metadata={"file_name": "test_doc.md"})
    
    print(f"Original Text Length: {len(text)} characters")
    print(f"Original Token Count: {_count_tokens(text)} tokens")
    
    print("\nRunning _small_to_big_parse (Small=256, Big=512)...")
    small_nodes, big_nodes = _small_to_big_parse(
        [doc], 
        small_chunk_size=256, 
        big_chunk_size=512
    )
    
    print(f"\nResults:")
    print(f"  Big (Parent) Nodes: {len(big_nodes)}")
    print(f"  Small (Child) Nodes: {len(small_nodes)}")
    
    # Check token counts
    for i, node in enumerate(big_nodes):
        tc = _count_tokens(node.get_content())
        print(f"  Big Node {i} Token Count: {tc}")
        if tc > 512 * 1.5:
             print(f"    WARNING: Big Node {i} exceeds limit!")

    for i, node in enumerate(small_nodes[:5]): # Check first 5 small nodes
        tc = _count_tokens(node.get_content())
        print(f"  Small Node {i} Token Count: {tc}")

if __name__ == "__main__":
    asyncio.run(test_chunking())
