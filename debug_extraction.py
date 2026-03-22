import os
import time
from llama_index.llms.ollama import Ollama
from dotenv import load_dotenv

load_dotenv()

def debug_llm_latency():
    print("--- Debugging Ollama Latency with Context ---")
    
    # 1. Setup Ollama
    llm = Ollama(
        model="llama3:latest",
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        request_timeout=60.0
    )
    
    # 2. Simulate a realistic context (20 entities)
    entities = [f"Entity_{i} (Type_{i%5})" for i in range(20)]
    context = (
        "Recently Extracted Entities for STRICT NORMALIZATION ALIGNMENT "
        "(Use exactly these names if referring to the same concept!): "
        + ", ".join(entities)
        + "\n"
    )
    
    sample_text = """
    John Doe is a Senior Software Engineer at Google. 
    He has 10 years of experience in Python and Java.
    Previously, he worked at Microsoft as a Junior Developer.
    He graduated from Stanford University.
    """
    
    prompt = f"{context}\n\nExtract the triplets now from this text:\n{sample_text}"
    
    print(f"Prompt length: {len(prompt)} chars (~{len(prompt)//4} tokens)")
    print("Starting LLM call...")
    
    start_time = time.time()
    try:
        response = llm.complete(prompt)
        duration = time.time() - start_time
        print(f"LLM call successful! Time taken: {duration:.2f} seconds")
        print(f"Response (first 100 chars): {response.text[:100]}...")
    except Exception as e:
        duration = time.time() - start_time
        print(f"LLM call failed after {duration:.2f} seconds.")
        print(f"Error: {e}")

if __name__ == "__main__":
    debug_llm_latency()
