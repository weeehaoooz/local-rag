import ollama
from engine import HybridEngine

engine = HybridEngine()

def chat():
    print("--- Local Hybrid RAG Active (Confidential Mode) ---")
    while True:
        user_query = input("\nAsk a question (or 'exit'): ")
        if user_query.lower() == 'exit': break
        
        # Get context from both tools
        context = engine.get_context(user_query)
        
        # Construct the final prompt for Llama 3
        prompt = f"""
        Use the following retrieved information to answer the user's question.
        
        RELATIONAL DATA (Graph): {context['graph_context']}
        DOCUMENT STRUCTURE (PageIndex): {context['summary_context']}
        
        USER QUESTION: {user_query}
        """
        
        # Generate final response locally
        response = ollama.generate(model='llama3', prompt=prompt)
        print("\nAI RESPONSE:\n", response['response'])

if __name__ == "__main__":
    chat()
