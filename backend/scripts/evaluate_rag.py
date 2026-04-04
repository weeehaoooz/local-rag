import os
import json
import time
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np
import nest_asyncio

# Initialize nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Path hacks if needed, but normally running from project root or backend dir
# Assuming we run from the project root or backend directory with PYTHONPATH set
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, "backend"))

from backend.config import setup_models, VECTOR_DIR, BM25_DIR, STORAGE_DIR, DATA_DIR
from backend.retrieval.engine import HybridEngine

class Evaluator:
    """
    Evaluator class that uses an LLM and Embedding model to score RAG responses.
    """
    def __init__(self, llm, embed_model):
        self.llm = llm
        self.embed_model = embed_model
        print(f"Evaluator initialized with LLM: {getattr(llm, 'model', 'default')}")

    def get_embedding(self, text: str) -> List[float]:
        return self.embed_model.get_text_embedding(text)

    def compute_cosine_similarity(self, text1: str, text2: str) -> float:
        """Computes semantic similarity between two strings."""
        if not text1 or not text2:
            return 0.0
        vec1 = np.array(self.get_embedding(text1))
        vec2 = np.array(self.get_embedding(text2))
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def evaluate_faithfulness(self, query: str, context: str, answer: str) -> Dict[str, Any]:
        """
        Scores if the answer is grounded in the context.
        Returns a score (0.0 to 1.0) and reasoning.
        """
        prompt = f"""
        Evaluate if the provided ANSWER is faithful to the CONTEXT provided.
        Faithfulness means the answer does NOT contain information that is NOT present in the context.

        QUERY: {query}
        CONTEXT: {context}
        ANSWER: {answer}

        Provide your evaluation in the following JSON format:
        {{
            "score": <float between 0 and 1>,
            "reasoning": "<brief explanation of why the score was given>"
        }}
        """
        try:
            response = str(self.llm.complete(prompt))
            # Extract JSON from response if LLM adds extra text
            start = response.find("{")
            end = response.rfind("}") + 1
            return json.loads(response[start:end])
        except Exception as e:
            return {"score": 0.0, "reasoning": f"Error during evaluation: {e}"}

    def evaluate_relevancy(self, query: str, answer: str) -> Dict[str, Any]:
        """
        Scores how well the answer addresses the query.
        """
        prompt = f"""
        Evaluate how relevant the ANSWER is to the original QUERY.
        A relevant answer directly addresses the user's question without extraneous information.

        QUERY: {query}
        ANSWER: {answer}

        Provide your evaluation in the following JSON format:
        {{
            "score": <float between 0 and 1>,
            "reasoning": "<brief explanation of why the score was given>"
        }}
        """
        try:
            response = str(self.llm.complete(prompt))
            start = response.find("{")
            end = response.rfind("}") + 1
            return json.loads(response[start:end])
        except Exception as e:
            return {"score": 0.0, "reasoning": f"Error during evaluation: {e}"}

def run_evaluation(test_set_path: str, output_dir: str, limit: Optional[int] = None):
    """
    Main evaluation loop.
    """
    if not os.path.exists(test_set_path):
        print(f"Error: Test set not found at {test_set_path}")
        return

    with open(test_set_path, "r") as f:
        test_cases = json.load(f)

    if limit:
        test_cases = test_cases[:limit]

    # Initialize Engine and Evaluator
    print("\n--- Initializing RAG Engine ---")
    engine = HybridEngine()
    llm, embed_model = setup_models()
    evaluator = Evaluator(llm, embed_model)

    results = []
    total_metrics = {
        "faithfulness": 0.0,
        "answer_relevancy": 0.0,
        "semantic_similarity": 0.0,
        "latency_sec": 0.0
    }

    print(f"\n--- Running Evaluation on {len(test_cases)} cases ---")
    
    for i, case in enumerate(test_cases):
        query = case.get("query")
        reference = case.get("reference_answer", "")
        
        print(f"[{i+1}/{len(test_cases)}] Evaluating Query: '{query}'")
        
        start_time = time.time()
        # Trigger RAG Engine
        response_data = engine.chat(query)
        latency = time.time() - start_time
        
        answer = response_data["response"]
        # Extract all retrieved text for context evaluation
        # Note: We need get_context to get the raw text if chat doesn't return it directly
        # But chat mode does print/log it. Let's adjust HybridEngine or use get_context here.
        context_data = engine.get_context(query)
        full_context = "\n".join([t for t, s in context_data["vector_context"] + context_data["graph_context"]])

        # Run Eval Metrics
        faith_eval = evaluator.evaluate_faithfulness(query, full_context, answer)
        rel_eval = evaluator.evaluate_relevancy(query, answer)
        similarity = evaluator.compute_cosine_similarity(answer, reference) if reference else 0.0

        case_result = {
            "query": query,
            "response": answer,
            "reference": reference,
            "latency": latency,
            "metrics": {
                "faithfulness": faith_eval["score"],
                "faithfulness_reasoning": faith_eval["reasoning"],
                "answer_relevancy": rel_eval["score"],
                "answer_relevancy_reasoning": rel_eval["reasoning"],
                "semantic_similarity": similarity
            }
        }
        results.append(case_result)
        
        # Accumulate totals
        total_metrics["faithfulness"] += faith_eval["score"]
        total_metrics["answer_relevancy"] += rel_eval["score"]
        total_metrics["semantic_similarity"] += similarity
        total_metrics["latency_sec"] += latency
        
        print(f"  > Faithfulness: {faith_eval['score']:.2f} | Relevancy: {rel_eval['score']:.2f} | Similarity: {similarity:.2f}")

    # Summary Stats
    count = len(test_cases)
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_cases": count,
        "averages": {k: (v / count) for k, v in total_metrics.items()}
    }

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"eval_report_{timestamp_str}.json")
    
    final_report = {
        "summary": summary,
        "detail": results
    }
    
    with open(output_file, "w") as f:
        json.dump(final_report, f, indent=2)

    print("\n--- Evaluation Complete ---")
    print(f"Report saved to: {output_file}")
    print(f"Mean Faithfulness: {summary['averages']['faithfulness']:.2f}")
    print(f"Mean Relevancy: {summary['averages']['answer_relevancy']:.2f}")
    print(f"Mean Similarity: {summary['averages']['semantic_similarity']:.2f}")
    print(f"Avg Latency: {summary['averages']['latency_sec']:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG system performance.")
    parser.add_argument("--test-set", type=str, default=os.path.join(DATA_DIR, "eval_set.json"), help="Path to evaluation JSON.")
    parser.add_argument("--output-dir", type=str, default=os.path.join(project_root, "backend", "artifacts", "eval_results"), help="Directory to save reports.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of test cases to run.")
    
    args = parser.parse_args()
    run_evaluation(args.test_set, args.output_dir, args.limit)
