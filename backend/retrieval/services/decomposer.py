import json
import logging

logger = logging.getLogger(__name__)

class QueryDecomposer:
    """
    Analyzes complex user queries and decomposes them into independent sub-queries.
    This enables Plan-and-Solve reasoning, where the engine searches for multiple
    facts in parallel before synthesizing a final answer.
    """

    def __init__(self, llm):
        self.llm = llm

    def split_query(self, query: str) -> list[str]:
        """
        Inspects the query. If it is complex (e.g. asking for comparisons, multiple
        unrelated facts, or step-by-step reasoning), splits it into 2-3 simpler
        sub-queries. Otherwise, returns a list containing just the original query.
        """
        prompt = (
            "You are an expert query decomposer for an AI search engine.\n"
            "Analyze the given user query. If the query is complex (asking for a comparison between two "
            "entities, or asking about multiple distinct topics), break it down into 2 or 3 standalone, "
            "simple sub-queries.\n"
            "If the query is already simple and focuses on a single factual topic, return an array containing just the original query.\n\n"
            "Guidelines:\n"
            "- Sub-queries must be completely standalone (e.g. do not say 'its revenue', say 'Company X revenue').\n"
            "- Do not split a simple query. Just return it.\n"
            "- Maximum of 3 sub-queries.\n\n"
            "You MUST respond with a valid JSON list of strings and nothing else.\n"
            "Example Complex: [\"Apple Q1 revenue\", \"Google Q1 revenue\"]\n"
            "Example Simple: [\"What is the capital of France?\"]\n\n"
            f"User Query: {query}"
        )

        try:
            response = self.llm.complete(prompt)
            raw_text = response.text.strip()
            
            # Clean possible markdown formatting
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
                raw_text = raw_text.strip()
                
            start = raw_text.find("[")
            end = raw_text.rfind("]") + 1
            
            if start != -1 and end > start:
                parsed = json.loads(raw_text[start:end])
                if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
                    # Filter empty or overly long gibberish
                    valid_sub_queries = [q for q in parsed if len(q.strip()) > 2]
                    
                    if not valid_sub_queries:
                        raise ValueError("Parsed JSON list was empty or invalid.")
                        
                    # Cap at max 3 to prevent runaway retrieval loops
                    final_queries = valid_sub_queries[:3]
                    
                    if len(final_queries) > 1:
                        logger.info(f"[Decomposer] Query split into {len(final_queries)} sub-queries: {final_queries}")
                    
                    return final_queries

        except Exception as e:
            logger.error(f"[Decomposer] Query split failed, defaulting to original query: {e}")

        # Fallback: Treat as simple query
        return [query]
