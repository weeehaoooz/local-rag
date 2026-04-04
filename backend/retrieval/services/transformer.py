import logging

logger = logging.getLogger(__name__)

class QueryTransformer:
    """
    Mutates and enhances user queries prior to semantic retrieval.
    Includes Coreference Resolution and HyDE (Hypothetical Document Embeddings).
    """

    def __init__(self, llm):
        self.llm = llm

    async def resolve_coreference(self, query: str, history: list[dict] = None) -> str:
        """
        Rewrites a query to replace pronouns (it, they, them, he, she) or 
        ambiguous references (that, this) with explicit entities from the chat history.
        """
        if not history:
            return query

        # Combine recent history into a concise transcript
        transcript_parts = []
        # Keep only the last 4 turns to avoid exceeding context or diluting focus
        recent_history = history[-4:]
        
        for turn in recent_history:
            role = turn.get("role", "User").upper()
            content = turn.get("content", turn.get("message", ""))
            if content:
                transcript_parts.append(f"{role}: {content}")
                
        transcript = "\n".join(transcript_parts)
        if not transcript:
            return query

        prompt = (
            "You are an intelligent query re-writer for a search engine.\n"
            "Given the Chat History below, look at the Latest User Query.\n"
            "If the Latest User Query contains unambiguous pronouns (like 'it', 'they', 'this', 'that company') "
            "that refer to something in the Chat History, rewrite the query to replace the pronouns with the explicit entity names.\n"
            "If the query is already self-contained, return it exactly as is.\n\n"
            "DO NOT answer the query. Return ONLY the rewritten query text. No quotes, no preamble.\n\n"
            f"--- CHAT HISTORY ---\n{transcript}\n\n"
            f"--- LATEST USER QUERY ---\n{query}"
        )

        try:
            response = await self.llm.acomplete(prompt)
            rewritten = response.text.strip().strip('"\'')
            if rewritten and len(rewritten) > 2:
                if rewritten.lower() != query.lower():
                    logger.info(f"[Transformer] Coreference resolved: '{query}' → '{rewritten}'")
                return rewritten
        except Exception as e:
            logger.error(f"[Transformer] Coreference resolution failed: {e}")

        return query

    async def generate_hyde_document(self, query: str) -> str:
        """
        Generates a Hypothetical Document Embedding (HyDE) string.
        By creating a hallucinated 'ideal' answer to the query, vector semantic 
        overlap is significantly improved compared to searching with a short question.
        """
        prompt = (
            "You are an expert answering a user's question.\n"
            "Please write a short, single-paragraph hypothetical answer to the user's question.\n"
            "Write the response strictly from an objective, factual perspective as if it were an encyclopedia excerpt.\n"
            "Do not include any conversational filler (e.g., 'Sure, here is...').\n\n"
            f"Question: {query}\n\n"
            "Hypothetical Answer:"
        )

        try:
            response = await self.llm.acomplete(prompt)
            hyde_doc = response.text.strip()
            
            # Combine the original query with the hallucinated document.
            # This provides both exact entity matching from the query and latent semantic surface from HyDE.
            combined = f"{query}\n\n{hyde_doc}"
            logger.info(f"[Transformer] Generated HyDE document for query.")
            return combined
        except Exception as e:
            logger.error(f"[Transformer] HyDE generation failed: {e}")
            return query
