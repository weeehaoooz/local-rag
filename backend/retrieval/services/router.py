import os
import json
import logging
from enum import Enum
from llama_index.llms.ollama import Ollama

logger = logging.getLogger(__name__)

class QueryType(str, Enum):
    LOCAL = "LOCAL"    # Specific entity/fact queries
    GLOBAL = "GLOBAL"  # Thematic/summary queries across documents
    HYBRID = "HYBRID"  # Specific topic needing broader context
    GENERIC = "GENERIC" # Non-searchable/conversational queries
    UNKNOWN = "UNKNOWN"

class RouterService:
    def __init__(self, llm: Ollama):
        self.llm = llm

    def is_global_query_fallback(self, query: str) -> bool:
        global_keywords = [
            "main theme", "overall", "summarize", "summary", "overview",
            "big picture", "across all", "general", "common pattern",
            "recurring", "high level", "key takeaway", "in general",
            "holistic", "what are the themes", "across documents",
        ]
        query_lower = query.lower()
        return any(kw in query_lower for kw in global_keywords)

    async def classify_query(self, query: str) -> tuple[QueryType, list[str]]:
        system_prompt = (
            "You are an intelligent query router for a RAG system.\n"
            "Given the user's query, classify it into exactly one of three categories:\n"
            "1. LOCAL: The query is specific, fact-based, and targets particular entities, metrics, or isolated facts. "
            "It requires precise retrieval from specific documents.\n"
            "2. GLOBAL: The query is broad, abstract, or asks for a high-level summary across the entire dataset or multiple documents. "
            "Examples: 'What are the main themes?', 'Summarize the whole report.'\n"
            "3. HYBRID: The query sits in between. It asks for thematic information but tied to specific concepts or entities. "
            "Requires both broad understanding and specific details.\n\n"
            "IMPORTANT: Your response MUST be valid JSON with two keys: 'type' (the classification) and 'keywords' (a list of keywords extracted from the query).\n"
            "Format exactly like this:\n"
            "{\"type\": \"GLOBAL\", \"keywords\": [\"themes\", \"report\"]}"
        )
        try:
            response = await self.llm.acomplete(f"{system_prompt}\n\nUser Query: {query}")
            raw_text = response.text.strip()
            # Clean possible markdown formatting
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
            raw_text = raw_text.strip()
            
            parsed = json.loads(raw_text)
            qtype_str = parsed.get("type", "UNKNOWN").upper()
            keywords = parsed.get("keywords", [])
            
            try:
                qtype = QueryType(qtype_str)
            except ValueError:
                qtype = QueryType.UNKNOWN
            
            # Fallback
            if qtype in (QueryType.UNKNOWN, QueryType.LOCAL) and self.is_global_query_fallback(query):
                 qtype = QueryType.GLOBAL
            
            return qtype, keywords
        except Exception as e:
            logger.error(f"Error classifying query: {str(e)}")
            return QueryType.HYBRID, []
