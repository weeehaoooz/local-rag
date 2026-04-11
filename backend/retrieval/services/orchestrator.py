import json
import logging
from dataclasses import dataclass, field
from retrieval.services.router import QueryType

logger = logging.getLogger(__name__)

@dataclass
class ToolPlan:
    tools: list[str]
    rationale: str
    fallback_query_type: QueryType
    is_generic: bool = False
    sub_queries: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    resolved_query: str = ""

class ToolOrchestrator:
    """
    LLM-driven tool orchestrator for the Hybrid RAG engine.
    Given a query, it selects the appropriate combination of retrieval tools
    and performs pre-processing (decomposition, coref resolution).
    """
    
    AVAILABLE_TOOLS = [
        {
            "name": "vector_search",
            "description": "Semantic search over local documents. Use for specific facts, entities, definitions, and exact quotes from the knowledge base."
        },
        {
            "name": "graph_search",
            "description": "Knowledge graph retrieval. Use to find relationships between entities, temporal facts, or to expand context around specific subjects mentioned in the query."
        },
        {
            "name": "summary_search",
            "description": "Retrieves high-level overviews of entire documents. Use for questions like 'What is this document about?' or 'Summarize the report'."
        },
        {
            "name": "community_search",
            "description": "Retrieves global themes and cross-document patterns. Use for questions like 'What are the main themes across all documents?' or 'What is the big picture?'"
        },
        {
            "name": "web_search",
            "description": "Searches the live internet. ONLY use if the query explicitly asks for recent news, live data, or external knowledge not likely to be in internal documents."
        },
        {
            "name": "arxiv_search",
            "description": "Searches academia/ArXiv for papers. ONLY use if the query explicitly asks for research papers or academic studies."
        }
    ]

    def __init__(self, llm, fail_open: bool = True):
        self.llm = llm
        self.fail_open = fail_open
        
    async def analyze_request(self, query: str, history: list[dict] = None) -> ToolPlan:
        """
        One-stop shop for query analysis:
        1. Coreference resolution (from history)
        2. Query decomposition (if complex)
        3. Tool selection
        4. Keyword extraction
        5. Query type classification
        """
        tools_desc = "\n".join([f"- {t['name']}: {t['description']}" for t in self.AVAILABLE_TOOLS])
        
        # Format history for coref resolution
        history_str = ""
        if history:
            recent = history[-4:]
            parts = []
            for t in recent:
                role = t.get("role", "User").upper()
                content = t.get("content", t.get("message", ""))
                parts.append(f"{role}: {content}")
            history_str = "\n".join(parts)

        prompt = (
            "You are an expert RAG query analyzer and orchestrator.\n"
            "Analyze the user's latest query in the context of the chat history and provide a comprehensive execution plan.\n\n"
            f"--- CHAT HISTORY ---\n{history_str or 'No history'}\n\n"
            f"--- LATEST USER QUERY ---\n{query}\n\n"
            f"--- AVAILABLE SEARCH TOOLS ---\n{tools_desc}\n\n"
            "Your tasks:\n"
            "1. RESOLVE COREFERENCE: If the query uses pronouns like 'it', 'they', or 'that project', rewrite it to be standalone using entities from history.\n"
            "2. DECOMPOSE: If the query is complex (asks for comparisons or multiple facts), break it into 1-3 independent sub-queries.\n"
            "3. SELECT TOOLS: Choose the best tools from the list above for EACH sub-query (merged set).\n"
            "4. EXTRACT KEYWORDS: Identify primary entities and concepts for keyword-based search.\n"
            "5. CLASSIFY: Determine if it's LOCAL (facts), GLOBAL (themes), HYBRID, or GENERIC (short/conversational).\n\n"
            "OUTPUT FORMAT: You MUST return valid JSON ONLY:\n"
            "{\n"
            "  \"resolved_query\": \"Standalone rewritten query\",\n"
            "  \"sub_queries\": [\"sub-query 1\", \"sub-query 2\"],\n"
            "  \"tools\": [\"tool1\", \"tool2\"],\n"
            "  \"keywords\": [\"keyword1\", \"keyword2\"],\n"
            "  \"fallback_type\": \"LOCAL|GLOBAL|HYBRID|GENERIC\",\n"
            "  \"is_generic\": true|false,\n"
            "  \"rationale\": \"Brief explanation\"\n"
            "}"
        )

        try:
            print(f"\n  [Orchestrator] Analyzing request: {query[:50]}...")
            response = await self.llm.acomplete(prompt)
            raw_text = response.text.strip()
            
            # Extract JSON
            start = raw_text.find("{")
            end = raw_text.rfind("}") + 1
            if start != -1 and end > start:
                parsed = json.loads(raw_text[start:end])
                
                resolved = parsed.get("resolved_query", query)
                sub_queries = parsed.get("sub_queries", [resolved])
                tools = parsed.get("tools", ["vector_search"])
                keywords = parsed.get("keywords", [])
                is_generic = parsed.get("is_generic", False)
                rationale = parsed.get("rationale", "")
                
                # Validation
                valid_tools = [t for t in tools if any(at["name"] == t for at in self.AVAILABLE_TOOLS)]
                if not valid_tools and not is_generic:
                    valid_tools = ["vector_search", "graph_search"]
                
                qtype_str = parsed.get("fallback_type", "UNKNOWN").upper()
                try:
                    fallback_type = QueryType(qtype_str)
                except ValueError:
                    fallback_type = QueryType.UNKNOWN

                return ToolPlan(
                    tools=valid_tools,
                    rationale=rationale,
                    fallback_query_type=fallback_type,
                    is_generic=is_generic,
                    sub_queries=sub_queries,
                    keywords=keywords,
                    resolved_query=resolved
                )
        except Exception as e:
            logger.error(f"[Orchestrator] Consolidated analysis failed: {e}")
        
        # Fallback to hard heuristics
        legacy_plan = self._hard_fallback_plan(query)
        legacy_plan.resolved_query = query
        legacy_plan.sub_queries = [query]
        return legacy_plan

    async def plan_tools(self, query: str) -> ToolPlan:
        """Legacy method for targeted tool selection."""
        # Wrap the new analysis but without history for backward compatibility if needed
        return await self.analyze_request(query)

    def _hard_fallback_plan(self, query: str) -> ToolPlan:
        # Very basic heuristic fallback if LLM fails completely
        query_lower = query.lower().strip()
        
        # 1. Very short or conversational queries
        conversational_words = ["hi", "hello", "hey", "howdy", "what's up", "who are you", "what can you do"]
        if len(query_lower.split()) <= 2 or any(query_lower.startswith(w) for w in conversational_words):
             return ToolPlan(
                tools=["community_search", "summary_search"],
                rationale="Prompt is too short or conversational. Falling back to global summaries of the KB.",
                fallback_query_type=QueryType.GENERIC,
                is_generic=True
            )

        # 2. Global pattern matching
        global_keywords = [
            "main theme", "overall", "summarize", "summary", "overview",
            "big picture", "across all", "general", "common pattern",
            "recurring", "high level", "key takeaway", "in general"
        ]
        
        if any(kw in query_lower for kw in global_keywords):
            return ToolPlan(
                tools=["summary_search", "community_search", "vector_search"],
                rationale="Fallback to global pattern matching.",
                fallback_query_type=QueryType.GLOBAL,
                is_generic=False
            )
        else:
            return ToolPlan(
                tools=["vector_search", "graph_search"],
                rationale="Fallback to local pattern matching.",
                fallback_query_type=QueryType.LOCAL,
                is_generic=False
            )
