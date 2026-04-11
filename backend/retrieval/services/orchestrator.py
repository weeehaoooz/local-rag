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

class ToolOrchestrator:
    """
    LLM-driven tool orchestrator for the Hybrid RAG engine.
    Given a query, it selects the appropriate combination of retrieval tools.
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
        
    async def plan_tools(self, query: str) -> ToolPlan:
        """
        Produce a plan of tools to use to answer the query.
        """
        tools_desc = "\n".join([f"- {t['name']}: {t['description']}" for t in self.AVAILABLE_TOOLS])
        
        prompt = (
            "You are an intelligent RAG query router and tool orchestrator.\n"
            "Your job is to select the most appropriate combination of retrieval tools to gather context for the user's query.\n\n"
            f"Available Tools:\n{tools_desc}\n\n"
            "Guidelines:\n"
            "- You can select multiple tools if necessary (e.g., passing 2-3 tools in the list).\n"
            "- Favor 'vector_search' and 'graph_search' for standard localized questions.\n"
            "- DO NOT use 'web_search' or 'arxiv_search' unless explicitly implied by the query (e.g. 'latest news', 'search the web', 'academic papers').\n"
            "- Identify if the query is too generic, conversational, or lacks context for a meaningful search (e.g. 'hi', 'how are you', 'tell me something', 'explain it').\n"
            "- If a query is generic or broad but relates to the knowledge base (e.g. 'tell me something', 'what is in here?'), favor 'community_search' or 'summary_search' instead of 'vector_search'.\n"
            "- Determine a fallback generic query type: 'LOCAL', 'GLOBAL', 'HYBRID', or 'GENERIC'.\n\n"
            "You MUST respond with valid JSON and nothing else:\n"
            "{\n"
            "  \"tools\": [\"tool_name_1\", \"tool_name_2\"],\n"
            "  \"is_generic\": true|false,\n"
            "  \"rationale\": \"Brief explanation of your selection\",\n"
            "  \"fallback_type\": \"LOCAL|GLOBAL|HYBRID|GENERIC\"\n"
            "}\n\n"
            f"User Query: {query}"
        )

        try:
            print(f"\n  [Orchestrator] Planning tools for: {query[:50]}...")
            response = await self.llm.acomplete(prompt)
            raw_text = response.text.strip()
            
            # Clean possible markdown formatting
            if raw_text.startswith("```"):
                raw_text = raw_text.split("```")[1]
                if raw_text.startswith("json"):
                    raw_text = raw_text[4:]
                raw_text = raw_text.strip()
                
            start = raw_text.find("{")
            end = raw_text.rfind("}") + 1
            if start != -1 and end > start:
                parsed = json.loads(raw_text[start:end])
                
                tools = parsed.get("tools", [])
                rationale = parsed.get("rationale", "")
                is_generic = parsed.get("is_generic", False)
                
                # Filter to only valid tools
                valid_tools = [t for t in tools if any(at["name"] == t for at in self.AVAILABLE_TOOLS)]
                
                # Fallback Type
                qtype_str = parsed.get("fallback_type", "UNKNOWN").upper()
                try:
                    fallback_type = QueryType(qtype_str)
                except ValueError:
                    fallback_type = QueryType.UNKNOWN
                    
                if not valid_tools and not is_generic:
                    # Give it a sane default if the list was empty AND it's not marked as generic
                    valid_tools = ["vector_search", "graph_search"]
                
                print(f"  [Orchestrator] Selected tools: {valid_tools} | Generic: {is_generic}")
                print(f"  [Orchestrator] Rationale: {rationale}")
                    
                return ToolPlan(
                    tools=valid_tools,
                    rationale=rationale,
                    fallback_query_type=fallback_type,
                    is_generic=is_generic
                )
                
        except Exception as e:
            logger.error(f"[Orchestrator] Failed to parse tool plan: {str(e)}")
            print(f"  [Orchestrator] Planning failed ({e}). Using hard fallback.")
            
        # Hard Fallback
        return self._hard_fallback_plan(query)

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
