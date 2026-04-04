import os
import json
from typing import List, Dict
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from config import setup_models

class ResearchPlanner:
    def __init__(self):
        # Ensure environment is set up
        setup_models()
        self.llm = Settings.llm

    def generate_plan(self, topic: str, mode: str = "arxiv") -> Dict:
        """
        Generate a research objective and a list of search queries based on the mode.
        For 'deep' mode, each query is labelled with a 'backend' field ('arxiv' or 'web').
        The LLM decides how many queries go to each backend.
        """
        if mode == "deep":
            mode_instruction = (
                "Generate a comprehensive mix of queries. "
                "For each query, assign it to the most appropriate backend: "
                "'arxiv' for academic/technical/research questions, "
                "'web' for recent news, industry use-cases, tutorials, or context. "
                "You decide how many go to each backend based on the topic."
            )
            query_format = (
                '"queries": [\n'
                '    {"query": "...", "backend": "arxiv"},\n'
                '    {"query": "...", "backend": "web"}\n'
                ']'
            )
        else:
            mode_instruction = {
                "arxiv": "The queries should be professional-grade ArXiv search strings for academic papers.",
                "web": "The queries should be optimized for a general web search to find the latest articles, reports, and context.",
                "news": "The queries should focus on latest news updates, company announcements, press releases, and recent events.",
            }.get(mode, "General search queries.")
            query_format = (
                '"queries": [\n'
                '    "query 1",\n'
                '    "query 2",\n'
                '    "query 3"\n'
                ']'
            )

        prompt = f"""
        You are a research orchestration agent. Your goal is to help a user research the following topic: "{topic}".
        Current search mode: {mode.upper()}
        {mode_instruction}

        Please provide your response in the following JSON format:
        {{
            "objective": "A brief summary of the research goal.",
            {query_format}
        }}

        Ensure you only return valid JSON. No other text.
        """

        response = self.llm.complete(prompt)
        try:
            content = str(response).strip()
            # Handle potential markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            plan = json.loads(content)

            # Normalise: ensure queries is always a list (string or dict)
            if not isinstance(plan.get("queries"), list):
                raise ValueError("queries must be a list")

            return plan
        except Exception:
            # Fallback if Ollama fails to return valid JSON
            if mode == "deep":
                fallback_queries = [
                    {"query": f"{topic} research survey", "backend": "arxiv"},
                    {"query": f"{topic} recent advances", "backend": "arxiv"},
                    {"query": f"{topic} latest developments 2024", "backend": "web"},
                    {"query": f"{topic} industry applications", "backend": "web"},
                ]
            else:
                fallback_queries = [
                    f"{topic}",
                    f"latest news on {topic}",
                    f"{topic} recent developments",
                ]
            return {
                "objective": f"Researching {topic} ({mode})",
                "queries": fallback_queries,
            }

    def discover_terms(self, results: List[Dict]) -> List[str]:
        """
        Analyze search results to find key technical terms or jargon that may need definition.
        Only called for web, news, and deep modes.
        """
        snippets = []
        for r in results[:10]:  # Limit to first 10 results for context
            text = f"Title: {r['title']}\nSnippet: {r.get('snippet', r.get('summary', ''))}"
            snippets.append(text)

        context = "\n---\n".join(snippets)

        prompt = f"""
        Analyze the following research snippets and identify 3-5 technical terms, acronyms, or complex jargon that are central to the topic but might need a clear definition for a non-expert.

        Snippets:
        {context}

        Return ONLY a JSON list of strings.
        Example: ["RAG", "Vector Database", "Cosine Similarity"]
        """

        try:
            response = self.llm.complete(prompt)
            content = str(response).strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            terms = json.loads(content)
            return [t for t in terms if isinstance(t, str)]
        except Exception:
            return []
