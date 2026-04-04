import json
from typing import List, Dict, Any, Optional
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from config import setup_models
from .prompts import (
    PLANNING_PROMPT, 
    REFINEMENT_PROMPT, 
    SYNTHESIS_PROMPT, 
    CONVERSATIONAL_QA_PROMPT, 
    ANALYSIS_PROMPT, 
    TERMINOLOGY_PROMPT
)

class ResearchPlanner:
    def __init__(self):
        # Ensure environment is set up
        setup_models()
        self.llm = Settings.llm

    def generate_plan(self, topic: str, mode: str = "arxiv") -> Dict:
        """
        Generate a research objective and a list of search queries based on the mode.
        """
        if mode == "deep":
            mode_instruction = (
                "Generate a comprehensive mix of queries. "
                "For each query, assign it to the most appropriate backend: 'arxiv' or 'web'."
            )
        else:
            mode_instruction = {
                "arxiv": "Professional-grade ArXiv search strings for academic papers.",
                "web": "Optimized for general web search for latest articles and reports.",
                "news": "Focus on latest news, company announcements, and recent events.",
            }.get(mode, "General search queries.")

        prompt = PLANNING_PROMPT.format(
            topic=topic,
            mode=mode.upper(),
            mode_instruction=mode_instruction
        )

        response = self.llm.complete(prompt)
        try:
            content = self._parse_json(str(response))
            # Normalise: ensure queries is always a list
            if not isinstance(content.get("queries"), list):
                raise ValueError("queries must be a list")
            return content
        except Exception:
            return self._fallback_plan(topic, mode)

    def refine_plan(self, topic: str, current_plan: Dict, feedback: str) -> Dict:
        """
        Update the research plan based on user feedback.
        """
        objective = current_plan.get("objective", "Unknown")
        queries = json.dumps(current_plan.get("queries", []), indent=2)

        prompt = REFINEMENT_PROMPT.format(
            topic=topic,
            objective=objective,
            queries=queries,
            feedback=feedback
        )

        response = self.llm.complete(prompt)
        try:
            return self._parse_json(str(response))
        except Exception:
            # If refinement fails, return the original plan
            return current_plan

    def synthesize_results(self, topic: str, results: List[Dict]) -> str:
        """
        Synthesize a research report from multiple search results.
        """
        if not results:
            return "No research results found to synthesize."

        context_items = []
        for r in results[:15]:  # Use top 15 results for synthesis
            source = r.get("source", "web").upper()
            title = r.get("title", "No Title")
            content = r.get("snippet", r.get("summary", ""))
            context_items.append(f"[{source}] {title}\n{content}")

        results_context = "\n---\n".join(context_items)
        prompt = SYNTHESIS_PROMPT.format(topic=topic, results_context=results_context)

        response = self.llm.complete(prompt)
        return str(response).strip()

    def chat_with_results(self, topic: str, question: str, results: List[Dict]) -> str:
        """
        Answer questions about the research findings using the context of search results.
        """
        context_items = []
        for r in results[:10]:
            title = r.get("title", "No Title")
            content = r.get("snippet", r.get("summary", ""))
            context_items.append(f"{title}: {content}")

        results_context = "\n".join(context_items)
        prompt = CONVERSATIONAL_QA_PROMPT.format(
            topic=topic,
            results_context=results_context,
            question=question
        )

        response = self.llm.complete(prompt)
        return str(response).strip()

    def analyze_result(self, topic: str, result: Dict) -> str:
        """
        Provide a deeper analysis of a single search result.
        """
        title = result.get("title", "No Title")
        source = result.get("source", "web").capitalize()
        content = result.get("snippet", result.get("summary", ""))

        prompt = ANALYSIS_PROMPT.format(
            topic=topic,
            title=title,
            source=source,
            content=content
        )

        response = self.llm.complete(prompt)
        return str(response).strip()

    def discover_terms(self, results: List[Dict]) -> List[str]:
        """
        Analyze results to find key technical terms/jargon.
        """
        snippets = []
        for r in results[:10]:
            text = f"Title: {r['title']}\nSnippet: {r.get('snippet', r.get('summary', ''))}"
            snippets.append(text)

        context = "\n---\n".join(snippets)
        prompt = TERMINOLOGY_PROMPT.format(context=context)

        try:
            response = self.llm.complete(prompt)
            terms = self._parse_json(str(response))
            return [t for t in terms if isinstance(t, str)]
        except Exception:
            return []

    def _parse_json(self, content: str) -> Any:
        content = content.strip()
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        return json.loads(content)

    def _fallback_plan(self, topic: str, mode: str) -> Dict:
        if mode == "deep":
            fallback_queries = [
                {"query": f"{topic} research survey", "backend": "arxiv"},
                {"query": f"{topic} latest developments 2024", "backend": "web"},
            ]
        else:
            fallback_queries = [topic, f"latest news on {topic}"]
        return {
            "objective": f"Researching {topic} ({mode})",
            "queries": fallback_queries,
        }
