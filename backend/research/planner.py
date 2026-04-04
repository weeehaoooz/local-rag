import os
import json
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from config import setup_models

class ResearchPlanner:
    def __init__(self):
        # Ensure environment is set up
        setup_models()
        self.llm = Settings.llm

    def generate_plan(self, topic: str, mode: str = "arxiv"):
        """
        Generate a research objective and a list of search queries based on the mode.
        """
        mode_instruction = {
            "arxiv": "The queries should be professional-grade ArXiv search strings for academic papers.",
            "web": "The queries should be optimized for a general web search to find the latest articles, reports, and context.",
            "news": "The queries should focus on latest news updates, company announcements, press releases, and recent events.",
            "deep": "Generate a comprehensive set of queries: 3 for ArXiv (academic) and 3 for general Web search."
        }.get(mode, "General search queries.")

        prompt = f"""
        You are a research orchestration agent. Your goal is to help a user research the following topic: "{topic}".
        Current search mode: {mode.upper()}
        {mode_instruction}
        
        Please provide your response in the following JSON format:
        {{
            "objective": "A brief summary of the research goal.",
            "queries": [
                "query 1",
                "query 2",
                "query 3"
            ]
        }}
        
        Ensure you only return valid JSON. No other text.
        """
        
        response = self.llm.complete(prompt)
        try:
            # Attempt to parse JSON from response
            content = str(response).strip()
            # Handle potential markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            return json.loads(content)
        except Exception as e:
            # Fallback if Ollama fails to return valid JSON
            return {
                "objective": f"Researching {topic} ({mode})",
                "queries": [f"{topic}", f"latest news on {topic}", f"{topic} recent developments"]
            }

    def discover_terms(self, results: List[dict]) -> List[str]:
        """
        Analyze search results to find key technical terms or jargon that may need definition.
        """
        snippets = []
        for r in results[:10]: # Limit to first 10 results for context
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
