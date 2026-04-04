"""
ReflectionService — Self-RAG / CRAG Evaluator
==============================================
Provides three agentic grading steps that are inserted into the retrieval
pipeline to make it self-correcting:

  1. grade_retrieval()  — Is the retrieved context relevant to the query?
  2. rewrite_query()    — Rewrite a failing query to be more effective.
  3. grade_answer()     — Is the generated answer grounded in the context
                          (hallucination check)?

All methods use the shared Ollama LLM instance and emit structured JSON so
that the engine can branch on the grades without string heuristics.
"""

import json
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ── Grade result dataclasses ───────────────────────────────────────────────────

@dataclass
class RetrievalGrade:
    relevant: bool
    reason: str
    raw: str = field(default="", repr=False)


@dataclass
class AnswerGrade:
    grounded: bool
    reason: str
    raw: str = field(default="", repr=False)


# ── Service ────────────────────────────────────────────────────────────────────

class ReflectionService:
    """
    Agentic self-reflection layer for a RAG pipeline.

    Parameters
    ----------
    llm:
        Any llama-index compatible LLM that exposes .complete(prompt) → response.
    fail_open:
        When True (default), an unparseable LLM response is treated as a PASS
        so that grading failures never silently block valid answers.
    """

    def __init__(self, llm, fail_open: bool = True):
        self.llm = llm
        self.fail_open = fail_open

    # ── Public API ─────────────────────────────────────────────────────────────

    def grade_retrieval(self, query: str, context_chunks: list[str]) -> RetrievalGrade:
        """
        Evaluate whether a list of retrieved context chunks contains sufficient
        information to answer the query.

        Returns a RetrievalGrade with relevant=True if the context is adequate,
        or relevant=False with an explanatory reason if the retrieval failed.
        """
        if not context_chunks:
            return RetrievalGrade(
                relevant=False,
                reason="No context was retrieved for this query.",
            )

        # Cap context to avoid blowing the LLM context window
        combined = "\n---\n".join(c[:600] for c in context_chunks[:6])

        prompt = (
            "You are an expert retrieval evaluator for a RAG system.\n"
            "Your job is to decide whether the retrieved context chunks contain\n"
            "ENOUGH RELEVANT INFORMATION to answer the user's query.\n\n"
            "Guidelines:\n"
            "- 'relevant: true'  → the context directly addresses the query topic and\n"
            "                      a reasonable answer can be constructed from it.\n"
            "- 'relevant: false' → the context is off-topic, too vague, or too sparse\n"
            "                      to produce a useful answer.\n\n"
            "You MUST respond with valid JSON and nothing else:\n"
            "{\"relevant\": <true|false>, \"reason\": \"<one concise sentence>\"}\n\n"
            f"Query: {query}\n\n"
            f"Retrieved Context:\n{combined}"
        )

        return self._parse_retrieval_grade(self._call_llm(prompt))

    def rewrite_query(self, original_query: str, failure_reason: str) -> str:
        """
        Produce an improved version of a query that failed the retrieval grade.

        The rewritten query should be more specific, use different vocabulary,
        or decompose a compound question so that vector/BM25 retrieval is more
        likely to surface relevant chunks.
        """
        prompt = (
            "You are a query optimisation expert for a RAG system.\n"
            "A semantic retrieval step has failed for the following reasons:\n\n"
            f"Original Query : {original_query}\n"
            f"Failure Reason : {failure_reason}\n\n"
            "Rewrite the original query to maximise the likelihood of retrieving\n"
            "relevant documents. Apply one or more of these techniques:\n"
            "  - Decompose compound questions into the most specific sub-question.\n"
            "  - Replace pronouns / ambiguous terms with explicit entity names.\n"
            "  - Add domain-relevant synonyms or alternative phrasings.\n"
            "  - Shift from a question form to a descriptive statement if helpful.\n\n"
            "Return ONLY the rewritten query as plain text. No preamble, no quotes."
        )

        rewritten = self._call_llm(prompt).strip().strip('"\'')
        if not rewritten or len(rewritten) < 5:
            logger.warning("[Reflection] Query rewrite produced empty output; using original.")
            return original_query

        logger.info(f"[Reflection] Query rewritten: '{original_query}' → '{rewritten}'")
        return rewritten

    def grade_answer(self, query: str, context_chunks: list[str], answer: str) -> AnswerGrade:
        """
        Check whether the generated answer is grounded in the retrieved context
        (hallucination detection).

        Returns an AnswerGrade with grounded=True if all key claims in the
        answer can be traced back to the context, or grounded=False otherwise.
        """
        if not answer.strip():
            return AnswerGrade(grounded=False, reason="The answer is empty.")

        combined = "\n---\n".join(c[:500] for c in context_chunks[:5])

        prompt = (
            "You are a factual grounding evaluator for a RAG system.\n"
            "Assess whether the AI Answer below is FULLY SUPPORTED by the\n"
            "Retrieved Context and does NOT introduce unsupported claims.\n\n"
            "Guidelines:\n"
            "- 'grounded: true'  → every key claim in the answer can be traced\n"
            "                      back to the provided context.\n"
            "- 'grounded: false' → the answer contains invented facts, over-generalises,\n"
            "                      or makes claims not present in the context.\n\n"
            "You MUST respond with valid JSON and nothing else:\n"
            "{\"grounded\": <true|false>, \"reason\": \"<one concise sentence>\"}\n\n"
            f"Query: {query}\n\n"
            f"Retrieved Context:\n{combined}\n\n"
            f"AI Answer:\n{answer}"
        )

        return self._parse_answer_grade(self._call_llm(prompt))

    # ── Private helpers ────────────────────────────────────────────────────────

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM and return the raw text response."""
        try:
            response = self.llm.complete(prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"[Reflection] LLM call failed: {e}")
            return ""

    def _parse_json(self, raw: str) -> dict | None:
        """Robustly extract a JSON object from LLM output."""
        text = raw.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        # Find first { … } block
        start = text.find("{")
        end = text.rfind("}") + 1
        if start == -1 or end <= start:
            return None
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            return None

    def _parse_retrieval_grade(self, raw: str) -> RetrievalGrade:
        parsed = self._parse_json(raw)
        if parsed is None:
            logger.warning(f"[Reflection] Could not parse retrieval grade JSON; fail_open={self.fail_open}. Raw: {raw[:120]}")
            return RetrievalGrade(
                relevant=self.fail_open,
                reason="Could not parse grading response — defaulting to pass." if self.fail_open
                       else "Could not parse grading response — defaulting to fail.",
                raw=raw,
            )
        return RetrievalGrade(
            relevant=bool(parsed.get("relevant", self.fail_open)),
            reason=parsed.get("reason", ""),
            raw=raw,
        )

    def _parse_answer_grade(self, raw: str) -> AnswerGrade:
        parsed = self._parse_json(raw)
        if parsed is None:
            logger.warning(f"[Reflection] Could not parse answer grade JSON; fail_open={self.fail_open}. Raw: {raw[:120]}")
            return AnswerGrade(
                grounded=self.fail_open,
                reason="Could not parse grading response — defaulting to pass." if self.fail_open
                       else "Could not parse grading response — defaulting to fail.",
                raw=raw,
            )
        return AnswerGrade(
            grounded=bool(parsed.get("grounded", self.fail_open)),
            reason=parsed.get("reason", ""),
            raw=raw,
        )
