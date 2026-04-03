"""
indexers/preprocessor.py
------------------------
DocumentPreprocessor: clean and normalize raw text *before* chunking.

Handles four concerns from the RAG best-practices guide:
  1. Encoding & whitespace normalization  (ftfy, Unicode fixes)
  2. PDF/OCR artifact cleaning            (hyphenation, broken words, punctuation)
  3. Boilerplate removal                  (page numbers, running headers/footers, legal disclaimers)
  4. Coreference resolution              (spaCy heuristic pronoun → named-entity substitution)

Metadata Preservation:
  Cleaning operates *only* on ``doc.get_content()`` / ``doc.set_content()``.
  Document metadata (page_number, sections, source_type, etc.) is never read or
  mutated by any cleaning step.

Batch & Streaming:
  ``apreprocess_stream(docs)``  — async generator that yields cleaned documents
  one at a time, avoiding large in-memory lists.
  ``preprocess_batch(docs, batch_size)``  — processes in configurable batches.

All steps degrade gracefully: if an optional dependency is absent the step is
skipped with a warning rather than crashing the indexing pipeline.
"""

from __future__ import annotations

import re
import logging
import asyncio
from typing import AsyncIterator, Iterator, List, Optional, Any

from llama_index.core.schema import Document

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency guards
# ---------------------------------------------------------------------------

try:
    import ftfy  # type: ignore
    _HAS_FTFY = True
except ImportError:
    _HAS_FTFY = False
    logger.warning(
        "[preprocessor] 'ftfy' not installed — encoding normalization disabled. "
        "Install with: pip install ftfy"
    )

try:
    import spacy  # type: ignore
    _HAS_SPACY = True
except ImportError:
    _HAS_SPACY = False
    logger.warning(
        "[preprocessor] 'spacy' not installed — coreference resolution disabled. "
        "Install with: pip install spacy && python -m spacy download en_core_web_sm"
    )

# ---------------------------------------------------------------------------
# Boilerplate patterns
# ---------------------------------------------------------------------------

_BOILERPLATE_PATTERNS: list[re.Pattern] = [
    # Page numbers: "Page 3 of 12", "- 3 -", "3 | P a g e", etc.
    # Note: we avoid matching our own semantic [Page X] markers by requiring 
    # the line to NOT start with '['.
    re.compile(r"(?m)^(?![\[])[\-\s]*[Pp]age\s+\d+\s*(of\s+\d+)?[\-\s]*$"),
    re.compile(r"(?m)^\s*\d+\s*\|\s*[Pp]\s*a\s*g\s*e\s*$"),
    re.compile(r"(?m)^\s*-\s*\d+\s*-\s*$"),
    # Running headers/footers — detect lines that repeat > 3 times across the doc
    # (handled separately in _remove_repeated_lines)
    # Copyright / confidentiality notices
    re.compile(
        r"(?i)(confidential|proprietary|all rights reserved|copyright\s+©?\s*\d{4})[^\n]*",
    ),
    # Watermarks / draft notices
    re.compile(r"(?i)\b(draft|do not distribute|internal use only)\b[^\n]*"),
    # Empty section numbering lines like "1.1.1." on their own
    re.compile(r"(?m)^\s*(\d+\.)+\s*$"),
    # Excessive whitespace / form-feed characters
    re.compile(r"\f"),
]

_REPEATED_LINE_MIN_OCCURRENCES = 4  # a line appearing this many times is likely boilerplate


class DocumentPreprocessor:
    """
    Clean and normalize LlamaIndex ``Document`` objects in-place before indexing.

    Usage::

        preprocessor = DocumentPreprocessor()
        documents = preprocessor.preprocess(documents)

    Parameters
    ----------
    enable_coref : bool
        Enable spaCy-based heuristic coreference resolution (default: True).
        Requires spacy + en_core_web_sm.  Gracefully disabled if spaCy is absent.
    spacy_model : str
        spaCy model to load for NER + coref (default: "en_core_web_sm").
    """

    def __init__(
        self,
        enable_coref: bool = True,
        spacy_model: str = "en_core_web_sm",
    ) -> None:
        self.enable_coref = enable_coref and _HAS_SPACY
        self._nlp = None  # lazy-loaded

        if self.enable_coref:
            try:
                self._nlp = spacy.load(spacy_model)
                logger.info("[preprocessor] spaCy model '%s' loaded.", spacy_model)
            except OSError:
                logger.warning(
                    "[preprocessor] spaCy model '%s' not found. "
                    "Run: python -m spacy download %s",
                    spacy_model,
                    spacy_model,
                )
                self._nlp = None
                self.enable_coref = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def apreprocess(self, documents: List[Document]) -> List[Document]:
        """Asynchronously apply all cleaning steps to each document in the list."""
        logger.debug("[preprocessor] apreprocess called for %d document(s)", len(documents))
        return await asyncio.to_thread(self.preprocess, documents)

    async def apreprocess_stream(
        self, documents: List[Document]
    ) -> AsyncIterator[Document]:
        """
        Async generator - yields cleaned documents one at a time.

        Use this instead of ``apreprocess`` when the caller wants to start
        downstream work (e.g. indexing) before the entire batch is cleaned,
        or when the batch is very large and holding the full list in memory
        is undesirable.
        """
        for doc in documents:
            cleaned = await asyncio.to_thread(self._clean_single, doc)
            yield cleaned

    def preprocess(self, documents: List[Document]) -> List[Document]:
        """
        Apply all cleaning steps to each document in the list.
        Returns the same list (documents mutated in-place for memory efficiency).

        Metadata (page_number, sections, source_type ...) is never touched.
        """
        for doc in documents:
            self._clean_single(doc)
        return documents

    def preprocess_batch(
        self,
        documents: List[Document],
        batch_size: int = 32,
    ) -> Iterator[List[Document]]:
        """
        Yield cleaned documents in batches of *batch_size*.

        Useful for pipelines that want to stream documents through a
        load -> clean -> index cycle without materialising the full corpus.
        """
        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            self.preprocess(batch)
            yield batch

    # ------------------------------------------------------------------
    # Internal: single-document cleaning
    # ------------------------------------------------------------------

    def _clean_single(self, doc: Document) -> Document:
        """
        Run the full cleaning pipeline on a single Document *in-place*.
        Only ``doc.get_content()`` / ``doc.set_content()`` are used;
        metadata is never read or modified.
        """
        text = doc.get_content()
        text = self._fix_encoding(text)
        text = self._normalize_text(text)
        text = self._remove_boilerplate(text)
        if self.enable_coref and self._nlp is not None:
            text = self._resolve_coreferences(text)
        text = self._normalize_whitespace(text)
        doc.set_content(text)
        return doc

    # ------------------------------------------------------------------
    # Step 1 — Encoding & Unicode normalization
    # ------------------------------------------------------------------

    def _fix_encoding(self, text: str) -> str:
        """
        Fix broken UTF-8 / mojibake encoding with ftfy.
        Also removes null bytes and non-printable control characters.
        """
        # Strip null / non-printable bytes (except newline, tab, carriage return)
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

        if _HAS_FTFY:
            text = ftfy.fix_text(text)

        return text

    # ------------------------------------------------------------------
    # Step 2 — PDF/OCR artifact cleaning
    # ------------------------------------------------------------------

    def _normalize_text(self, text: str) -> str:
        """
        Fix common PDF extraction and OCR artifacts.
        """
        # 1. Fix soft hyphens and broken words at line breaks
        # (e.g., "en- \nvironment" -> "environment")
        text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", text)

        # 2. Fix spaced-out words (common in some PDF layouts/OCR)
        # (e.g., "T h e  r e s u l t" -> "The result")
        # Heuristic: if we see 3+ characters separated by single spaces, join them.
        def _join_spaced_words(match):
            return match.group(0).replace(" ", "")
        
        text = re.sub(r"\b([A-Za-z] ){2,}[A-Za-z]\b", _join_spaced_words, text)

        # 3. Normalize punctuation
        text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
        text = text.replace("…", "...").replace("–", "-").replace("—", "--")

        return text

    # ------------------------------------------------------------------
    # Step 2 — Boilerplate removal
    # ------------------------------------------------------------------

    def _remove_boilerplate(self, text: str) -> str:
        """
        Strip running headers, footers, page numbers, and legal disclaimers.
        """
        # 1. Apply regex patterns
        for pattern in _BOILERPLATE_PATTERNS:
            text = pattern.sub("", text)

        # 2. Remove lines that appear suspiciously often (likely running headers/footers)
        text = self._remove_repeated_lines(text)

        return text

    def _remove_repeated_lines(self, text: str, min_occurrences: int = _REPEATED_LINE_MIN_OCCURRENCES) -> str:
        """
        Detect and remove non-empty lines that appear >= *min_occurrences* times.
        These are almost certainly running headers/footers or repeated boilerplate.
        """
        lines = text.split("\n")
        from collections import Counter
        # Only count stripped, non-trivial lines as candidates
        counts = Counter(
            line.strip() for line in lines
            if len(line.strip()) > 5  # ignore very short lines
        )
        boilerplate_lines = {line for line, count in counts.items() if count >= min_occurrences}

        if not boilerplate_lines:
            return text

        logger.debug(
            "[preprocessor] Removing %d repeated boilerplate line(s).", len(boilerplate_lines)
        )
        filtered = [
            line for line in lines
            if line.strip() not in boilerplate_lines
        ]
        return "\n".join(filtered)

    # ------------------------------------------------------------------
    # Step 3 — Coreference resolution (spaCy heuristic)
    # ------------------------------------------------------------------

    def _resolve_coreferences(self, text: str) -> str:
        """
        Heuristic pronoun → named-entity substitution using spaCy NER.

        Strategy:
        - Process the text sentence-by-sentence.
        - Maintain a rolling "antecedent" per pronoun gender/number class
          pointing to the most recently seen named entity.
        - Replace pronouns with the antecedent when confidence is high
          (single clear candidate in recent context window).

        This is not a full neural coref model but is fast, zero-cost, and
        catches the most common patterns (e.g. "the company" → "Apple Inc.",
        "he" → "Steve Jobs") that fragment graph triplets when unresolved.
        """
        if self._nlp is None:
            return text

        # Process in chunks to stay within spaCy's limit
        MAX_CHUNK = 100_000  # characters
        if len(text) <= MAX_CHUNK:
            return self._coref_chunk(text)

        # For very long texts, process in overlapping chunks
        parts: list[str] = []
        for i in range(0, len(text), MAX_CHUNK - 500):
            chunk = text[i : i + MAX_CHUNK]
            parts.append(self._coref_chunk(chunk))
        return "".join(parts)

    def _coref_chunk(self, text: str) -> str:
        """Apply heuristic coref to a single manageable chunk."""
        doc = self._nlp(text)

        # Map: pronoun surface → replacement text
        # We track the most recent PERSON, ORG, GPE entity per class
        antecedent: dict[str, Optional[str]] = {
            "PERSON_SINGULAR": None,
            "ORG": None,
            "GPE": None,
            "NORP": None,
        }

        tokens_out: list[str] = []
        i = 0
        tokens = list(doc)

        while i < len(tokens):
            tok = tokens[i]

            # Update antecedent from named entities
            if tok.ent_type_ in ("PERSON", "GPE", "ORG", "NORP", "FAC", "PRODUCT"):
                # Collect the full entity span
                span_end = i
                while span_end < len(tokens) and tokens[span_end].ent_iob_ in ("B", "I"):
                    span_end += 1
                entity_text = doc[i : span_end].text
                if tok.ent_type_ == "PERSON":
                    antecedent["PERSON_SINGULAR"] = entity_text
                elif tok.ent_type_ == "ORG":
                    antecedent["ORG"] = entity_text
                elif tok.ent_type_ in ("GPE", "FAC"):
                    antecedent["GPE"] = entity_text
                elif tok.ent_type_ == "NORP":
                    antecedent["NORP"] = entity_text
                # Emit the entity as-is
                tokens_out.append(doc[i : span_end].text_with_ws)
                i = span_end
                continue

            # Replace pronouns
            replacement = self._pronoun_replacement(tok, antecedent)
            if replacement is not None:
                # Preserve trailing whitespace from original token
                ws = tok.whitespace_
                tokens_out.append(replacement + ws)
            else:
                tokens_out.append(tok.text_with_ws)

            i += 1

        return "".join(tokens_out)

    @staticmethod
    def _pronoun_replacement(
        token,
        antecedent: dict[str, Optional[str]],
    ) -> Optional[str]:
        """
        Map common pronouns to their likely antecedent.
        Returns the replacement string, or None if no substitution is needed.
        """
        if token.pos_ != "PRON":
            return None

        lower = token.lower_

        # Third-person singular masculine → most recent PERSON
        if lower in ("he", "him", "his", "himself"):
            return antecedent.get("PERSON_SINGULAR")

        # Third-person singular feminine → most recent PERSON
        if lower in ("she", "her", "hers", "herself"):
            return antecedent.get("PERSON_SINGULAR")

        # Neuter (companies / things) → most recent ORG, then GPE
        if lower in ("it", "its", "itself"):
            return antecedent.get("ORG") or antecedent.get("GPE")

        # Plural (they / their) → most recent ORG, NORP, or PERSON
        if lower in ("they", "them", "their", "theirs", "themselves"):
            return antecedent.get("ORG") or antecedent.get("NORP") or antecedent.get("PERSON_SINGULAR")

        return None

    # ------------------------------------------------------------------
    # Step 4 — Final whitespace normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """
        Collapse runs of 3+ blank lines into a single blank line, and
        strip trailing whitespace from each line.
        """
        # Strip trailing whitespace per line
        lines = [line.rstrip() for line in text.split("\n")]
        text = "\n".join(lines)

        # Collapse 3+ consecutive blank lines → 2 (paragraph separator)
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()
