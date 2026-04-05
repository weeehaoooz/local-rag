import os
import json
import hashlib
from typing import List, Dict, Optional
from llama_index.core import Settings
from llama_index.core.schema import Document


from config import GUARDRAILS_DIR, SUMMARIES_DIR


GUARDRAILS_DIR = GUARDRAILS_DIR
SUMMARIES_DIR = SUMMARIES_DIR


def _derive_category(file_path: str, data_root: str = "./data") -> str:
    """
    Derive a document category from its path relative to the data root.
    Files directly under data_root get category 'general'.
    Files in subdirectories use the first-level subdirectory name as category.
    """
    rel = os.path.relpath(file_path, data_root)
    parts = rel.split(os.sep)
    if len(parts) <= 1:
        return "general"
    return parts[0].lower().replace(" ", "_")


def _derive_title(file_path: str) -> str:
    """Derive a human-readable title from the filename."""
    basename = os.path.splitext(os.path.basename(file_path))[0]
    return basename.replace("_", " ").replace("-", " ").title()


class GuardrailManager:
    """
    Manages category-specific guardrails for knowledge graph construction.

    Guardrails define relationship conventions (entity types, relationship types,
    extraction rules) so documents of the same category produce a consistent
    knowledge graph structure.

    Summary-first approach
    ----------------------
    Rather than feeding raw document snippets directly to the guardrail-generation
    prompt (which is noisy and often dominated by formatting/boilerplate), the
    manager first asks the LLM to produce a concise semantic summary of each
    document.  These summaries are:

    1. Stored on disk (``generated_summaries/<file_hash>.json``) and reused on
       subsequent runs if the file has not changed.
    2. Used as the input to ``generate_guardrails`` so the schema reflects the
       *meaning* of the content rather than its surface form.
    3. Attached to each document as ``doc.metadata["summary"]`` so that
       ``GraphIndexer`` can inject them into every small chunk's prompt prefix,
       giving the extractor full document context even when looking at a 256-token
       slice.
    """

    def __init__(
        self,
        guardrails_dir: str = GUARDRAILS_DIR,
        summaries_dir: str = SUMMARIES_DIR,
    ):
        self.guardrails_dir = guardrails_dir
        self.summaries_dir = summaries_dir
        os.makedirs(self.guardrails_dir, exist_ok=True)
        os.makedirs(self.summaries_dir, exist_ok=True)
        self._cache: Dict[str, Dict] = {}
        self._summary_cache: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Summary API
    # ------------------------------------------------------------------

    def get_document_summary(self, file_path: str) -> Optional[str]:
        """Return a cached summary for *file_path*, or None if not yet generated."""
        key = self._summary_key(file_path)
        if key in self._summary_cache:
            return self._summary_cache[key]
        path = os.path.join(self.summaries_dir, f"{key}.json")
        if not os.path.exists(path):
            return None
        with open(path, "r") as f:
            data = json.load(f)
            summary = data.get("summary", "")
            self._summary_cache[key] = summary
            return summary

    def generate_document_summary(self, file_path: str, documents: List[Document]) -> str:
        """
        Ask the LLM to produce a concise semantic summary of *documents* (which
        all belong to the same file).  The summary is persisted keyed by a hash
        of the file content so it is automatically invalidated when the file changes.

        Returns the summary string.
        """
        combined_text = "\n\n".join(
            doc.get_content()[:2000] for doc in documents
        )[:6000]  # cap total input to ~6 k chars

        prompt = (
            "You are a document analyst. Produce a concise but thorough semantic summary "
            "of the document below. The summary will be used to guide knowledge graph "
            "extraction, so focus on:\n"
            "- The main subject(s) and domain of the document\n"
            "- Key entities mentioned (people, organisations, products, concepts)\n"
            "- Core relationships and facts described\n"
            "- The document's overall purpose\n\n"
            "Keep the summary under 300 words. Do not include any preamble.\n\n"
            f"--- DOCUMENT ---\n{combined_text}"
        )
        response = Settings.llm.complete(prompt)
        summary = response.text.strip()

        key = self._summary_key(file_path)
        path = os.path.join(self.summaries_dir, f"{key}.json")
        with open(path, "w") as f:
            json.dump({"file_path": file_path, "summary": summary}, f, indent=2)
        self._summary_cache[key] = summary
        return summary

    def ensure_document_summary(
        self, file_path: str, documents: List[Document], force: bool = False
    ) -> str:
        """
        Return an existing summary for *file_path* or generate one if absent.
        Pass ``force=True`` to regenerate even when a cached version exists.
        """
        if not force:
            existing = self.get_document_summary(file_path)
            if existing:
                return existing
        return self.generate_document_summary(file_path, documents)

    async def agenerate_document_summary(self, file_path: str, documents: List[Document]) -> str:
        combined_text = "\n\n".join(
            doc.get_content()[:2000] for doc in documents
        )[:6000]

        prompt = (
            "You are a document analyst. Produce a concise but thorough semantic summary "
            "of the document below. The summary will be used to guide knowledge graph "
            "extraction, so focus on:\n"
            "- The main subject(s) and domain of the document\n"
            "- Key entities mentioned (people, organisations, products, concepts)\n"
            "- Core relationships and facts described\n"
            "- The document's overall purpose\n\n"
            "Keep the summary under 300 words. Do not include any preamble.\n\n"
            f"--- DOCUMENT ---\n{combined_text}"
        )
        response = await Settings.llm.acomplete(prompt)
        summary = response.text.strip()

        key = self._summary_key(file_path)
        path = os.path.join(self.summaries_dir, f"{key}.json")
        with open(path, "w") as f:
            json.dump({"file_path": file_path, "summary": summary}, f, indent=2)
        self._summary_cache[key] = summary
        return summary

    async def aensure_document_summary(
        self, file_path: str, documents: List[Document], force: bool = False
    ) -> str:
        if not force:
            existing = self.get_document_summary(file_path)
            if existing:
                return existing
        return await self.agenerate_document_summary(file_path, documents)

    # ------------------------------------------------------------------
    # Public Guardrail API
    # ------------------------------------------------------------------

    def get_all_categories(self) -> List[str]:
        """Return a list of all existing guardrail categories."""
        categories = []
        for filename in os.listdir(self.guardrails_dir):
            if filename.endswith(".json"):
                categories.append(filename[:-5])
        return categories

    def get_guardrails(self, category: str) -> Optional[Dict]:
        """Return the stored guardrails for *category*, or None."""
        if category in self._cache:
            return self._cache[category]

        path = self._path_for(category)
        if not os.path.exists(path):
            return None
        with open(path, "r") as f:
            gr = json.load(f)
            self._cache[category] = gr
            return gr

    def get_similar_categories(self, target_category: str, threshold: float = 0.3) -> List[str]:
        """
        Find categories that are similar to *target_category* based on
        overlapping business objects or relationship types.
        """
        target_gr = self.get_guardrails(target_category)
        if not target_gr:
            return []

        target_obj_names = {obj["name"].lower() for obj in target_gr.get("business_objects", [])}
        target_rels = {
            (rel if isinstance(rel, str) else rel.get("name", "")).upper()
            for rel in target_gr.get("relationship_types", [])
        }

        if not target_obj_names and not target_rels:
            return []

        similar = []
        for category in self.get_all_categories():
            if category == target_category:
                continue

            other_gr = self.get_guardrails(category)
            if not other_gr:
                continue

            other_obj_names = {obj["name"].lower() for obj in other_gr.get("business_objects", [])}
            other_rels = {
                (rel if isinstance(rel, str) else rel.get("name", "")).upper()
                for rel in other_gr.get("relationship_types", [])
            }

            obj_intersection = target_obj_names.intersection(other_obj_names)
            obj_union = target_obj_names.union(other_obj_names)
            obj_sim = len(obj_intersection) / len(obj_union) if obj_union else 0

            rel_intersection = target_rels.intersection(other_rels)
            rel_union = target_rels.union(other_rels)
            rel_sim = len(rel_intersection) / len(rel_union) if rel_union else 0

            overall_sim = (obj_sim + rel_sim) / 2 if (obj_union or rel_union) else 0

            if overall_sim >= threshold:
                similar.append(category)

        return similar

    def generate_guardrails(
        self,
        category: str,
        documents: List[Document],
        summaries: Optional[List[str]] = None,
    ) -> Dict:
        """
        Use the configured LLM to produce guardrails (entity types, relationship
        types, conventions) for *category*.

        When *summaries* are provided they are used as the primary input — this
        produces significantly better schemas because summaries are dense, clean
        descriptions of document semantics rather than raw potentially-noisy text.

        Falls back to collecting raw text snippets when no summaries are available
        (backward-compatible behaviour).

        Persists the result to disk and returns it.
        """
        if summaries:
            input_text = self._collect_from_summaries(summaries)
            input_label = "document summaries"
        else:
            input_text = self._collect_samples(documents)
            input_label = "document samples"

        prompt = self._build_generation_prompt(category, input_text, input_label)
        response = Settings.llm.complete(prompt)

        guardrails = self._parse_guardrails_response(response.text, category)
        self._save(category, guardrails)
        return guardrails

    def optimize_guardrails(self, category: str) -> Dict:
        """
        Re-evaluate existing guardrails and refine them.  Returns the
        optimised version and persists it.
        """
        current = self.get_guardrails(category)
        if current is None:
            raise ValueError(
                f"No existing guardrails for category '{category}' to optimise."
            )

        prompt = self._build_optimization_prompt(category, current)
        response = Settings.llm.complete(prompt)

        optimized = self._parse_guardrails_response(response.text, category)
        optimized["version"] = current.get("version", 1) + 1
        self._save(category, optimized)
        return optimized

    def is_optimized(self, category: str) -> bool:
        """True when guardrails have been optimized at least once."""
        gr = self.get_guardrails(category)
        return gr is not None and gr.get("version", 1) > 1

    def guardrails_hash(self, category: str) -> Optional[str]:
        """Return SHA-256 of the guardrails file for change detection."""
        path = self._path_for(category)
        if not os.path.exists(path):
            return None
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(4096), b""):
                sha.update(block)
        return sha.hexdigest()

    def build_kg_prompt_prefix(
        self,
        category: str,
        title: str,
        document_summary: Optional[str] = None,
    ) -> str:
        """
        Build a prompt prefix that instructs the LLM to extract knowledge
        graph triplets following the guardrails for *category*.

        When a *document_summary* is provided it is embedded directly after the
        header so the extractor retains full-document context even when operating
        on small 256-token chunks.
        """
        gr = self.get_guardrails(category)

        # Base Persona and Context
        header = (
            f"You are a Knowledge Graph Extraction Expert. Your goal is to build a highly accurate, "
            f"non-redundant graph from the document '{title}' (Category: '{category}').\n\n"
        )

        # Inject document-level summary so every chunk is grounded in the
        # overall meaning of the source document.
        summary_block = ""
        if document_summary:
            summary_block = (
                "Document Summary (use this to resolve ambiguous references and maintain "
                "consistency across all chunks extracted from this document):\n"
                f"{document_summary}\n\n"
            )

        if gr is None:
            return (
                header
                + summary_block
                + "Extract clear and factual knowledge graph triplets from the text below.\n\n"
            )

        # Handle both old 'entity_types' and new 'business_objects'
        entity_types = gr.get("entity_types", [])
        business_objects = gr.get("business_objects", [])

        objects_desc = ""
        if business_objects:
            objects_desc = "Standard Business Objects and their expected Properties:\n"
            for obj in business_objects:
                props = ", ".join(obj.get("properties", []))
                desc = obj.get("description", "")
                desc_str = f" - ({desc})" if desc else ""
                objects_desc += f"- {obj['name']}{desc_str}: [{props}]\n"
        elif entity_types:
            objects_desc = f"Allowed Entity Types: {', '.join(entity_types)}\n"

        raw_rels = gr.get("relationship_types", [])
        rel_desc_block = "Allowed Relationship Types:\n"
        for r in raw_rels:
            if isinstance(r, str):
                rel_desc_block += f"- {r}\n"
            elif isinstance(r, dict) and "name" in r:
                name = r["name"]
                desc = r.get("description", "")
                desc_str = f": {desc}" if desc else ""
                rel_desc_block += f"- {name}{desc_str}\n"

        raw_conv = gr.get("conventions", "")
        if isinstance(raw_conv, dict):
            import json as _json
            conventions = _json.dumps(raw_conv)
        else:
            conventions = str(raw_conv)

        # Few-Shot Examples for grounding
        few_shot_examples = (
            "--- High-Quality Triplet Examples ---\n"
            "Text: 'Apple Inc. was founded by Steve Jobs in Cupertino.'\n"
            "Triplets:\n"
            "(Apple_Inc, FOUNDED_BY, Steve_Jobs)\n"
            "(Apple_Inc, LOCATED_IN, Cupertino)\n"
            "(Apple_Inc, HAS_PROPERTY, legal_name: Apple Inc.)\n"
            "(Steve_Jobs, HAS_PROPERTY, common_name: Steve Jobs)\n"
            "---\n\n"
        )

        return (
            f"{header}"
            f"{summary_block}"
            f"{objects_desc}\n"
            f"{rel_desc_block}\n"
            f"Conventions: {conventions}\n\n"
            "Strict Guidelines for High Accuracy & Normalization:\n"
            "1. STRICT NORMALIZATION & NO DUPLICATES: Always use the most canonical, universally accepted name for an entity "
            "(e.g., JavaScript instead of JS, ReactJS instead of React).\n"
            "   If a similar business object was already mentioned or is provided in the alignment context, use that EXACT name "
            "to avoid duplicates. DO NOT wrap names in literal quotes.\n"
            "   IMPORTANT: Use lowercase_with_underscores for ALL entity names (e.g., 'apple_inc' not 'Apple Inc', "
            "'steve_jobs' not 'Steve Jobs'). This is mandatory for graph alignment.\n"
            "2. ATOMIC TRIPLETS: Each extraction must be a single (Subject, Predicate, Object) fact.\n"
            "3. RESOLVE PRONOUNS: Replace 'he', 'it', 'they' with the specific entity names they refer to.\n"
            "4. EXTRACT PROPERTIES: For each Business Object extracted, also extract its relevant properties as additional triplets. "
            "   Format property extraction as: (EntityName, HAS_PROPERTY, PropertyName: Value).\n"
            "5. NO SPECULATION: Extract only what is explicitly stated in the text. If the text is vague, DO NOT extract the triplet.\n"
            "6. CONSISTENT TYPES: Stick strictly to the allowed business objects and relationship types.\n"
            "7. CONSISTENT NAMING: Never produce two different names for the same entity across triplets. "
            "If you referred to an entity as 'company_x' in one triplet, always use 'company_x' in all subsequent triplets.\n"
            "8. TEMPORAL AWARENESS: If a relationship has a time qualifier "
            "(e.g., 'from 2020', 'since January', 'between 2018 and 2022', 'joined in March 2019'), "
            "preserve it. The system will extract valid_from/valid_to metadata automatically.\n\n"
            f"{few_shot_examples}"
            "Extract the triplets now from the following text:\n\n"
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _path_for(self, category: str) -> str:
        return os.path.join(self.guardrails_dir, f"{category}.json")

    def _summary_key(self, file_path: str) -> str:
        """Stable, filesystem-safe key derived from the file path."""
        return hashlib.sha256(os.path.abspath(file_path).encode()).hexdigest()[:24]

    def _save(self, category: str, guardrails: Dict):
        with open(self._path_for(category), "w") as f:
            json.dump(guardrails, f, indent=2)
        self._cache[category] = guardrails

    def _collect_samples(self, documents: List[Document], max_chars: int = 3000) -> str:
        """Concatenate beginning of each document up to *max_chars* total."""
        parts: List[str] = []
        total = 0
        for doc in documents:
            snippet = doc.text[: max(500, max_chars - total)]
            parts.append(snippet)
            total += len(snippet)
            if total >= max_chars:
                break
        return "\n---\n".join(parts)

    def _collect_from_summaries(self, summaries: List[str], max_chars: int = 4000) -> str:
        """
        Concatenate summaries up to *max_chars*.  Summaries are much denser
        than raw text, so we can afford a slightly larger budget and still stay
        well within LLM context limits.
        """
        parts: List[str] = []
        total = 0
        for summary in summaries:
            if not summary:
                continue
            snippet = summary[: max(300, max_chars - total)]
            parts.append(snippet)
            total += len(snippet)
            if total >= max_chars:
                break
        return "\n---\n".join(parts)

    def _build_generation_prompt(
        self, category: str, input_text: str, input_label: str = "document samples"
    ) -> str:
        return (
            f'You are a Senior Domain Architect and Knowledge Graph Expert specializing in granular Business Object modeling.\n\n'
            f'Analyse the {input_label} below which belong to the "{category}" category.\n'
            "Identify the core Business Objects and their relevant properties. "
            "Break down high-level concepts into extremely granular, specific entities to maximize knowledge graph utility "
            "(e.g., instead of just \"Experience\", use \"Role\", \"Company\", \"Project\", \"TechnologyArea\").\n\n"
            "Produce a JSON object with exactly these keys:\n"
            '- "category": the category name\n'
            '- "business_objects": a JSON array of objects, each with:\n'
            '    - "name": PascalCase entity type (e.g. "Company", "FinancialReport", "SoftwareEngineer")\n'
            '    - "description": a brief explanation of what this entity represents to ensure consistent interpretation\n'
            '    - "properties": a list of relevant property names for this object\n'
            '- "relationship_types": a JSON array of relationship type strings (UPPER_SNAKE_CASE, e.g. ["ISSUED_BY", "WORKS_FOR", "SKILLED_IN"])\n'
            '- "conventions": a short string explaining strict naming, modelling conventions, and normalization rules for rigorous high accuracy\n'
            '- "version": 1\n\n'
            "Respond ONLY with the JSON object, no extra text.\n\n"
            f"--- {input_label.upper()} ---\n{input_text}\n"
        )

    def _build_optimization_prompt(self, category: str, current: Dict) -> str:
        return (
            f'You are a rigorous knowledge-graph schema optimiser enforcing normalization and strict data quality.\n\n'
            f'Below are the current guardrails for the "{category}" category:\n'
            f"{json.dumps(current, indent=2)}\n\n"
            "Review and improve them for a highly granular Business Object centered model:\n"
            "1. Ensure 'business_objects' is the primary key instead of 'entity_types'.\n"
            "2. Refine the list of properties for each object to be essential and descriptive. Break down overly broad entities into more specific sub-types.\n"
            "3. Add a clear 'description' field to every business object if missing, summarizing its exact scope.\n"
            "4. Merge truly redundant objects or relationship types, but preserve necessary granularity.\n"
            "5. Ensure naming conventions are strictly followed (UPPER_SNAKE for relationships, PascalCase for objects).\n"
            "6. Add strict guardrail conventions for NORMALIZATION: instruct the extraction to definitively align similar references "
            "(e.g., \"AngularJS\" vs \"Angular\" -> \"Angular\", \"JS\" vs \"JavaScript\" -> \"JavaScript\").\n\n"
            "Respond ONLY with the improved JSON object (same keys, ensuring 'business_objects' and 'description' are present), no extra text.\n"
        )

    def _parse_guardrails_response(self, text: str, category: str) -> Dict:
        """Best-effort JSON extraction from LLM output."""
        text = text.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
        return {
            "category": category,
            "entity_types": ["Entity"],
            "relationship_types": ["RELATED_TO"],
            "conventions": "Default fallback guardrails – LLM output could not be parsed.",
            "version": 1,
        }
