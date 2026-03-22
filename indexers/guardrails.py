import os
import json
import hashlib
from typing import List, Dict, Optional
from llama_index.core import Settings
from llama_index.core.schema import Document


GUARDRAILS_DIR = os.path.join(os.path.dirname(__file__), "generated_guardrails")


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
    """

    def __init__(self, guardrails_dir: str = GUARDRAILS_DIR):
        self.guardrails_dir = guardrails_dir
        os.makedirs(self.guardrails_dir, exist_ok=True)
        self._cache: Dict[str, Dict] = {}

    # ------------------------------------------------------------------
    # Public API
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
        target_rels = {(rel if isinstance(rel, str) else rel.get("name", "")).upper() for rel in target_gr.get("relationship_types", [])}
        
        # If no objects or relationships, can't determine similarity
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
            other_rels = {(rel if isinstance(rel, str) else rel.get("name", "")).upper() for rel in other_gr.get("relationship_types", [])}

            # Intersection over Union (IoU) / Jaccard similarity for entity types
            obj_intersection = target_obj_names.intersection(other_obj_names)
            obj_union = target_obj_names.union(other_obj_names)
            obj_sim = len(obj_intersection) / len(obj_union) if obj_union else 0

            # Jaccard similarity for relationship types
            rel_intersection = target_rels.intersection(other_rels)
            rel_union = target_rels.union(other_rels)
            rel_sim = len(rel_intersection) / len(rel_union) if rel_union else 0

            # Weight them equally (or adjust weights if needed)
            overall_sim = (obj_sim + rel_sim) / 2 if (obj_union or rel_union) else 0

            if overall_sim >= threshold:
                similar.append(category)

        return similar

    def generate_guardrails(
        self, category: str, documents: List[Document]
    ) -> Dict:
        """
        Use the configured LLM to analyse sample documents and produce
        guardrails (entity types, relationship types, conventions) for the
        given *category*.  Persists the result to disk and returns it.
        """
        sample_texts = self._collect_samples(documents)

        prompt = self._build_generation_prompt(category, sample_texts)
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

    def build_kg_prompt_prefix(self, category: str, title: str) -> str:
        """
        Build a prompt prefix that instructs the LLM to extract knowledge
        graph triplets following the guardrails for *category*.
        The document *title* is included for better context.
        """
        gr = self.get_guardrails(category)
        
        # Base Persona and Context
        header = (
            f"You are a Knowledge Graph Extraction Expert. Your goal is to build a highly accurate, "
            f"non-redundant graph from the document '{title}' (Category: '{category}').\n\n"
        )

        if gr is None:
            return header + "Extract clear and factual knowledge graph triplets from the text below.\n\n"

        # Handle both old 'entity_types' and new 'business_objects' for backward compatibility
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

        # Handle relationship_types as list of strings OR list of dicts
        raw_rels = gr.get("relationship_types", [])
        rel_names = []
        for r in raw_rels:
            if isinstance(r, str):
                rel_names.append(r)
            elif isinstance(r, dict) and "name" in r:
                rel_names.append(r["name"])
        relationship_types = ", ".join(rel_names)

        # Handle conventions as string or dict
        raw_conv = gr.get("conventions", "")
        if isinstance(raw_conv, dict):
            import json as _json
            conventions = _json.dumps(raw_conv)
        else:
            conventions = str(raw_conv)

        return (
            f"{header}"
            f"{objects_desc}"
            f"Allowed Relationship Types: {relationship_types}\n"
            f"Conventions: {conventions}\n\n"
            "Strict Guidelines for High Accuracy & Normalization:\n"
            "1. STRICT NORMALIZATION & NO DUPLICATES: Always use the most canonical, universally accepted name for an entity (e.g., JavaScript instead of JS, ReactJS instead of React).\n"
            "   If a similar business object was already mentioned or is provided in the alignment context, use that EXACT name to avoid duplicates. DO NOT wrap names in literal quotes.\n"
            "2. ATOMIC TRIPLETS: Each extraction must be a single (Subject, Predicate, Object) fact.\n"
            "3. RESOLVE PRONOUNS: Replace 'he', 'it', 'they' with the specific entity names they refer to.\n"
            "4. EXTRACT PROPERTIES: For each Business Object extracted, also extract its relevant properties as additional triplets. "
            "   Format property extraction as: (EntityName, HAS_PROPERTY, PropertyName: Value).\n"
            "5. NO SPECULATION: Extract only what is explicitly stated in the text.\n"
            "6. CONSISTENT TYPES: Stick strictly to the allowed business objects and relationship types.\n\n"
            "Extract the triplets now from the following text:\n\n"
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _path_for(self, category: str) -> str:
        return os.path.join(self.guardrails_dir, f"{category}.json")

    def _save(self, category: str, guardrails: Dict):
        with open(self._path_for(category), "w") as f:
            json.dump(guardrails, f, indent=2)
        # Invalidate cache
        self._cache[category] = guardrails

    def _collect_samples(
        self, documents: List[Document], max_chars: int = 3000
    ) -> str:
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

    def _build_generation_prompt(self, category: str, samples: str) -> str:
        return f"""You are a Senior Domain Architect and Knowledge Graph Expert specializing in granular Business Object modeling.

Analyse the sample documents below which belong to the "{category}" category.
Identify the core Business Objects and their relevant properties. 
Break down high-level concepts into extremely granular, specific entities to maximize knowledge graph utility (e.g., instead of just "Experience", use "Role", "Company", "Project", "TechnologyArea").

Produce a JSON object with exactly these keys:
- "category": the category name
- "business_objects": a JSON array of objects, each with:
    - "name": PascalCase entity type (e.g. "Company", "FinancialReport", "SoftwareEngineer")
    - "description": a brief explanation of what this entity represents to ensure consistent interpretation
    - "properties": a list of relevant property names for this object
- "relationship_types": a JSON array of relationship type strings (UPPER_SNAKE_CASE, e.g. ["ISSUED_BY", "WORKS_FOR", "SKILLED_IN"])
- "conventions": a short string explaining strict naming, modelling conventions, and normalization rules for rigorous high accuracy
- "version": 1

Respond ONLY with the JSON object, no extra text.

--- SAMPLE DOCUMENTS ---
{samples}
"""

    def _build_optimization_prompt(self, category: str, current: Dict) -> str:
        return f"""You are a rigorous knowledge-graph schema optimiser enforcing normalization and strict data quality.

Below are the current guardrails for the "{category}" category:
{json.dumps(current, indent=2)}

Review and improve them for a highly granular Business Object centered model:
1. Ensure 'business_objects' is the primary key instead of 'entity_types'.
2. Refine the list of properties for each object to be essential and descriptive. Break down overly broad entities into more specific sub-types.
3. Add a clear 'description' field to every business object if missing, summarizing its exact scope.
4. Merge truly redundant objects or relationship types, but preserve necessary granularity.
5. Ensure naming conventions are strictly followed (UPPER_SNAKE for relationships, PascalCase for objects).
6. Add strict guardrail conventions for NORMALIZATION: instruct the extraction to definitively align similar references (e.g., "AngularJS" vs "Angular" -> "Angular", "JS" vs "JavaScript" -> "JavaScript").

Respond ONLY with the improved JSON object (same keys, ensuring 'business_objects' and 'description' are present), no extra text.
"""

    def _parse_guardrails_response(self, text: str, category: str) -> Dict:
        """Best-effort JSON extraction from LLM output."""
        # Try to find a JSON block
        text = text.strip()
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end > start:
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                pass
        # Fallback: return a minimal valid structure
        return {
            "category": category,
            "entity_types": ["Entity"],
            "relationship_types": ["RELATED_TO"],
            "conventions": "Default fallback guardrails – LLM output could not be parsed.",
            "version": 1,
        }
