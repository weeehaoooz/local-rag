from llama_index.llms.ollama import Ollama
from datetime import datetime, timezone
import os
import re
import asyncio
import difflib
import nest_asyncio
import numpy as np
from typing import List, Any, Optional, Dict, Tuple

from llama_index.core import PropertyGraphIndex, StorageContext, Settings
from llama_index.core.schema import Document, TextNode, NodeRelationship
from llama_index.core.indices.property_graph import (
    SchemaLLMPathExtractor,
    SimpleLLMPathExtractor,
    ImplicitPathExtractor,
)

from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from indexing.base import BaseIndexer


class RobustSchemaExtractor(SchemaLLMPathExtractor):
    """
    Bypasses SchemaLLMPathExtractor.__call__ entirely to avoid the
    asyncio.run() -> sniffio context loss chain on Python 3.14.
    Calls llm.structured_predict() (sync) directly instead.
    """

    def _verify_triplets(self, node_text: str, triplets: List[Any]) -> List[Any]:
        """
        Actor-Critic verification: asks the LLM to verify if the extracted
        triplets are explicitly supported by the source text.
        """
        if not triplets:
            return []

        claims = []
        for i, t in enumerate(triplets):
            subj = (t.subject.name or "Unknown") if hasattr(t, "subject") else "Unknown"
            obj = (t.object.name or "Unknown") if hasattr(t, "object") else "Unknown"
            rel = (t.relation.type or "RELATED_TO") if hasattr(t, "relation") else "RELATED_TO"
            claims.append(f"{i+1}. ({subj}) --[{rel}]--> ({obj})")

        claims_str = "\n".join(claims)
        
        prompt = (
            "You are a Fact Checker. Verify if the following Knowledge Graph triplets "
            "are EXPLICITLY supported by the text provided below.\n\n"
            f"--- TEXT ---\n{node_text}\n\n"
            f"--- CLAIMS ---\n{claims_str}\n\n"
            "Response Instructions:\n"
            "- For each claim, respond with ONLY 'YES' or 'NO'.\n"
            "- Use the format: '1: YES', '2: NO', etc. Ensure every claim has a response.\n"
            "- If a claim is partially true or vague, respond with 'NO'.\n"
            "- DO NOT provide any explanation."
        )

        try:
            response = self.llm.complete(prompt).text.strip()
            lines = response.split("\n")
            valid_indices = []
            for line in lines:
                if ":" in line:
                    parts = line.split(":", 1)
                    idx_str = parts[0].strip()
                    val = parts[1].strip().upper()
                    if "YES" in val:
                        try:
                            # Strip any non-digit chars from idx_str (like '1.')
                            clean_idx = "".join(c for c in idx_str if c.isdigit())
                            if clean_idx:
                                valid_indices.append(int(clean_idx) - 1)
                        except ValueError:
                            pass
            
            verified = [triplets[i] for i in valid_indices if 0 <= i < len(triplets)]
            print(f"     [DEBUG] Actor-Critic verified {len(verified)}/{len(triplets)} triplets from chunk", flush=True)
            return verified
        except Exception as e:
            print(f"     [DEBUG] Verification failed: {e}. Falling back to original extractions.", flush=True)
            return triplets

    def _extract_temporal_properties(self, rel_type: str, node_text: str, subj_name: str, obj_name: str) -> dict:
        """
        Ask the LLM to detect temporal qualifiers for a specific relationship.
        Returns a dict with optional 'valid_from' and 'valid_to' keys.
        """
        # Quick heuristic: only call LLM if temporal keywords are present
        temporal_keywords = [
            "since", "from", "until", "to", "between", "during",
            "started", "ended", "joined", "left", "resigned",
            "appointed", "established", "founded", "dissolved",
        ]
        text_lower = node_text.lower()
        if not any(kw in text_lower for kw in temporal_keywords):
            return {}

        prompt = (
            "You are a temporal metadata extractor. Given the relationship below and its source text, "
            "determine if there are any time qualifiers (start date, end date) for this relationship.\n\n"
            f"Relationship: ({subj_name}) --[{rel_type}]--> ({obj_name})\n"
            f"Source text: {node_text[:1000]}\n\n"
            "Respond with ONLY a JSON object with these optional keys:\n"
            '- "valid_from": start date/year as string (e.g. "2020", "2020-01", "2020-01-15")\n'
            '- "valid_to": end date/year as string, or "present" if ongoing\n\n'
            'If no temporal info is found, respond with: {}\n'
            "Do NOT include any explanation."
        )
        try:
            response = self.llm.complete(prompt).text.strip()
            import json as _json
            # Extract JSON from response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end > start:
                parsed = _json.loads(response[start:end])
                result = {}
                if parsed.get("valid_from"):
                    result["valid_from"] = str(parsed["valid_from"])
                if parsed.get("valid_to"):
                    result["valid_to"] = str(parsed["valid_to"])
                return result
        except Exception:
            pass
        return {}

    def __call__(self, nodes, show_progress=False, **kwargs):
        from llama_index.core.graph_stores.types import KG_RELATIONS_KEY, KG_NODES_KEY, Relation, EntityNode

        now_iso = datetime.now(timezone.utc).isoformat()

        result_nodes = []
        for node in nodes:
            try:
                node_text = node.get_content(metadata_mode="llm")

                # Extract source_section from the chunk's content (first heading line)
                source_section = ""
                for line in node_text.split("\n"):
                    stripped = line.strip()
                    if stripped and _SECTION_HEADING_RE.match(stripped):
                        source_section = stripped.lstrip("# ").strip()
                        break

                kg_schema = self.llm.structured_predict(
                    self.kg_schema_cls,
                    self.extract_prompt,
                    text=node_text,
                )
                
                raw_triplets = []
                if kg_schema and hasattr(kg_schema, "triplets"):
                    raw_triplets = kg_schema.triplets or []
                
                # Actor-Critic Validation Step
                verified_triplets = self._verify_triplets(node_text, raw_triplets)
                
                kg_nodes_list = []
                kg_relations_list = []
                for triplet in verified_triplets:
                    try:
                        subj_name = (triplet.subject.name or "").strip()
                        obj_name = (triplet.object.name or "").strip()
                        rel_type = (triplet.relation.type or "").strip()

                        if not subj_name or not obj_name or not rel_type:
                            continue

                        subj_id = subj_name.replace(" ", "_").lower()
                        obj_id  = obj_name.replace(" ", "_").lower()

                        subj_type = triplet.subject.type or "Entity"
                        obj_type  = triplet.object.type or "Entity"

                        # Temporal metadata: detect valid_from / valid_to
                        temporal_props = self._extract_temporal_properties(
                            rel_type, node_text, subj_name, obj_name
                        )

                        # Use the normalized ID as the EntityNode name so that
                        # Neo4j's MERGE matches it with the Relation endpoints.
                        # The original human-readable name is stored in "title".
                        node_props = {
                            "entity_type": subj_type,
                            "title": subj_name,
                            "indexed_at": now_iso,
                        }
                        if source_section:
                            node_props["source_section"] = source_section

                        subj = EntityNode(
                            name=subj_id,
                            label=subj_type,
                            properties=node_props,
                        )

                        obj_props = {
                            "entity_type": obj_type,
                            "title": obj_name,
                            "indexed_at": now_iso,
                        }
                        if source_section:
                            obj_props["source_section"] = source_section

                        obj = EntityNode(
                            name=obj_id,
                            label=obj_type,
                            properties=obj_props,
                        )

                        rel_properties = {"indexed_at": now_iso}
                        rel_properties.update(temporal_props)

                        rel = Relation(
                            source_id=subj_id,
                            target_id=obj_id,
                            label=rel_type,
                            properties=rel_properties,
                        )
                        kg_nodes_list.extend([subj, obj])
                        kg_relations_list.append(rel)
                    except Exception as e:
                        print(f"     [DEBUG] Skipping triplet in conversion: {e}", flush=True)
                        continue
                
                node.metadata[KG_RELATIONS_KEY] = kg_relations_list
                node.metadata[KG_NODES_KEY] = kg_nodes_list
                print(f"     [DEBUG] Validated {len(kg_relations_list)} relations and {len(kg_nodes_list)} nodes", flush=True)
            except Exception as e:
                print(f"     [DEBUG] Extraction/Validation failed: {e}", flush=True)
                node.metadata[KG_RELATIONS_KEY] = []
                node.metadata[KG_NODES_KEY] = []

            result_nodes.append(node)

        return result_nodes


# ---------------------------------------------------------------------------
# Chunking Helpers
# ---------------------------------------------------------------------------

# Regex that matches lines which look like section headings:
#   - Markdown headings:          "# Title" / "## Sub-section"
#   - ALL-CAPS lines (≥4 chars):  "INTRODUCTION", "METHODOLOGY"
#   - Numbered section headings:  "1.2 Background", "3. Results"
_SECTION_HEADING_RE = re.compile(
    r"^(?:"
    r"#{1,4}\s+.+"           # Markdown headings
    r"|[A-Z][A-Z\s]{3,}$"   # All-caps headings (≥4 chars)
    r"|\d+(?:\.\d+)*\.?\s+[A-Z].+"  # Numbered sections
    r")",
    re.MULTILINE,
)

_PAGE_MARKER_RE = re.compile(r"\[Page (\d+)\]")


def _split_by_sections(text: str) -> List[Dict[str, Any]]:
    """
    Split *text* at section-heading boundaries.
    Returns a list of dicts: {"text": str, "section": str, "page": int}
    """
    # 1. Identify all page markers
    page_markers = list(_PAGE_MARKER_RE.finditer(text))
    
    # 2. Identify all section markers
    section_markers = list(_SECTION_HEADING_RE.finditer(text))
    
    # Combine and sort markers by position
    all_markers = []
    for m in page_markers:
        all_markers.append({"pos": m.start(), "end": m.end(), "type": "page", "val": int(m.group(1))})
    for m in section_markers:
        # Extract title: remove markdown prefix or just strip
        title = m.group(0).lstrip("# ").strip()
        all_markers.append({"pos": m.start(), "end": m.end(), "type": "section", "val": title})
    
    all_markers.sort(key=lambda x: x["pos"])
    
    if not all_markers:
        return [{"text": text, "section": "Preamble", "page": 1}]

    results: List[Dict[str, Any]] = []
    current_page = 1
    current_section = "Preamble"
    prev_end = 0
    
    for marker in all_markers:
        # Segment before this marker
        segment_text = text[prev_end : marker["pos"]].strip()
        if segment_text:
            results.append({
                "text": segment_text,
                "section": current_section,
                "page": current_page
            })
        
        # Update state based on marker
        if marker["type"] == "page":
            current_page = marker["val"]
        else:
            current_section = marker["val"]
            
        prev_end = marker["end"]
        
    # Final segment
    final_text = text[prev_end:].strip()
    if final_text:
        results.append({
            "text": final_text,
            "section": current_section,
            "page": current_page
        })
        
    return results


def _agentic_find_split(text: str, llm, window: int = 2000) -> int:
    """
    Ask the LLM to identify the most semantically coherent split point within
    a *window*-character excerpt of *text*.

    The LLM returns a character offset (integer) relative to the start of the
    window.  This is used as an "agentic" override of a fixed-size boundary.

    Returns a character index into *text*.  Falls back to ``len(text) // 2``
    if the LLM response cannot be parsed.

    .. warning::
        This makes one LLM call per invocation.  Only use when ``agentic_chunk=True``.
    """
    excerpt = text[:window]
    prompt = (
        "You are a document chunking expert. Your job is to find the BEST point to split "
        "the following text so that each resulting chunk is self-contained and covers a single topic.\n\n"
        f"TEXT (first {window} characters):\n"
        "---\n"
        f"{excerpt}\n"
        "---\n\n"
        "Reply with ONLY a single integer: the character index (0-based, relative to the start of the text above) "
        "where the split should occur. Pick a point AFTER a sentence ends and BEFORE a new topic begins. "
        "Do not include any explanation."
    )
    try:
        response = llm.complete(prompt).text.strip()
        # Extract first integer from response
        import re as _re
        m = _re.search(r"\d+", response)
        if m:
            idx = int(m.group())
            # Clamp to valid range
            if 0 < idx < len(text):
                return idx
    except Exception as exc:
        print(f"     [agentic_chunk] LLM split-point detection failed: {exc}", flush=True)
    return len(text) // 2


def _small_to_big_parse(
    documents: List[Document],
    small_chunk_size: int = 256,
    small_chunk_overlap: int = 32,
    big_chunk_size: int = 1024,
    big_chunk_overlap: int = 128,
    agentic_chunk: bool = False,
    llm: Optional[Any] = None,
) -> Tuple[List[TextNode], List[TextNode]]:
    """
    Hierarchical & Recursive Small-to-Big parsing strategy.

    Steps
    -----
    1. **Section splitting** (recursive boundary detection)
       Detect natural section boundaries (Markdown headings, ALL-CAPS headings,
       numbered sections) and split there *first* — keeping all content for a
       given topic inside the same parent chunk.

    2. **Agentic split (opt-in)**
       When ``agentic_chunk=True``, each oversized section is further refined
       by asking the LLM to choose the semantically best split point within a
       2 000-char window.  This is expensive (1 LLM call per split) but produces
       the most coherent chunks for highly heterogeneous content.

    3. **SentenceSplitter fallback**
       Any section that still exceeds ``big_chunk_size`` tokens is split by the
       ``SentenceSplitter`` (sentence-boundary aware, no LLM cost).

    4. **Small (child) nodes**
       Each big (parent) node is further split into ``small_chunk_size``-token
       child nodes for precise triplet extraction.

    Returns
    -------
    (small_nodes, big_nodes)
    """
    from llama_index.core import Settings as _Settings
    _llm = llm or _Settings.llm
    _embed_model = _Settings.embed_model

    # Use SemanticSplitter for "Big" (Parent) chunks to ensure topic coherence
    big_splitter = SemanticSplitterNodeParser(
        buffer_size=1,
        breakpoint_percentile_threshold=95,
        embed_model=_embed_model,
    )
    
    # Use SentenceSplitter for "Small" (Child) chunks for precise triplet extraction
    small_splitter = SentenceSplitter(
        chunk_size=small_chunk_size,
        chunk_overlap=small_chunk_overlap,
    )

    big_nodes: List[TextNode] = []
    small_nodes: List[TextNode] = []

    for doc in documents:
        # ----------------------------------------------------------------
        # 1. Recursive section & page splitting
        # ----------------------------------------------------------------
        sections = _split_by_sections(doc.get_content())
        print(
            f"     [chunking] '{doc.metadata.get('file_name', '?')}': "
            f"{len(sections)} structural segment(s) detected.",
            flush=True,
        )

        section_docs: List[Document] = []
        for sec in sections:
            sec_text = sec["text"]
            sec_metadata = {
                **doc.metadata,
                "page_number": sec["page"],
                "section_title": sec["section"]
            }
            
            # Agentic refinement for long sections
            if agentic_chunk and _llm and len(sec_text) > big_chunk_size * 4:
                print(
                    f"     [agentic_chunk] Running LLM split-point detection on a "
                    f"{len(sec_text)}-char segment...",
                    flush=True,
                )
                split_idx = _agentic_find_split(sec_text, _llm)
                left, right = sec_text[:split_idx].strip(), sec_text[split_idx:].strip()
                for part in [left, right]:
                    if part:
                        section_docs.append(
                            Document(text=part, metadata=sec_metadata)
                        )
            else:
                section_docs.append(
                    Document(text=sec_text, metadata=sec_metadata)
                )

        # ----------------------------------------------------------------
        # 2. SentenceSplitter for oversized sections → big (parent) nodes
        # ----------------------------------------------------------------
        # Ensure metadata is preserved through splitting
        parent_chunks = big_splitter.get_nodes_from_documents(section_docs)
        for parent in parent_chunks:
            big_nodes.append(parent)

            # ----------------------------------------------------------------
            # 3. Small (child) nodes from each parent
            # ----------------------------------------------------------------
            child_chunks = small_splitter.get_nodes_from_documents(
                [Document(text=parent.text, metadata=parent.metadata)]
            )
            for child in child_chunks:
                if not child.node_id:
                    import uuid
                    child.id_ = str(uuid.uuid4())
                
                # Metadata already contains page_number and section_title from parent.metadata
                
                # Clear relationships with null node_id
                for rel_type in (
                    NodeRelationship.SOURCE,
                    NodeRelationship.NEXT,
                    NodeRelationship.PREVIOUS,
                ):
                    rel_info = child.relationships.get(rel_type)
                    if rel_info is not None and not rel_info.node_id:
                        child.relationships.pop(rel_type, None)
                child.metadata["parent_node_id"] = parent.node_id
                small_nodes.append(child)

    return small_nodes, big_nodes


# ---------------------------------------------------------------------------
# Schema helpers – build extractors from guardrail JSON
# ---------------------------------------------------------------------------

def _build_extractors_from_guardrails(
    guardrails: Optional[Dict],
    max_triplets_per_chunk: int = 15,
    llm: Optional[Any] = None,
    include_free_form: bool = False,
) -> list:
    """
    Build a list of KG extractors for PropertyGraphIndex based on guardrails.

    Strategy:
    1. **SchemaLLMPathExtractor** – constrained to the entity/relationship
       types declared in the guardrails (schema-guided).
    2. **SimpleLLMPathExtractor** – (Optional) fallback free-form extraction.
    3. **ImplicitPathExtractor** – captures document-structural relations
       (NEXT, PREVIOUS, SOURCE) for free (no LLM calls).
    """
    if llm is None:
        llm = Settings.llm

    if llm is None:
        raise ValueError(
            "_build_extractors_from_guardrails: llm is None. "
            "Ensure Settings.llm is set before calling index_documents(), "
            "or pass the llm argument explicitly."
        )

    extractors = []

    # 1. Schema-guided extractor (primary)
    if guardrails:
        # Collect entity types from business_objects or entity_types
        business_objects = guardrails.get("business_objects", [])
        entity_types_from_objects = [obj["name"] for obj in business_objects if "name" in obj]
        entity_types = entity_types_from_objects or guardrails.get("entity_types", ["Entity"])

        relationship_types_raw = guardrails.get("relationship_types", ["RELATED_TO"])
        relationship_types = []
        for r in relationship_types_raw:
            if isinstance(r, str):
                relationship_types.append(r)
            elif isinstance(r, dict) and "name" in r:
                relationship_types.append(r["name"])

        extractors.append(
            RobustSchemaExtractor(
                llm=llm,
                possible_entities=entity_types,
                possible_relations=relationship_types,
                strict=False,
                max_triplets_per_chunk=max_triplets_per_chunk,
            )
        )

    # 2. Free-form fallback extractor (Optional/Conditional)
    # If we have guardrails AND include_free_form=False, we skip this to save LLM calls.
    if not guardrails or include_free_form:
        extractors.append(
            SimpleLLMPathExtractor(
                llm=llm,
                max_paths_per_chunk=max(5, max_triplets_per_chunk // 3),
            )
        )

    # 3. Implicit structural extractor (no LLM cost)
    extractors.append(ImplicitPathExtractor())

    return extractors


# ===========================================================================
# GraphIndexer
# ===========================================================================

