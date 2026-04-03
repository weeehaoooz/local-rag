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

from llama_index.core.node_parser import SentenceSplitter
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore
from indexers.base import BaseIndexer

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

    big_splitter = SentenceSplitter(
        chunk_size=big_chunk_size,
        chunk_overlap=big_chunk_overlap,
    )
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

class GraphIndexer(BaseIndexer):
    """Indexer for Property Graph (Neo4j) using PropertyGraphIndex."""

    def __init__(self, storage_context: StorageContext):
        super().__init__("graph")
        self.storage_context = storage_context
        self.index: Optional[PropertyGraphIndex] = None
        self._entity_cache: Dict[str, List[str]] = {}

    def clear_cache(self):
        """Clear the entity cache."""
        self._entity_cache = {}

    # ------------------------------------------------------------------
    # PUBLIC: Synchronous wrapper
    # ------------------------------------------------------------------

    def index_documents(
        self,
        documents: List[Document],
        max_triplets_per_chunk: int = 15,
        title: Optional[str] = None,
        category: Optional[str] = None,
        kg_prompt_prefix: Optional[str] = None,
        num_passes: int = 1,
        similar_categories: Optional[List[str]] = None,
        guardrails: Optional[Dict] = None,
        small_chunk_size: int = 512,
        big_chunk_size: int = 1024,
        include_free_form: bool = False,
        agentic_chunk: bool = False,
    ) -> PropertyGraphIndex:
        """
        Main entry point for indexing documents into a PropertyGraphIndex.
        Uses small-to-big parsing and hybrid extraction.
        """
        if not documents:
            return self.index

        # 0b. Collect per-document summaries from metadata (set by indexer.py).
        # IMPORTANT: remove "summary" from doc.metadata BEFORE chunking.
        # SentenceSplitter measures metadata length against chunk_size; a 300-word
        # summary will exceed a 256-token small-chunk budget and raise ValueError.
        # We stash the summaries in a side-dict keyed by file path and re-attach
        # them to nodes only after chunking is complete.
        doc_summaries: Dict[str, str] = {}
        for doc in documents:
            # pop() removes "summary" from doc.metadata so SentenceSplitter never
            # measures it against chunk_size (a 300-word summary would exceed the
            # 256–512 token small-chunk budget and raise ValueError).
            summary = doc.metadata.pop("summary", "")
            if summary:
                src = doc.metadata.get("file_path") or doc.metadata.get("file_name", "")
                if src:
                    doc_summaries[src] = summary

        # 0. Context-aware entity alignment — use the document summary to rank
        #    existing graph entities by relevance, injecting only the top-K.
        first_summary = next(iter(doc_summaries.values()), None)
        existing_entities = self._fetch_existing_entities(
            category, similar_categories,
            document_summary=first_summary,
        )
        contextual_prefix = self._add_context_to_prefix(kg_prompt_prefix, existing_entities)

        # 1. Small-to-Big chunking (recursive + optional agentic)
        mode = "agentic + recursive" if agentic_chunk else "recursive section-aware"
        print(f"  -> Smart chunking ({mode}, small={small_chunk_size}, big={big_chunk_size})...")
        small_nodes, big_nodes = _small_to_big_parse(
            documents,
            small_chunk_size=small_chunk_size,
            big_chunk_size=big_chunk_size,
            agentic_chunk=agentic_chunk,
            llm=Settings.llm,
        )
        print(f"     {len(small_nodes)} small chunks, {len(big_nodes)} big (parent) chunks")

        # 1b. Apply contextual prefix + per-document summary to each small node.
        # Injecting the summary into every chunk ensures the extractor always has
        # full-document context, even when the active window is only 256 tokens.
        for node in small_nodes:
            node_src = node.metadata.get("file_path") or node.metadata.get("file_name", "")
            node_summary = doc_summaries.get(node_src, "")

            prefix_parts: list[str] = []
            if contextual_prefix:
                prefix_parts.append(contextual_prefix)
            if node_summary:
                summary_header = (
                    "[Document Summary – use to resolve ambiguous references]\n"
                    + node_summary
                )
                prefix_parts.append(summary_header)

            if prefix_parts:
                combined_prefix = "\n\n".join(prefix_parts)
                node.set_content(f"{combined_prefix}\n\n{node.get_content()}")

        # 2. Use LLM from global settings – resolve eagerly so extractors always
        #    receive a concrete instance, never None.
        llm = Settings.llm
        if llm is None:
            raise RuntimeError(
                "GraphIndexer.index_documents: Settings.llm is None. "
                "Configure Settings.llm before calling index_documents()."
            )

        # 3. Build extractor template
        extractors = _build_extractors_from_guardrails(
            guardrails, max_triplets_per_chunk, llm=llm, include_free_form=include_free_form
        )
        extractor_names = [type(e).__name__ for e in extractors]
        print(f"  -> Extractors: {', '.join(extractor_names)}")

        # 4. Initialize (or update) index with current extractors/transformations
        # Re-creating the index object ensures extractors and transformers are properly registered.
        self.index = PropertyGraphIndex(
            nodes=[],
            property_graph_store=self.storage_context.property_graph_store,
            kg_extractors=extractors,
            transformations=Settings.transformations,
            llm=llm,
            embed_model=Settings.embed_model,
            embed_kg_nodes=True,
            use_async=False,
            show_progress=True,
        )

        # Patch upsert_llama_nodes to guarantee ChunkNode always has a valid id_
        # Python's hash() can return negative numbers which Neo4j rejects as null
        import uuid as _uuid
        _original_upsert = self.index.property_graph_store.upsert_llama_nodes
        def _safe_upsert_llama_nodes(llama_nodes):
            for n in llama_nodes:
                if not n.id_:
                    n.id_ = str(_uuid.uuid4())
            return _original_upsert(llama_nodes)
        self.index.property_graph_store.upsert_llama_nodes = _safe_upsert_llama_nodes

        # Patch upsert_relations to drop any Relation whose source_id or target_id is
        # null — these cause Neo4j's MERGE to raise a SemanticError.
        _original_upsert_relations = self.index.property_graph_store.upsert_relations
        def _safe_upsert_relations(relations):
            valid = [
                r for r in relations
                if getattr(r, "source_id", None) and getattr(r, "target_id", None)
            ]
            skipped = len(relations) - len(valid)
            if skipped:
                print(f"     [DEBUG] upsert_relations: dropped {skipped} relation(s) with null source/target id", flush=True)
            if valid:
                return _original_upsert_relations(valid)
        self.index.property_graph_store.upsert_relations = _safe_upsert_relations

        # 5. Insertion passes — batched & semaphore-controlled
        # Nodes are inserted in configurable batches to reduce LLM round-trip
        # overhead. A semaphore limits concurrent Neo4j writes.
        import anyio
        import asyncio
        nest_asyncio.apply()
        index_ref = self.index
        BATCH_SIZE = 4  # nodes per insert batch (tune based on VRAM / context)
        MAX_CONCURRENT = 2  # max parallel batch inserts

        def _sanitize_node(node):
            """Ensure a node has a valid id and clean relation metadata."""
            import uuid as _uuid
            from llama_index.core.graph_stores.types import KG_RELATIONS_KEY, KG_NODES_KEY
            if not node.node_id:
                node.id_ = str(_uuid.uuid4())
            raw_rels = node.metadata.get(KG_RELATIONS_KEY, [])
            clean_rels = []
            for r in raw_rels:
                src = getattr(r, "source_id", None)
                tgt = getattr(r, "target_id", None)
                nid = getattr(r, "id", None)
                if src is not None and tgt is not None:
                    clean_rels.append(r)
                elif src is None and tgt is None and nid is not None:
                    clean_rels.append(r)
            node.metadata[KG_RELATIONS_KEY] = clean_rels
            return node

        async def _insert_batch(batch, sem, pass_num):
            """Insert a batch of nodes under a concurrency semaphore."""
            async with sem:
                failed = 0
                for node in batch:
                    node = _sanitize_node(node)
                    try:
                        index_ref.insert_nodes([node])
                    except Exception as exc:
                        if "ConstraintValidationFailed" in str(exc):
                            pass
                        else:
                            failed += 1
                            print(f"     [Pass {pass_num}] Insert failed: {exc}", flush=True)
                return failed

        async def _insert_pass(pass_num):
            sem = asyncio.Semaphore(MAX_CONCURRENT)
            batches = [
                small_nodes[i : i + BATCH_SIZE]
                for i in range(0, len(small_nodes), BATCH_SIZE)
            ]
            tasks = [_insert_batch(b, sem, pass_num) for b in batches]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            total_failed = sum(r for r in results if isinstance(r, int))
            exceptions = [r for r in results if isinstance(r, Exception)]
            if total_failed:
                print(f"     ({total_failed} chunk(s) failed during pass {pass_num})")
            if exceptions:
                print(f"     ({len(exceptions)} batch(es) raised exceptions during pass {pass_num})")

        for i in range(num_passes):
            print(f"  -> PropertyGraph Extraction Pass {i + 1}/{num_passes} ({len(small_nodes)} nodes, batch={BATCH_SIZE})...")
            try:
                loop = asyncio.get_event_loop()
                loop.run_until_complete(_insert_pass(i + 1))
            except Exception as e:
                err_str = str(e)
                if "ConstraintValidationFailed" in err_str:
                    print(f"     (Constraint error on pass {i + 1}, continuing)")
                else:
                    print(f"     Warning: Extraction pass error: {e}", flush=True)
                    import traceback
                    print(traceback.format_exc(), flush=True)

        # 5. Post-processing: collapse HAS_PROPERTY triplets into node props
        print("  -> Refining Business Object properties in Neo4j...")
        self._process_properties_in_graph()

        # 6. Promote entity_type → Neo4j label (enables per-type colors in Browser)
        #    and stamp category property so every node knows its document group.
        print("  -> Applying entity-type labels and category node...")
        self._apply_entity_labels(category)
        self._apply_category_node(category)

        return self.index

    # ------------------------------------------------------------------
    # Post-processing helpers (unchanged from original)
    # ------------------------------------------------------------------

    def _process_properties_in_graph(self):
        """
        Finds all (n)-[HAS_PROPERTY]->(p) relationships and converts them
        to actual properties on node n.
        """
        if not isinstance(self.storage_context.property_graph_store, Neo4jPropertyGraphStore):
            return

        fetch_query = """
        MATCH (n)-[r:HAS_PROPERTY]->(p)
        RETURN elementId(n) as node_id, p.name as prop_raw, elementId(p) as prop_node_id, n.name as node_name
        """
        results = self.storage_context.property_graph_store.structured_query(fetch_query)

        if not results:
            return

        for record in results:
            node_id = record.get("node_id")
            prop_raw = record.get("prop_raw")
            prop_node_id = record.get("prop_node_id")
            node_name = record.get("node_name")

            if not prop_raw or ":" not in prop_raw:
                continue

            parts = prop_raw.split(":", 1)
            key = parts[0].strip().replace(" ", "_").lower()
            value = parts[1].strip()

            if key == "id":
                continue

            params = {"node_id": node_id, "val": value}
            if key in ["name", "title", "full_name"] and (not node_name or len(value) > len(node_name)):
                # Sanitize key name for safe interpolation as property key
                import re as _re
                key_safe = _re.sub(r'[^a-zA-Z0-9_]', '', key)
                set_query = f"MATCH (n) WHERE elementId(n) = $node_id SET n.name = $val, n.`{key_safe}` = $val"
            else:
                import re as _re
                key_safe = _re.sub(r'[^a-zA-Z0-9_]', '', key)
                set_query = f"MATCH (n) WHERE elementId(n) = $node_id SET n.`{key_safe}` = $val"

            try:
                self.storage_context.property_graph_store.structured_query(set_query, param_map=params)
                del_query = f"MATCH (p) WHERE elementId(p) = $prop_node_id DETACH DELETE p"
                self.storage_context.property_graph_store.structured_query(del_query, param_map={"prop_node_id": prop_node_id})
            except Exception as e:
                print(f"    Warning: Failed to refine property '{key}' for node {node_id}: {e}")

    # ------------------------------------------------------------------
    # Contextual alignment helpers
    # ------------------------------------------------------------------

    def _fetch_existing_entities(
        self,
        category: Optional[str],
        similar_categories: Optional[List[str]] = None,
        document_summary: Optional[str] = None,
        top_k: int = 20,
    ) -> List[str]:
        """
        Fetch semantically relevant existing entities from the graph.

        Instead of returning 150 random entities, this:
        1. Fetches all entity names + types from Neo4j.
        2. Embeds the current document's summary.
        3. Embeds all entity names.
        4. Returns only the top-K entities most similar to the document summary.

        This prevents polluting the extraction prompt with unrelated entities
        (e.g. finance entities when indexing a healthcare document).
        """
        cache_key = f"{category or 'global'}_{'_'.join(sorted(similar_categories or []))}"
        # Only use cache when there is no document_summary to rank against
        if document_summary is None and cache_key in self._entity_cache:
            return self._entity_cache[cache_key]

        if not isinstance(self.storage_context.property_graph_store, Neo4jPropertyGraphStore):
            return []

        query = """
        MATCH (n)
        WHERE (n.name IS NOT NULL OR n.id IS NOT NULL)
          AND NOT any(lbl IN labels(n) WHERE lbl IN ['__Community__', 'Category', 'CommunitySummary'])
        RETURN DISTINCT coalesce(n.name, n.id) as name,
               coalesce(n.title, n.name, n.id) as title,
               labels(n)[0] as type
        """
        results = self.storage_context.property_graph_store.structured_query(query)
        raw_entities: List[Dict[str, str]] = []
        if isinstance(results, list):
            for r in results:
                if isinstance(r, dict):
                    raw_entities.append({
                        "name": r.get("name", ""),
                        "title": r.get("title", r.get("name", "")),
                        "type": r.get("type", "Entity"),
                    })
                elif hasattr(r, "values"):
                    vals = list(r.values())
                    raw_entities.append({
                        "name": vals[0] if len(vals) > 0 else "",
                        "title": vals[1] if len(vals) > 1 else (vals[0] if vals else ""),
                        "type": vals[2] if len(vals) > 2 and vals[2] else "Entity",
                    })

        raw_entities = [e for e in raw_entities if e["name"]]
        if not raw_entities:
            return []

        # ── Semantic ranking (when summary is available) ──────────────
        if document_summary and Settings.embed_model is not None and len(raw_entities) > top_k:
            try:
                summary_emb = np.array(
                    Settings.embed_model.get_text_embedding(document_summary)
                )
                entity_labels = [
                    f"{e['title']} ({e['type']})" for e in raw_entities
                ]
                entity_embs = np.array([
                    Settings.embed_model.get_text_embedding(label)
                    for label in entity_labels
                ])
                # Cosine similarity: dot(a, B^T) / (|a| * |B|)
                norms = np.linalg.norm(entity_embs, axis=1)
                norms[norms == 0] = 1e-10
                summary_norm = np.linalg.norm(summary_emb)
                if summary_norm == 0:
                    summary_norm = 1e-10
                similarities = entity_embs @ summary_emb / (norms * summary_norm)
                top_indices = np.argsort(similarities)[-top_k:][::-1]

                ranked = [
                    f"{raw_entities[i]['title']} ({raw_entities[i]['type']})"
                    for i in top_indices
                ]
                print(
                    f"     [entity-injection] Ranked {len(raw_entities)} entities → top {len(ranked)} by relevance",
                    flush=True,
                )
                self._entity_cache[cache_key] = ranked
                return ranked
            except Exception as exc:
                print(
                    f"     [entity-injection] Semantic ranking failed ({exc}); falling back to unranked",
                    flush=True,
                )

        # ── Fallback: return all (capped) ─────────────────────────────
        fallback = [
            f"{e['title']} ({e['type']})" for e in raw_entities
        ][:top_k]
        self._entity_cache[cache_key] = fallback
        return fallback

    def _add_context_to_prefix(
        self, prefix: Optional[str], entities: List[str]
    ) -> Optional[str]:
        """Inject semantically-ranked existing entities into the prompt prefix."""
        if not prefix or not entities:
            return prefix

        context = (
            "IMPORTANT — Previously Extracted Entities (Reuse these EXACT names "
            "if referring to the same concept to avoid duplicates!):\n"
            + "\n".join(f"  • {e}" for e in entities)
            + "\n"
        )
        return prefix.replace(
            "Extract the triplets now",
            f"{context}\nExtract the triplets now",
        )

    # ------------------------------------------------------------------
    # Neo4j label & category node post-processing
    # ------------------------------------------------------------------

    def _apply_entity_labels(self, category: Optional[str]):
        """
        After insertion, promote each node's ``entity_type`` property into a
        real Neo4j label.  This is what makes Neo4j Browser color nodes by type.

        LlamaIndex stores everything under the generic ``__Entity__`` label;
        this pass adds the domain label (e.g. ``Company``, ``Person``) on top
        so the Browser's legend reflects actual entity types.

        Optionally also stamps the ``category`` property (document folder) onto
        every node that doesn't already have it.
        """
        if not isinstance(self.storage_context.property_graph_store, Neo4jPropertyGraphStore):
            return

        store = self.storage_context.property_graph_store

        # 1. Promote entity_type → Neo4j label for every node that has it.
        #    APOC is the cleanest way; fall back to a pure-Cypher loop when APOC
        #    is not available.
        apoc_query = """
        MATCH (n)
        WHERE n.entity_type IS NOT NULL AND n.entity_type <> ''
        WITH n, n.entity_type AS lbl
        CALL apoc.create.addLabels(n, [lbl]) YIELD node
        RETURN count(node) AS promoted
        """
        try:
            result = store.structured_query(apoc_query)
            count = 0
            if isinstance(result, list) and result:
                r = result[0]
                count = r.get("promoted", 0) if isinstance(r, dict) else (list(r.values())[0] if hasattr(r, "values") else 0)
            print(f"  -> [Labels] Promoted entity_type to Neo4j label for {count} node(s) via APOC.")
        except Exception:
            # APOC unavailable – collect distinct types and apply with CALL-IN-TRANSACTIONS
            print("  -> [Labels] APOC not available; using pure-Cypher label promotion...")
            type_query = "MATCH (n) WHERE n.entity_type IS NOT NULL RETURN DISTINCT n.entity_type AS t"
            type_results = store.structured_query(type_query)
            entity_types = []
            if isinstance(type_results, list):
                for r in type_results:
                    t = r.get("t") if isinstance(r, dict) else (list(r.values())[0] if hasattr(r, "values") else None)
                    if t:
                        entity_types.append(t)
            for et in entity_types:
                safe = et.replace("`", "")
                q = f"MATCH (n {{entity_type: '{safe}'}}) SET n:`{safe}`"
                try:
                    store.structured_query(q)
                except Exception as e:
                    print(f"     Warning: could not apply label '{safe}': {e}")
            print(f"  -> [Labels] Applied {len(entity_types)} entity-type label(s).")

        # 2. Stamp the category property on nodes that don't have it yet.
        if category:
            cat_safe = category.replace("'", "''")
            cat_q = f"""
            MATCH (n)
            WHERE n.entity_type IS NOT NULL AND (n.category IS NULL OR n.category = '')
            SET n.category = '{cat_safe}'
            """
            try:
                store.structured_query(cat_q)
            except Exception as e:
                print(f"     Warning: could not stamp category on nodes: {e}")

    def _apply_category_node(self, category: Optional[str]):
        """
        Create (or merge) a ``(:Category {{name: '<category>'}})`` anchor node and
        draw ``BELONGS_TO`` edges from every entity in the category to it.

        This gives you a top-level overview node in the Browser and lets you
        visually distinguish document groups with a dedicated color.
        """
        if not category:
            return
        if not isinstance(self.storage_context.property_graph_store, Neo4jPropertyGraphStore):
            return

        store = self.storage_context.property_graph_store
        cat_safe = category.replace("'", "''")

        # Ensure the Category node exists.
        merge_cat = f"MERGE (:Category {{name: '{cat_safe}'}})"
        try:
            store.structured_query(merge_cat)
        except Exception as e:
            print(f"     Warning: could not create Category node '{category}': {e}")
            return

        # Connect all entities that belong to this category.
        connect_q = f"""
        MATCH (n), (c:Category {{name: '{cat_safe}'}})
        WHERE n.category = '{cat_safe}'
          AND NOT (n)-[:BELONGS_TO]->(c)
          AND NOT n:Category
        MERGE (n)-[:BELONGS_TO]->(c)
        """
        try:
            store.structured_query(connect_q)
            print(f"  -> [Category] Linked entities to Category node '{category}'.")
        except Exception as e:
            print(f"     Warning: could not link entities to Category '{category}': {e}")

    def persist(self, persist_dir: str):
        """Neo4j handles its own persistence; no local files to save."""
        pass

    def load(self, persist_dir: str) -> bool:
        """Neo4j is live; index object is re-initialized during index_documents."""
        return True

    # ------------------------------------------------------------------
    # Graph cleanup (unchanged)
    # ------------------------------------------------------------------

    def clean_graph(self, similarity_threshold: float = 0.85, rel_threshold: float = None):
        """
        Refines the knowledge graph by merging nodes representing the same concept.
        """
        if not isinstance(self.storage_context.property_graph_store, Neo4jPropertyGraphStore):
            print("Cleanup only supported for Neo4jPropertyGraphStore.")
            return

        cleaner = GraphCleaner(
            self.storage_context.property_graph_store,
            embed_model=Settings.embed_model
        )
        cleaner.run_cleanup(similarity_threshold, rel_threshold=rel_threshold)


# ===========================================================================
# GraphCleaner (unchanged)
# ===========================================================================

class GraphCleaner:
    """Handles entity resolution/merging in Neo4j using semantic embeddings."""

    def __init__(self, graph_store: Neo4jPropertyGraphStore, embed_model: Any = None):
        self.graph_store = graph_store
        self.embed_model = embed_model or Settings.embed_model

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        a, b = np.asarray(v1), np.asarray(v2)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def run_cleanup(self, similarity_threshold: float = 0.85, rel_threshold: float = None):
        """Main entry point for graph cleaning using semantic embeddings."""
        if rel_threshold is None:
            rel_threshold = similarity_threshold

        print(f"\n--- Starting Knowledge Graph Cleanup (Semantic Threshold: {similarity_threshold}) ---")

        nodes = self._fetch_all_nodes()
        if not nodes:
            print("No nodes found in graph.")
            return

        print(f"Total nodes to analyze: {len(nodes)}")

        # 0. Fast pass: merge nodes whose names are identical after case + whitespace normalisation.
        #    This catches the most common duplicate (same title, different casing) without
        #    needing embeddings.
        print("  -> Running case-insensitive exact-match dedup pass...")
        exact_clusters = self._cluster_exact_case_insensitive(nodes)
        if exact_clusters:
            print(f"     Found {len(exact_clusters)} exact-match cluster(s).")
            for cluster in exact_clusters:
                canonical = self._pick_canonical(cluster)
                duplicates = [n for n in cluster if n["id"] != canonical["id"]]
                dup_names = ", ".join([n["name"] for n in duplicates])
                print(f"     Merging (exact): [{dup_names}] -> [{canonical['name']}]")
                self._merge_nodes(canonical, duplicates)
            # Re-fetch after merges so the semantic pass works on the clean set.
            nodes = self._fetch_all_nodes()
            print(f"     Nodes remaining after exact-match pass: {len(nodes)}")

        if not nodes:
            self._merge_duplicate_relationships()
            print("--- Cleanup Complete ---\n")
            return

        # 1. Generate embeddings for all node names (batch where possible)
        print(f"  -> Generating embeddings for {len(nodes)} unique node names...")
        node_names = [n["name"] for n in nodes]
        try:
            # Simple iteration if the embed_model doesn't support batching well
            embeddings = []
            for i, name in enumerate(node_names):
                if i % 50 == 0 and i > 0:
                    print(f"     Embedded {i}/{len(nodes)}...")
                embeddings.append(self.embed_model.get_text_embedding(name))
            
            for i, node in enumerate(nodes):
                node["_embedding"] = embeddings[i]
        except Exception as e:
            print(f"    Warning: Embedding generation failed: {e}. Falling back to lexical matching.")
            for n in nodes:
                n["_embedding"] = None

        self._normalize_relationship_types(rel_threshold)

        clusters = self._cluster_similar_nodes(nodes, similarity_threshold)
        if not clusters:
            print("No duplicates detected.")
            return

        print(f"Found {len(clusters)} cluster(s) of potential duplicates.")

        for cluster in clusters:
            canonical = self._pick_canonical(cluster)
            duplicates = [n for n in cluster if n["id"] != canonical["id"]]

            duplicate_names = ", ".join([n["name"] for n in duplicates])
            print(f"  -> Merging: [{duplicate_names}] -> [{canonical['name']}]")

            self._merge_nodes(canonical, duplicates)

        self._merge_duplicate_relationships()

        print("--- Cleanup Complete ---\n")

    def _fetch_all_nodes(self) -> List[Dict]:
        """Fetch all nodes with their IDs, names, and labels."""
        query = """
        MATCH (n)
        WHERE (n.name IS NOT NULL OR n.id IS NOT NULL)
          AND NOT any(lbl IN labels(n) WHERE lbl IN ['__Community__', 'Category', 'CommunitySummary'])
        RETURN elementId(n) AS id, coalesce(n.name, n.id) AS name, labels(n) AS labels, properties(n) as props
        """
        result = self.graph_store.structured_query(query)
        nodes = []
        if isinstance(result, list):
            for record in result:
                if isinstance(record, dict):
                    nodes.append(record)
                elif hasattr(record, "values"):
                    values = record.values()
                    nodes.append({"id": values[0], "name": values[1], "labels": values[2], "props": values[3]})
        return [n for n in nodes if n.get("name")]

    def _cluster_exact_case_insensitive(self, nodes: List[Dict]) -> List[List[Dict]]:
        """
        Group nodes whose names are identical after lowering + stripping.
        Returns only groups with 2+ members (i.e. actual duplicates).
        """
        from collections import defaultdict
        buckets: Dict[str, List[Dict]] = defaultdict(list)
        for node in nodes:
            key = node["name"].strip().lower().replace(" ", "_")
            buckets[key].append(node)
        return [group for group in buckets.values() if len(group) > 1]

    def _cluster_similar_nodes(self, nodes: List[Dict], threshold: float) -> List[List[Dict]]:
        """
        Group nodes that are semantically similar using cosine similarity.

        Optimisations over the naive O(N²) approach:
        1. **Label Blocking** — only compare nodes that share at least one
           Neo4j label.  This drastically reduces comparisons when the graph
           contains many entity types (e.g. Person, Company, Technology).
        2. **Vectorised similarity** — when embeddings are available, stack
           them into a NumPy matrix and compute the full cosine similarity
           matrix in one shot (BLAS-accelerated).
        """
        from collections import defaultdict

        # ── Step 1: Group by label (blocking key) ──────────────────────
        label_buckets: Dict[str, List[int]] = defaultdict(list)
        for idx, node in enumerate(nodes):
            labels = node.get("labels", [])
            # Filter out generic LlamaIndex labels
            meaningful = [l for l in labels if l not in ("__Entity__", "__Node__")]
            if meaningful:
                for lbl in meaningful:
                    label_buckets[lbl].append(idx)
            else:
                # Nodes without meaningful labels go into a catch-all bucket
                label_buckets["__UNLABELED__"].append(idx)

        all_clusters: List[List[Dict]] = []
        globally_visited: set = set()

        for label, indices in label_buckets.items():
            bucket_nodes = [nodes[i] for i in indices]
            if len(bucket_nodes) < 2:
                continue

            # ── Step 2: Vectorised cosine similarity (per-block) ───────
            has_embeddings = all(n.get("_embedding") is not None for n in bucket_nodes)

            if has_embeddings:
                emb_matrix = np.array([n["_embedding"] for n in bucket_nodes])
                norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
                norms[norms == 0] = 1e-10
                normed = emb_matrix / norms
                sim_matrix = normed @ normed.T

                visited: set = set()
                for i in range(len(bucket_nodes)):
                    nid = bucket_nodes[i]["id"]
                    if nid in visited or nid in globally_visited:
                        continue
                    cluster = [bucket_nodes[i]]
                    visited.add(nid)
                    for j in range(i + 1, len(bucket_nodes)):
                        other_id = bucket_nodes[j]["id"]
                        if other_id in visited or other_id in globally_visited:
                            continue
                        if sim_matrix[i, j] >= threshold:
                            cluster.append(bucket_nodes[j])
                            visited.add(other_id)
                    if len(cluster) > 1:
                        all_clusters.append(cluster)
                        for n in cluster:
                            globally_visited.add(n["id"])
            else:
                # Fallback: lexical matching within the block
                visited = set()
                for i, node in enumerate(bucket_nodes):
                    if node["id"] in visited or node["id"] in globally_visited:
                        continue
                    cluster = [node]
                    visited.add(node["id"])
                    norm1 = node["name"].lower().strip()
                    for j in range(i + 1, len(bucket_nodes)):
                        other = bucket_nodes[j]
                        if other["id"] in visited or other["id"] in globally_visited:
                            continue
                        norm2 = other["name"].lower().strip()
                        sim = difflib.SequenceMatcher(None, norm1, norm2).ratio()
                        if sim >= threshold:
                            cluster.append(other)
                            visited.add(other["id"])
                    if len(cluster) > 1:
                        all_clusters.append(cluster)
                        for n in cluster:
                            globally_visited.add(n["id"])

        return all_clusters

    def _pick_canonical(self, cluster: List[Dict]) -> Dict:
        """
        Pick the 'best' node to keep.
        Strategy: Most relationships > Most properties > Shortest name.
        """
        best_node = cluster[0]
        max_score = -1

        for node in cluster:
            query = "MATCH (n)-[r]-() WHERE elementId(n) = $node_id RETURN count(r) as rel_count"
            result = self.graph_store.structured_query(query, param_map={"node_id": node["id"]})
            rel_count = 0
            try:
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], dict):
                        rel_count = result[0].get("rel_count", 0)
                    elif hasattr(result[0], "values"):
                        rel_count = result[0].values()[0]
            except Exception:
                pass

            prop_count = len(node.get("props", {}))
            score = (rel_count * 10) + prop_count

            if score > max_score:
                max_score = score
                best_node = node
            elif score == max_score:
                if len(node["name"]) < len(best_node["name"]):
                    best_node = node

        return best_node

    def _merge_nodes(self, canonical: Dict, duplicates: List[Dict]):
        """
        Merge duplicate nodes into the canonical node.
        """
        canonical_id = canonical["id"]

        for dup in duplicates:
            dup_id = dup["id"]

            rel_types_query = "MATCH (d)-[r]-() WHERE elementId(d) = $dup_id RETURN DISTINCT type(r) as type"
            types_res = self.graph_store.structured_query(rel_types_query, param_map={"dup_id": dup_id})
            types = []
            if isinstance(types_res, list):
                for record in types_res:
                    if isinstance(record, dict):
                        types.append(record["type"])
                    elif hasattr(record, "values"):
                        types.append(record.values()[0])

            for t in types:
                t_safe = f"`{t}`"
                # Move outgoing relationships
                self.graph_store.structured_query(f"""
                MATCH (d)-[r:{t_safe}]->(target)
                WHERE elementId(d) = $dup_id AND elementId(target) <> $canonical_id
                MATCH (c) WHERE elementId(c) = $canonical_id
                MERGE (c)-[new_r:{t_safe}]->(target)
                SET new_r += properties(r)
                """, param_map={"dup_id": dup_id, "canonical_id": canonical_id})
                
                # Move incoming relationships
                self.graph_store.structured_query(f"""
                MATCH (source)-[r:{t_safe}]->(d)
                WHERE elementId(d) = $dup_id AND elementId(source) <> $canonical_id
                MATCH (c) WHERE elementId(c) = $canonical_id
                MERGE (source)-[new_r:{t_safe}]->(c)
                SET new_r += properties(r)
                """, param_map={"dup_id": dup_id, "canonical_id": canonical_id})

            # Merge properties using parameters
            d_props = dup.get("props", {})
            set_clauses = []
            params = {"canonical_id": canonical_id}
            for k, v in d_props.items():
                if k not in ["id", "name"]:
                    param_name = f"p_{k.replace(' ', '_')}"
                    set_clauses.append(f"c.`{k}` = ${param_name}")
                    params[param_name] = v

            if set_clauses:
                set_query = f"MATCH (c) WHERE elementId(c) = $canonical_id SET " + ", ".join(set_clauses)
                try:
                    self.graph_store.structured_query(set_query, param_map=params)
                except Exception as e:
                    print(f"    Warning: Failed to merge properties for {canonical_id}: {e}")

            self.graph_store.structured_query("MATCH (d) WHERE elementId(d) = $dup_id DETACH DELETE d", 
                                             param_map={"dup_id": dup_id})

    def _normalize_relationship_types(self, threshold: float):
        """Standardize similar sounding relationship types globally."""
        print("  -> Normalizing relationship types...")
        query = "CALL db.relationshipTypes()"
        result = self.graph_store.structured_query(query)
        rel_types = []
        if isinstance(result, list):
            for record in result:
                if isinstance(record, dict):
                    rel_types.append(record["relationshipType"])
                elif hasattr(record, "values"):
                    rel_types.append(record.values()[0])

        if len(rel_types) < 2:
            return

        clusters = []
        visited = set()
        normalized_map = {rt: rt.upper().replace("_", " ").strip() for rt in rel_types}
        sorted_types = sorted(rel_types, key=len)

        for i, rt in enumerate(sorted_types):
            if rt in visited:
                continue
            cluster = [rt]
            visited.add(rt)
            norm_rt = normalized_map[rt]

            for j in range(i + 1, len(sorted_types)):
                other = sorted_types[j]
                if other in visited:
                    continue
                norm_other = normalized_map[other]
                sim = difflib.SequenceMatcher(None, norm_rt, norm_other).ratio()
                if sim >= threshold:
                    cluster.append(other)
                    visited.add(other)

            if len(cluster) > 1:
                clusters.append(cluster)

        for cluster in clusters:
            canonical = sorted(cluster, key=len)[0]
            duplicates = [rt for rt in cluster if rt != canonical]

            for dup in duplicates:
                print(f"    Merging relationship types: [{dup}] -> [{canonical}]")
                merge_query = f"""
                MATCH (a)-[r:`{dup}`]->(b)
                WITH a, b, properties(r) as props, r
                MERGE (a)-[new_r:`{canonical}`]->(b)
                SET new_r += props
                DELETE r
                """
                try:
                    self.graph_store.structured_query(merge_query)
                except Exception as e:
                    print(f"    Warning: Failed to merge relationship type {dup}: {e}")

    def _merge_duplicate_relationships(self):
        """Find nodes with multiple relationships of the same type between them and merge them."""
        print("  -> Merging duplicate relationships between same node pairs...")
        find_dups_query = """
        MATCH (a)-[r]->(b)
        WITH a, b, type(r) as type, count(r) as count, collect(r) as rels
        WHERE count > 1
        RETURN elementId(a) as source_id, elementId(b) as target_id, type
        """
        results = self.graph_store.structured_query(find_dups_query)
        if not results:
            return

        for record in results:
            source_id = record["source_id"]
            target_id = record["target_id"]
            rel_type = record["type"]

            print(f"    Merging duplicate '{rel_type}' rels between nodes.")
            merge_props_query = f"""
            MATCH (a)-[rs:`{rel_type}`]->(b)
            WHERE elementId(a) = $source_id AND elementId(b) = $target_id
            WITH rs
            ORDER BY elementId(rs) ASC
            WITH collect(rs) as rel_list
            WITH rel_list[0] as first_rel, rel_list[1..] as other_rels
            FOREACH (r IN other_rels | SET first_rel += properties(r) DELETE r)
            """
            try:
                self.graph_store.structured_query(merge_props_query, 
                                                 param_map={"source_id": source_id, "target_id": target_id})
            except Exception as e:
                print(f"    Warning: Failed to merge duplicate rels: {e}")