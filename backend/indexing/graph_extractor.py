from llama_index.llms.ollama import Ollama
from datetime import datetime, timezone
import os
import re
import asyncio
import concurrent.futures
import threading
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

    Performance improvements over the original:
    - FIX 1: Temporal properties are extracted in a single batched LLM call
             per chunk instead of one call per triplet.
    - FIX 2: Chunks are processed concurrently via a ThreadPoolExecutor so
             independent LLM calls overlap (wall-time / workers).
    - FIX 3: Actor-Critic verification is skipped for chunks with <=3 triplets
             where the overhead exceeds the benefit.
    - FIX 4: EntityNode objects are cached by ID within each __call__ invocation
             to avoid duplicate MERGE operations in Neo4j.

    Verbose per-step progress logging is emitted for every chunk so you can
    always tell whether the extractor is running or stuck.
    """

    # ------------------------------------------------------------------
    # FIX 1 - Batched temporal extraction
    # ------------------------------------------------------------------

    def _extract_temporal_properties_batch(
        self,
        node_text: str,
        triplets_info: List[Tuple[str, str, str]],
    ) -> List[Dict]:
        """
        Extract temporal metadata for *all* triplets in a single LLM call.

        Args:
            node_text:     The source chunk text.
            triplets_info: List of (rel_type, subj_name, obj_name) tuples.

        Returns:
            Parallel list of dicts, each with optional 'valid_from' / 'valid_to'.
        """
        empty = [{} for _ in triplets_info]
        if not triplets_info:
            return empty

        temporal_keywords = [
            "since", "from", "until", "to", "between", "during",
            "started", "ended", "joined", "left", "resigned",
            "appointed", "established", "founded", "dissolved",
        ]
        if not any(kw in node_text.lower() for kw in temporal_keywords):
            return empty

        items = "\n".join(
            f"{i + 1}. ({s}) --[{r}]--> ({o})"
            for i, (r, s, o) in enumerate(triplets_info)
        )
        prompt = (
            "You are a temporal metadata extractor.\n\n"
            f"Source text:\n{node_text[:1000]}\n\n"
            f"Relationships:\n{items}\n\n"
            "For EACH relationship return a JSON array where every element is an object "
            'with optional keys "valid_from" and "valid_to" (string values). '
            'Use {} when no temporal info exists for that relationship.\n'
            "Return ONLY the JSON array with no explanation or markdown fences.\n"
            'Example for 2 relationships: [{}, {"valid_from": "2020", "valid_to": "present"}]'
        )
        try:
            import json as _json
            response = self.llm.complete(prompt).text.strip()
            start = response.find("[")
            end = response.rfind("]") + 1
            if start != -1 and end > start:
                parsed = _json.loads(response[start:end])
                if isinstance(parsed, list) and len(parsed) == len(triplets_info):
                    result = []
                    for entry in parsed:
                        props = {}
                        if isinstance(entry, dict):
                            if entry.get("valid_from"):
                                props["valid_from"] = str(entry["valid_from"])
                            if entry.get("valid_to"):
                                props["valid_to"] = str(entry["valid_to"])
                        result.append(props)
                    return result
        except Exception as e:
            print(f"     [DEBUG] Batched temporal extraction failed: {e}. Using empty props.", flush=True)
        return empty

    # ------------------------------------------------------------------
    # Actor-Critic verification
    # ------------------------------------------------------------------

    def _verify_triplets(self, node_text: str, triplets: List[Any], chunk_label: str = "") -> List[Any]:
        """
        Actor-Critic verification: asks the LLM to verify if the extracted
        triplets are explicitly supported by the source text.

        FIX 3: Only called when len(triplets) > 3.
        """
        if not triplets:
            return []

        claims = []
        for i, t in enumerate(triplets):
            subj = (t.subject.name or "Unknown") if hasattr(t, "subject") else "Unknown"
            obj  = (t.object.name  or "Unknown") if hasattr(t, "object")  else "Unknown"
            rel  = (t.relation.type or "RELATED_TO") if hasattr(t, "relation") else "RELATED_TO"
            claims.append(f"{i + 1}. ({subj}) --[{rel}]--> ({obj})")

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
                            clean_idx = "".join(c for c in idx_str if c.isdigit())
                            if clean_idx:
                                valid_indices.append(int(clean_idx) - 1)
                        except ValueError:
                            pass

            verified = [triplets[i] for i in valid_indices if 0 <= i < len(triplets)]
            print(
                f"     [Extractor] {chunk_label} verification: {len(verified)}/{len(triplets)} triplets kept",
                flush=True,
            )
            return verified
        except Exception as e:
            print(
                f"     [Extractor] {chunk_label} verification failed ({e}), keeping all raw triplets",
                flush=True,
            )
            return triplets

    # ------------------------------------------------------------------
    # FIX 2 - Single-node processing extracted for thread-pool use
    # ------------------------------------------------------------------

    def _process_single_node(
        self,
        node: Any,
        i: int,
        total: int,
        now_iso: str,
        entity_cache: Dict,
        cache_lock: Any,
        done_counter: Dict,
    ) -> Any:
        """
        Process one chunk: extract, verify, and annotate triplets.
        Safe to call from a ThreadPoolExecutor worker.

        Args:
            entity_cache:  Shared dict mapping entity_id -> EntityNode (FIX 4).
            cache_lock:    threading.Lock protecting entity_cache writes.
            done_counter:  {"n": int, "lock": threading.Lock} for progress tracking.
        """
        from llama_index.core.graph_stores.types import (
            KG_RELATIONS_KEY,
            KG_NODES_KEY,
            Relation,
            EntityNode,
        )

        chunk_label = f"chunk {i + 1}/{total}"

        def _tick_done(relations_count: int = 0, nodes_count: int = 0, failed: bool = False):
            with done_counter["lock"]:
                done_counter["n"] += 1
                completed = done_counter["n"]
            if failed:
                print(
                    f"     [Extractor] {chunk_label} FAILED  ({completed}/{total} complete)",
                    flush=True,
                )
            else:
                print(
                    f"     [Extractor] {chunk_label} DONE    ({completed}/{total} complete)"
                    f" -- {relations_count} relations, {nodes_count} nodes",
                    flush=True,
                )

        try:
            node_name = node.metadata.get("file_name") or node.metadata.get("title", "Unknown")
            print(f"     [Extractor] {chunk_label} STARTED  -- '{node_name}'", flush=True)

            node_text = node.get_content(metadata_mode="llm")

            # Extract source_section from first heading line
            source_section = ""
            for line in node_text.split("\n"):
                stripped = line.strip()
                if stripped and _SECTION_HEADING_RE.match(stripped):
                    source_section = stripped.lstrip("# ").strip()
                    break

            # ---- Step 1: Extract raw triplets --------------------------------
            print(f"     [Extractor] {chunk_label} step 1/3 -- extracting triplets (LLM)...", flush=True)
            kg_schema = self.llm.structured_predict(
                self.kg_schema_cls,
                self.extract_prompt,
                text=node_text,
            )
            raw_triplets = []
            if kg_schema and hasattr(kg_schema, "triplets"):
                raw_triplets = kg_schema.triplets or []
            print(
                f"     [Extractor] {chunk_label} step 1/3 done -- {len(raw_triplets)} raw triplets",
                flush=True,
            )

            # ---- Step 2: Actor-Critic verification (FIX 3: skip if <=3) -----
            if len(raw_triplets) > 3:
                print(
                    f"     [Extractor] {chunk_label} step 2/3 -- verifying {len(raw_triplets)} triplets (LLM)...",
                    flush=True,
                )
                verified_triplets = self._verify_triplets(node_text, raw_triplets, chunk_label)
            else:
                print(
                    f"     [Extractor] {chunk_label} step 2/3 -- skipping verification (<=3 triplets)",
                    flush=True,
                )
                verified_triplets = raw_triplets

            # ---- Step 3: Batched temporal extraction (FIX 1) -----------------
            triplets_info = []
            for t in verified_triplets:
                r = str(getattr(t.relation, "type", "RELATED_TO") or "RELATED_TO").strip()
                s = str(getattr(t.subject,  "name", "") or "").strip()
                o = str(getattr(t.object,   "name", "") or "").strip()
                triplets_info.append((r, s, o))

            if triplets_info:
                print(
                    f"     [Extractor] {chunk_label} step 3/3 -- temporal props"
                    f" for {len(triplets_info)} triplets (batched LLM)...",
                    flush=True,
                )
            else:
                print(
                    f"     [Extractor] {chunk_label} step 3/3 -- no triplets, skipping temporal extraction",
                    flush=True,
                )
            temporal_props_list = self._extract_temporal_properties_batch(node_text, triplets_info)
            if triplets_info:
                print(f"     [Extractor] {chunk_label} step 3/3 done", flush=True)

            # ---- Build KG objects --------------------------------------------
            kg_nodes_list = []
            kg_relations_list = []

            for idx, triplet in enumerate(verified_triplets):
                try:
                    raw_subj = getattr(triplet.subject, "name", None)
                    raw_obj  = getattr(triplet.object,  "name", None)
                    raw_rel  = getattr(triplet.relation, "type", None)

                    if raw_subj is None or raw_obj is None or raw_rel is None:
                        continue

                    subj_name = str(raw_subj).strip()
                    obj_name  = str(raw_obj).strip()
                    rel_type  = str(raw_rel).strip()

                    if not subj_name or not obj_name or not rel_type:
                        continue

                    subj_id = subj_name.replace(" ", "_").lower()
                    obj_id  = obj_name.replace(" ", "_").lower()

                    subj_type = str(getattr(triplet.subject, "type", "Entity") or "Entity").strip()
                    obj_type  = str(getattr(triplet.object,  "type", "Entity") or "Entity").strip()

                    temporal_props = temporal_props_list[idx] if idx < len(temporal_props_list) else {}

                    # FIX 4 - Reuse cached EntityNode objects (lock for thread safety)
                    with cache_lock:
                        if subj_id not in entity_cache:
                            node_props = {
                                "entity_type": subj_type,
                                "title": subj_name,
                                "indexed_at": now_iso,
                            }
                            if source_section:
                                node_props["source_section"] = source_section
                            entity_cache[subj_id] = EntityNode(
                                name=subj_id,
                                label=subj_type,
                                properties=node_props,
                            )
                        subj = entity_cache[subj_id]

                        if obj_id not in entity_cache:
                            obj_props = {
                                "entity_type": obj_type,
                                "title": obj_name,
                                "indexed_at": now_iso,
                            }
                            if source_section:
                                obj_props["source_section"] = source_section
                            entity_cache[obj_id] = EntityNode(
                                name=obj_id,
                                label=obj_type,
                                properties=obj_props,
                            )
                        obj = entity_cache[obj_id]

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
                    t_str = (
                        f"({getattr(triplet.subject, 'name', '?')}) "
                        f"--[{getattr(triplet.relation, 'type', '?')}]--> "
                        f"({getattr(triplet.object, 'name', '?')})"
                    )
                    print(f"     [DEBUG] Skipping triplet [{t_str}]: {e}", flush=True)
                    continue

            node.metadata[KG_RELATIONS_KEY] = kg_relations_list
            node.metadata[KG_NODES_KEY] = kg_nodes_list
            _tick_done(len(kg_relations_list), len(kg_nodes_list))

        except Exception as e:
            from llama_index.core.graph_stores.types import KG_RELATIONS_KEY, KG_NODES_KEY
            node.metadata[KG_RELATIONS_KEY] = []
            node.metadata[KG_NODES_KEY] = []
            _tick_done(failed=True)
            print(f"     [DEBUG] {chunk_label} exception detail: {e}", flush=True)

        return node

    # ------------------------------------------------------------------
    # FIX 2 - Parallel __call__ via ThreadPoolExecutor
    # ------------------------------------------------------------------

    def __call__(self, nodes, show_progress=False, **kwargs):
        """
        Process all nodes concurrently. Each node runs in its own thread so
        blocking LLM calls overlap (wall-time / num_workers).

        kwargs:
            num_workers (int): Thread pool size. Defaults to min(8, len(nodes)).
        """
        now_iso = datetime.now(timezone.utc).isoformat()
        total = len(nodes)

        # Shared state (all thread-safe)
        entity_cache: Dict = {}
        cache_lock = threading.Lock()
        done_counter = {"n": 0, "lock": threading.Lock()}

        num_workers = kwargs.get("num_workers", min(8, max(1, total)))
        print(
            f"     [Extractor] Starting {total} chunk(s) across {num_workers} worker thread(s)...",
            flush=True,
        )

        if num_workers == 1 or total == 1:
            # Fast path: skip thread overhead for tiny workloads
            return [
                self._process_single_node(
                    node, i, total, now_iso, entity_cache, cache_lock, done_counter
                )
                for i, node in enumerate(nodes)
            ]

        result_nodes: List[Any] = [None] * total

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_idx = {
                executor.submit(
                    self._process_single_node,
                    node, i, total, now_iso, entity_cache, cache_lock, done_counter,
                ): i
                for i, node in enumerate(nodes)
            }
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result_nodes[idx] = future.result()
                except Exception as e:
                    print(
                        f"     [DEBUG] Worker for chunk {idx + 1} raised unhandled exception: {e}",
                        flush=True,
                    )
                    result_nodes[idx] = nodes[idx]  # return node unmodified on failure

        print(f"     [Extractor] All {total} chunk(s) complete.", flush=True)
        return result_nodes


# ---------------------------------------------------------------------------
# Chunking Helpers
# ---------------------------------------------------------------------------

# Regex that matches lines which look like section headings:
#   - Markdown headings:          "# Title" / "## Sub-section"
#   - ALL-CAPS lines (>=4 chars): "INTRODUCTION", "METHODOLOGY"
#   - Numbered section headings:  "1.2 Background", "3. Results"
_SECTION_HEADING_RE = re.compile(
    r"^(?:"
    r"#{1,4}\s+.+"           # Markdown headings
    r"|[A-Z][A-Z\s]{3,}$"   # All-caps headings (>=4 chars)
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
    page_markers = list(_PAGE_MARKER_RE.finditer(text))
    section_markers = list(_SECTION_HEADING_RE.finditer(text))

    all_markers = []
    for m in page_markers:
        all_markers.append({"pos": m.start(), "end": m.end(), "type": "page", "val": int(m.group(1))})
    for m in section_markers:
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
        segment_text = text[prev_end : marker["pos"]].strip()
        if segment_text:
            results.append({
                "text": segment_text,
                "section": current_section,
                "page": current_page,
            })
        if marker["type"] == "page":
            current_page = marker["val"]
        else:
            current_section = marker["val"]
        prev_end = marker["end"]

    final_text = text[prev_end:].strip()
    if final_text:
        results.append({
            "text": final_text,
            "section": current_section,
            "page": current_page,
        })

    return results


def _agentic_find_split(text: str, llm, target_size: int = 4000) -> int:
    """
    Ask the LLM to identify the most semantically coherent split point within
    the provided *text*, aiming for a split near *target_size* characters.

    Returns a character index into *text*.
    """
    window_start = max(0, target_size - 1500)
    window_end = min(len(text), target_size + 1500)
    excerpt = text[window_start:window_end]

    prompt = (
        "You are a document chunking expert. Your job is to find the BEST point to split "
        "the following text so that each resulting chunk is self-contained and covers a single topic.\n\n"
        f"TEXT EXCERPT (centered around character index {target_size}):\n"
        "---\n"
        f"{excerpt}\n"
        "---\n\n"
        f"Reply with ONLY a single integer: the character index (relative to the start of the WHOLE text, "
        f"NOT the excerpt) where the split should occur. This index must be between {window_start} and {window_end}. "
        "Pick a point AFTER a sentence ends and BEFORE a new topic begins. "
        "Do not include any explanation."
    )
    try:
        response = llm.complete(prompt).text.strip()
        import re as _re
        numbers = _re.findall(r"\d+", response)
        for n in numbers:
            idx = int(n)
            if window_start < idx < window_end:
                return idx
    except Exception as exc:
        print(f"     [agentic_chunk] LLM split-point detection failed: {exc}", flush=True)

    newline_pos = text.find("\n", target_size - 200, target_size + 200)
    if newline_pos != -1:
        return newline_pos
    return target_size


def _small_to_big_parse(
    documents: List[Document],
    small_chunk_size: int = 256,
    small_chunk_overlap: int = 32,
    big_chunk_size: int = 1024,
    big_chunk_overlap: int = 128,
    agentic_chunk: bool = False,
    llm: Optional[Any] = None,
) -> Tuple[List[TextNode], List[TextNode]]:
    from llama_index.core import Settings as _Settings
    _llm = llm or _Settings.llm
    _embed_model = _Settings.embed_model

    small_splitter = SentenceSplitter(
        chunk_size=small_chunk_size,
        chunk_overlap=small_chunk_overlap,
    )

    big_nodes: List[TextNode] = []
    small_nodes: List[TextNode] = []

    max_chars = big_chunk_size * 4

    for doc in documents:
        sections = _split_by_sections(doc.get_content())
        print(
            f"     [chunking] '{doc.metadata.get('file_name', '?')}': "
            f"{len(sections)} structural segment(s) detected.",
            flush=True,
        )

        topic_blocks: List[Dict[str, Any]] = []
        current_block_text = ""
        current_block_metadata = {}

        for sec in sections:
            sec_text = sec["text"]
            if current_block_text and (len(current_block_text) + len(sec_text) > max_chars * 1.2):
                topic_blocks.append({"text": current_block_text, "metadata": current_block_metadata})
                current_block_text = ""

            if not current_block_text:
                current_block_metadata = {
                    **doc.metadata,
                    "page_number": sec["page"],
                    "section_title": sec["section"],
                }

            current_block_text += "\n\n" + sec_text if current_block_text else sec_text

        if current_block_text:
            topic_blocks.append({"text": current_block_text, "metadata": current_block_metadata})

        final_parent_docs: List[Document] = []

        def _recursive_split(text: str, metadata: dict):
            if len(text) <= max_chars * 1.1:
                final_parent_docs.append(Document(text=text, metadata=metadata))
                return

            if agentic_chunk and _llm:
                print(f"     [agentic_chunk] Splitting {len(text)}-char segment...", flush=True)
                split_idx = _agentic_find_split(text, _llm, target_size=max_chars)
                left, right = text[:split_idx].strip(), text[split_idx:].strip()
                if left and right:
                    _recursive_split(left, metadata)
                    _recursive_split(right, metadata)
                    return

            splitter = SentenceSplitter(chunk_size=big_chunk_size, chunk_overlap=big_chunk_overlap)
            parts = splitter.split_text(text)
            for p in parts:
                final_parent_docs.append(Document(text=p, metadata=metadata))

        for block in topic_blocks:
            _recursive_split(block["text"], block["metadata"])

        for parent_doc in final_parent_docs:
            parent_node = TextNode(text=parent_doc.text, metadata=parent_doc.metadata)
            if not parent_node.node_id:
                import uuid
                parent_node.id_ = str(uuid.uuid4())
            big_nodes.append(parent_node)

            child_chunks = small_splitter.get_nodes_from_documents(
                [Document(text=parent_node.text, metadata=parent_node.metadata)]
            )
            for child in child_chunks:
                if not child.node_id:
                    import uuid
                    child.id_ = str(uuid.uuid4())

                for rel_type in (NodeRelationship.SOURCE, NodeRelationship.NEXT, NodeRelationship.PREVIOUS):
                    rel_info = child.relationships.get(rel_type)
                    if rel_info is not None and not rel_info.node_id:
                        child.relationships.pop(rel_type, None)

                child.metadata["parent_node_id"] = parent_node.node_id
                small_nodes.append(child)

    return small_nodes, big_nodes


# ---------------------------------------------------------------------------
# Schema helpers - build extractors from guardrail JSON
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
    1. RobustSchemaExtractor - constrained to the entity/relationship types
       declared in the guardrails (schema-guided).
    2. SimpleLLMPathExtractor - (Optional) fallback free-form extraction.
    3. ImplicitPathExtractor - captures document-structural relations
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

    if guardrails:
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

    if not guardrails or include_free_form:
        extractors.append(
            SimpleLLMPathExtractor(
                llm=llm,
                max_paths_per_chunk=max(5, max_triplets_per_chunk // 3),
            )
        )

    extractors.append(ImplicitPathExtractor())

    return extractors


# ===========================================================================
# GraphIndexer
# ===========================================================================