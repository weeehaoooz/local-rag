import logging
from llama_index.llms.ollama import Ollama

logger = logging.getLogger(__name__)

class GraphService:
    def __init__(self, llm: Ollama, graph_store):
        self.llm = llm
        self.graph_store = graph_store

    async def expand_graph_context(
        self,
        seed_entities: list[str],
        max_hops: int = 1,
        limit: int = 20,
    ) -> list[dict]:
        """
        Given a list of seed entity names, traverse 1..max_hops in Neo4j
        to discover neighboring entities and their relationships.

        Returns a list of dicts with keys:
          name, title, labels, rel_types, valid_from, valid_to, is_expanded
        """
        if not seed_entities:
            return []

        # Normalize seeds to match graph IDs (lowercase + underscores)
        normalized_seeds = [s.replace(" ", "_").lower() for s in seed_entities]

        query = f"""
        MATCH (seed)
        WHERE seed.name IN $entity_names
           OR seed.title IN $raw_names
        WITH seed
        MATCH path = (seed)-[*1..{max_hops}]-(neighbor)
        WHERE neighbor <> seed
          AND NOT any(lbl IN labels(neighbor) WHERE lbl IN ['__Community__', 'Category', 'CommunitySummary'])
          AND neighbor.name IS NOT NULL
        WITH DISTINCT neighbor,
             [r IN relationships(path) | type(r)] AS rel_types,
             [r IN relationships(path) | r.valid_from] AS valid_froms,
             [r IN relationships(path) | r.valid_to] AS valid_tos
        RETURN neighbor.name AS name,
               coalesce(neighbor.title, neighbor.name) AS title,
               labels(neighbor) AS labels,
               rel_types,
               head(valid_froms) AS valid_from,
               head(valid_tos) AS valid_to
        LIMIT $limit
        """
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None, 
                lambda: self.graph_store.structured_query(
                    query,
                    param_map={
                        "entity_names": normalized_seeds,
                        "raw_names": seed_entities,
                        "limit": limit,
                    },
                )
            )
            expanded = []
            if isinstance(results, list):
                seen_names = set(normalized_seeds)
                for r in results:
                    if isinstance(r, dict):
                        name = r.get("name", "")
                    elif hasattr(r, "values"):
                        vals = list(r.values())
                        name = vals[0] if vals else ""
                        r = {
                            "name": name,
                            "title": vals[1] if len(vals) > 1 else name,
                            "labels": vals[2] if len(vals) > 2 else [],
                            "rel_types": vals[3] if len(vals) > 3 else [],
                            "valid_from": vals[4] if len(vals) > 4 else None,
                            "valid_to": vals[5] if len(vals) > 5 else None,
                        }
                    else:
                        continue

                    if name and name not in seen_names:
                        seen_names.add(name)
                        expanded.append({
                            "name": r.get("name", ""),
                            "title": r.get("title", r.get("name", "")),
                            "labels": r.get("labels", []),
                            "rel_types": r.get("rel_types", []),
                            "valid_from": r.get("valid_from"),
                            "valid_to": r.get("valid_to"),
                            "is_expanded": True,
                        })
            print(f"  [N-Hop] Expanded {len(seed_entities)} seeds → {len(expanded)} neighbors ({max_hops}-hop)")
            return expanded
        except Exception as e:
            print(f"  [N-Hop] Expansion failed: {e}")
            return []

    async def summarize_entity_context(
        self,
        graph_context: list[tuple[str, str]],
        expanded_entities: list[dict],
    ) -> list[tuple[str, str]]:
        """
        Convert raw KG triplets + expanded entity data into natural-language
        entity profiles using the LLM.

        Returns a list of (summary_text, entity_label) tuples.
        """
        if not graph_context and not expanded_entities:
            return []

        # Build a combined profile block for the LLM
        profile_parts = []

        # From direct graph retrieval
        for text, source in graph_context:
            profile_parts.append(f"[Direct] {text} (Source: {source})")

        # From N-hop expanded entities
        for ent in expanded_entities:
            title = ent.get("title", ent.get("name", "?"))
            labels = [l for l in ent.get("labels", []) if l not in ("__Entity__", "__Node__")]
            label_str = f" ({', '.join(labels)})" if labels else ""
            rel_str = ", ".join(ent.get("rel_types", []))
            temporal = ""
            if ent.get("valid_from") or ent.get("valid_to"):
                vf = ent.get("valid_from", "?")
                vt = ent.get("valid_to", "present")
                temporal = f" [{vf} – {vt}]"
            profile_parts.append(
                f"[Related] {title}{label_str} — connected via: {rel_str}{temporal}"
            )

        if not profile_parts:
            return graph_context  # nothing to summarize

        combined = "\n".join(profile_parts[:30])  # cap to avoid exceeding context

        prompt = (
            "You are a Knowledge Graph summarizer. Given the following raw graph data "
            "(both direct facts and related/expanded context), produce a concise, "
            "readable summary grouped by entity.\n\n"
            "For each entity mentioned, write 1-3 sentences describing what it is, "
            "its key relationships, and any temporal context (dates). "
            "Clearly separate direct facts from expanded/related context.\n\n"
            f"--- GRAPH DATA ---\n{combined}\n\n"
            "Write the summary now. Keep it under 300 words. Do NOT include any preamble."
        )

        try:
            response = await self.llm.acomplete(prompt)
            summary_text = response.text.strip()
            # Return as a single summarized block
            return [(summary_text, "Knowledge Graph (Summarized)")]
        except Exception as e:
            print(f"  [Entity Summary] LLM summarization failed: {e}")
            return graph_context  # fallback to raw context

    def hybrid_graph_traversal(self, top_fused: list[dict], max_nodes: int = 3) -> dict:
        """
        Given the top fused document/chunk results, attempt to find
        connected entities and structural neighbors in the graph.
        
        This mimics a "local" expansion from hit points.
        """
        # Placeholder implementation for now: return empty context
        # (This previously required specific relationship labels (Chunk)-[MENTIONS]->(Entity))
        return {"chunks": [], "entities": []}
