"""
indexers/community.py
---------------------
CommunitySummarizer: generates LLM summaries for detected graph communities
and stores them as (:CommunitySummary) nodes in Neo4j.

Each CommunitySummary node contains:
  - community_id (int)
  - summary (str)       — LLM-generated description of the community
  - entity_count (int)  — number of entities in the community
  - key_entities (str)  — comma-separated list of top entity names

These nodes are linked to their constituent entities via:
  (entity)-[:MEMBER_OF]->(community_summary)
"""

from __future__ import annotations

import os
from typing import List, Optional

from llama_index.core import Settings
from llama_index.graph_stores.neo4j import Neo4jPropertyGraphStore


class CommunitySummarizer:
    """
    Generate and store LLM summaries for Knowledge Graph communities.

    Usage::

        summarizer = CommunitySummarizer(graph_store)
        summarizer.summarize_and_store(
            community_id=0,
            entity_names=["Apple Inc.", "Steve Jobs", "Cupertino"],
            relationships=["Apple Inc. --[FOUNDED_BY]--> Steve Jobs", ...],
        )
    """

    def __init__(
        self,
        graph_store: Neo4jPropertyGraphStore,
        llm=None,
    ):
        self.graph_store = graph_store
        self.llm = llm or Settings.llm

    def summarize_and_store(
        self,
        community_id: int,
        entity_names: List[str],
        relationships: List[str],
        max_entities_in_prompt: int = 50,
        max_rels_in_prompt: int = 80,
    ) -> str:
        """
        Generate a summary for a community and persist it in Neo4j.

        Parameters
        ----------
        community_id : int
            The numeric community identifier.
        entity_names : list[str]
            Names of all entities in this community.
        relationships : list[str]
            Human-readable relationship descriptions (e.g. "A --[REL]--> B").
        max_entities_in_prompt : int
            Cap on entities to include in the LLM prompt.
        max_rels_in_prompt : int
            Cap on relationships to include in the LLM prompt.

        Returns
        -------
        str
            The generated summary.
        """
        # Deduplicate
        unique_entities = list(dict.fromkeys(entity_names))[:max_entities_in_prompt]
        unique_rels = list(dict.fromkeys(relationships))[:max_rels_in_prompt]

        entities_str = ", ".join(unique_entities)
        rels_str = "\n".join(f"  - {r}" for r in unique_rels)

        prompt = (
            "You are a Knowledge Graph analyst. Below is a cluster (community) of "
            "related entities and their relationships extracted from a knowledge graph.\n\n"
            f"ENTITIES ({len(unique_entities)}): {entities_str}\n\n"
            f"RELATIONSHIPS:\n{rels_str}\n\n"
            "Write a concise but comprehensive summary (150-250 words) that:\n"
            "1. Identifies the main theme or domain this community represents.\n"
            "2. Lists the most important entities and their roles.\n"
            "3. Describes the key relationships and how entities are connected.\n"
            "4. Notes any interesting patterns or hierarchies.\n\n"
            "Do NOT include any preamble. Start directly with the summary."
        )

        try:
            response = self.llm.complete(prompt)
            summary = response.text.strip()
        except Exception as e:
            print(f"     Warning: LLM summarization failed for community {community_id}: {e}")
            summary = f"Community of {len(unique_entities)} entities: {entities_str[:200]}"

        # Key entities = first 10
        key_entities = ", ".join(unique_entities[:10])

        # Store in Neo4j
        self._upsert_community_node(
            community_id=community_id,
            summary=summary,
            entity_count=len(unique_entities),
            key_entities=key_entities,
        )

        # Link entities to the community summary node
        self._link_entities_to_community(community_id)

        print(
            f"     Community {community_id}: {len(unique_entities)} entities, "
            f"{len(unique_rels)} rels — summarized ({len(summary)} chars)"
        )
        return summary

    def _upsert_community_node(
        self,
        community_id: int,
        summary: str,
        entity_count: int,
        key_entities: str,
    ):
        """Create or update a (:CommunitySummary) node in Neo4j."""
        query = """
        MERGE (cs:CommunitySummary {community_id: $cid})
        SET cs.summary = $summary,
            cs.entity_count = $entity_count,
            cs.key_entities = $key_entities,
            cs.name = 'Community ' + toString($cid)
        """
        try:
            self.graph_store.structured_query(
                query,
                param_map={
                    "cid": community_id,
                    "summary": summary,
                    "entity_count": entity_count,
                    "key_entities": key_entities,
                },
            )
        except Exception as e:
            print(f"     Warning: Failed to upsert CommunitySummary {community_id}: {e}")

    def _link_entities_to_community(self, community_id: int):
        """
        Create (entity)-[:MEMBER_OF]->(CommunitySummary) edges for all entities
        in this community.
        """
        query = """
        MATCH (n {community_id: $cid}), (cs:CommunitySummary {community_id: $cid})
        WHERE NOT n:CommunitySummary
          AND NOT (n)-[:MEMBER_OF]->(cs)
        MERGE (n)-[:MEMBER_OF]->(cs)
        """
        try:
            self.graph_store.structured_query(
                query, param_map={"cid": community_id}
            )
        except Exception as e:
            print(f"     Warning: Failed to link entities to community {community_id}: {e}")

    def get_all_summaries(self) -> list[dict]:
        """Retrieve all community summaries from Neo4j."""
        query = """
        MATCH (cs:CommunitySummary)
        RETURN cs.community_id AS community_id,
               cs.summary AS summary,
               cs.entity_count AS entity_count,
               cs.key_entities AS key_entities
        ORDER BY cs.entity_count DESC
        """
        results = self.graph_store.structured_query(query)
        summaries = []
        if isinstance(results, list):
            for r in results:
                if isinstance(r, dict):
                    summaries.append(r)
                elif hasattr(r, "values"):
                    vals = list(r.values())
                    summaries.append({
                        "community_id": vals[0],
                        "summary": vals[1],
                        "entity_count": vals[2],
                        "key_entities": vals[3],
                    })
        return summaries

    def get_relevant_summaries(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Retrieve the community summaries most relevant to a user query
        using embedding-based semantic search.

        Falls back to returning the largest communities if embedding fails.
        """
        all_summaries = self.get_all_summaries()
        if not all_summaries:
            return []

        embed_model = Settings.embed_model
        if embed_model is None:
            # Fallback: return largest communities
            return all_summaries[:top_k]

        try:
            import numpy as np

            query_emb = np.array(embed_model.get_text_embedding(query))
            summary_texts = [s.get("summary", "") for s in all_summaries]
            summary_embs = np.array([
                embed_model.get_text_embedding(text) for text in summary_texts
            ])

            norms = np.linalg.norm(summary_embs, axis=1)
            norms[norms == 0] = 1e-10
            query_norm = np.linalg.norm(query_emb)
            if query_norm == 0:
                query_norm = 1e-10

            similarities = summary_embs @ query_emb / (norms * query_norm)
            top_indices = np.argsort(similarities)[-top_k:][::-1]

            return [all_summaries[i] for i in top_indices]
        except Exception as e:
            print(f"     Warning: Semantic community search failed: {e}")
            return all_summaries[:top_k]
