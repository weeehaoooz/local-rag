import re

with open("retrieval/engine.py", "r") as f:
    content = f.read()

start_marker = "        current_query = user_message\n        reflection_loops = 0\n        retrieval_grade_result = \"pass\"\n        answer_grade_result = \"grounded\""
end_marker = "        else:\n            answer_grade_result = \"grounded\" if a_grade.grounded else \"ungrounded\""

start_idx = content.find(start_marker)
end_idx = content.find(end_marker) + len(end_marker)

if start_idx == -1 or end_idx == -1:
    print("Could not find markers!")
    exit(1)

new_content = """        reflection_loops = 0
        retrieval_grade_result = "pass"
        answer_grade_result = "grounded"

        # 1. Transform: Coreference Resolution
        history = history or []
        resolved_query = self.transformer.resolve_coreference(user_message, history)
        
        # 2. Decompose: Split compound queries
        sub_queries = self.decomposer.split_query(resolved_query)

        # Container for pooled contexts
        all_context = {
            "graph_context": [], "vector_context": [], "summary_context": [], 
            "community_context": [], "sources": [], "tools_used": [],
            "orchestrator_rationale": f"Decomposed {len(sub_queries)} queries.",
            "query_type": None
        }
        all_context_chunks = []

        # ── Retrieval → Grade → (Rewrite + Retry) loop ───────────────────
        for sq in sub_queries:
            print(f"  [Agent] Processing sub-query: {sq}")
            sq_context = self.get_context(sq)
            sq_chunks = self._collect_context_texts(sq_context)

            # Grading loop per sub-query
            for loop in range(max_reflection_loops):
                r_grade = self.reflection.grade_retrieval(sq, sq_chunks)
                print(f"  [Reflection] Retrieval grade for '{sq}' (loop {loop}): relevant={r_grade.relevant} — {r_grade.reason}")

                if r_grade.relevant:
                    break   # retrieval passed

                retrieval_grade_result = "fail"
                reflection_loops += 1

                if loop < max_reflection_loops - 1:
                    sq = self.reflection.rewrite_query(sq, r_grade.reason)
                    print(f"  [Reflection] Rewritten sub-query: {sq}")
                    sq_context = self.get_context(sq)
                    sq_chunks = self._collect_context_texts(sq_context)
                else:
                    print("  [Reflection] Max retrieval loops reached for sub-query.")

            # Merge results into pooled context
            all_context["graph_context"].extend(sq_context.get("graph_context", []))
            all_context["vector_context"].extend(sq_context.get("vector_context", []))
            all_context["summary_context"].extend(sq_context.get("summary_context", []))
            all_context["community_context"].extend(sq_context.get("community_context", []))
            
            # Deduplicate sources
            for src in sq_context.get("sources", []):
                if src not in all_context["sources"]:
                   all_context["sources"].append(src)
                   
            for t in sq_context.get("tools_used", []):
                if t not in all_context["tools_used"]:
                    all_context["tools_used"].append(t)
            
            if not all_context["query_type"]:
                all_context["query_type"] = sq_context.get("query_type")
                
            all_context_chunks.extend(sq_chunks)

        # ── Answer generation ────────────────────────────────────────────
        full_prompt = self._build_prompt(all_context, user_message, system_prompt)
        response = self.llm.complete(full_prompt)
        answer_text = response.text

        # ── Answer grounding check ───────────────────────────────────────
        a_grade = self.reflection.grade_answer(user_message, all_context_chunks, answer_text)
        print(f"  [Reflection] Answer grade: grounded={a_grade.grounded} — {a_grade.reason}")

        if not a_grade.grounded and reflection_loops < max_reflection_loops:
            answer_grade_result = "ungrounded"
            reflection_loops += 1
            print("  [Reflection] Answer ungrounded — attempting one corrective rewrite.")
            
            # Corrective retrieval directly targets the user's main requirement
            corrective_query = self.reflection.rewrite_query(
                user_message,
                failure_reason=f"The answer was ungrounded. Reason: {a_grade.reason}. Retrieve factual information to correct this."
            )
            corrective_context = self.get_context(corrective_query)
            corrective_chunks = self._collect_context_texts(corrective_context)

            # Generate new answer with corrective context
            corrective_prompt = self._build_prompt(corrective_context, user_message, system_prompt)
            corrective_response = self.llm.complete(corrective_prompt)
            
            # Final grade check
            final_a_grade = self.reflection.grade_answer(user_message, corrective_chunks, corrective_response.text)
            
            if final_a_grade.grounded:
                answer_text = corrective_response.text
                all_context = corrective_context  # Update visible logic to use the corrective sources
                answer_grade_result = "grounded"
        else:
            answer_grade_result = "grounded" if a_grade.grounded else "ungrounded"
"""

with open("retrieval/engine.py", "w") as f:
    f.write(content[:start_idx] + new_content + content[end_idx:])

print("Successfully replaced chat block.")
