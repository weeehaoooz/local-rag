        async def chat_stream_status_async(
            self,
            user_message: str,
            mode: str = "fast",
            history: list = None,
            system_prompt: str = "",
            max_reflection_loops: int = 2,
        ):
            import json
            import time
            import asyncio
            
            def _yield_status(msg: str, tokens: int = 0):
                return "data: " + json.dumps({"type": "status", "message": msg, "tokens": tokens}) + "\n\n"
                
            def _yield_final(data: dict):
                return "data: " + json.dumps({"type": "done", "response": data}) + "\n\n"
    
            yield _yield_status("Resolving coreferences...")
            history = history or []
            resolved_query = self.transformer.resolve_coreference(user_message, history)
    
            yield _yield_status("Decomposing queries...")
            sub_queries = self.decomposer.split_query(resolved_query)
    
            all_context = {
                "graph_context": [], "vector_context": [], "summary_context": [],
                "community_context": [], "sources": [], "tools_used": [],
                "orchestrator_rationale": f"Decomposed {len(sub_queries)} queries.",
                "query_type": None
            }
            all_context_chunks = []
            reflection_loops = 0
            retrieval_grade_result = "pass"
            answer_grade_result = "grounded"
    
            for sq in sub_queries:
                yield _yield_status(f"Retrieving context for: {sq[:30]}...")
                sq_context = await self.get_context_async(sq)
                sq_chunks = self._collect_context_texts(sq_context)
    
                for loop_idx in range(max_reflection_loops):
                    yield _yield_status(f"Grading retrieval (Loop {loop_idx+1})")
                    r_grade = self.reflection.grade_retrieval(sq, sq_chunks)
                    
                    if r_grade.relevant:
                        break
                    
                    retrieval_grade_result = "fail"
                    reflection_loops += 1
    
                    if loop_idx < max_reflection_loops - 1:
                        yield _yield_status(f"Rewriting query...")
                        sq = self.reflection.rewrite_query(sq, r_grade.reason)
                        yield _yield_status(f"Re-retrieving context...")
                        sq_context = await self.get_context_async(sq)
                        sq_chunks = self._collect_context_texts(sq_context)
                    else:
                        break
    
                all_context["graph_context"].extend(sq_context.get("graph_context", []))
                all_context["vector_context"].extend(sq_context.get("vector_context", []))
                all_context["summary_context"].extend(sq_context.get("summary_context", []))
                all_context["community_context"].extend(sq_context.get("community_context", []))
                for src in sq_context.get("sources", []):
                    if src not in all_context["sources"]:
                        all_context["sources"].append(src)
                for t in sq_context.get("tools_used", []):
                    if t not in all_context["tools_used"]:
                        all_context["tools_used"].append(t)
                if not all_context["query_type"]:
                    all_context["query_type"] = sq_context.get("query_type")
                all_context_chunks.extend(sq_chunks)
    
            full_prompt = self._build_prompt(all_context, user_message, system_prompt)
            
            # Generation phase
            yield _yield_status("Generating answer...", tokens=0)
            start_t = time.time()
            
            # Async stream processing
            loop = asyncio.get_event_loop()
            response_gen = await self.llm.astream_complete(full_prompt)
            answer_text = ""
            token_count = 0
            async for chunk in response_gen:
                answer_text += chunk.delta
                token_count += 1
                if token_count % 5 == 0:
                    yield _yield_status("Generating answer...", tokens=token_count)
            
            end_t = time.time()
            ans_duration = end_t - start_t
            tps = token_count / ans_duration if ans_duration > 0 else 0.0
    
            yield _yield_status("Evaluating groundedness...")
            a_grade = self.reflection.grade_answer(user_message, all_context_chunks, answer_text)
    
            if not a_grade.grounded and reflection_loops < max_reflection_loops:
                answer_grade_result = "ungrounded"
                reflection_loops += 1
                yield _yield_status("Answer ungrounded, correcting...")
    
                corrective_query = self.reflection.rewrite_query(
                    user_message,
                    failure_reason=f"The answer was ungrounded. Reason: {a_grade.reason}."
                )
                corrective_context = await self.get_context_async(corrective_query)
                corrective_chunks = self._collect_context_texts(corrective_context)
                corrective_prompt = self._build_prompt(corrective_context, user_message, system_prompt)
                
                yield _yield_status("Generating correction...", tokens=0)
                cor_gen = await self.llm.astream_complete(corrective_prompt)
                cor_text = ""
                cor_tokens = 0
                async for chunk in cor_gen:
                    cor_text += chunk.delta
                    cor_tokens += 1
                    if cor_tokens % 5 == 0:
                        yield _yield_status("Generating correction...", tokens=cor_tokens)
                
                final_a_grade = self.reflection.grade_answer(user_message, corrective_chunks, cor_text)
    
                if final_a_grade.grounded:
                    answer_text = cor_text
                    all_context = corrective_context
                    answer_grade_result = "grounded"
            else:
                answer_grade_result = "grounded" if a_grade.grounded else "ungrounded"
    
            final_response = {
                "response": answer_text,
                "sources": all_context["sources"],
                "query_type": all_context["query_type"].value if all_context.get("query_type") else "HYBRID",
                "suggested_prompts": self.get_suggestions(),
                "reflection_loops": reflection_loops,
                "retrieval_grade": retrieval_grade_result,
                "answer_grade": answer_grade_result,
                "tools_used": all_context.get("tools_used", []),
                "orchestrator_rationale": all_context.get("orchestrator_rationale", ""),
                "sub_queries": sub_queries,
                "stats": {
                    "tps": round(tps, 1),
                    "context_utilization": min(1.0, (len(full_prompt) // 4 + token_count) / 8192.0),
                    "prompt_tokens": len(full_prompt) // 4,
                    "completion_tokens": token_count,
                    "total_tokens": len(full_prompt) // 4 + token_count,
                    "context_window": 8192
                }
            }
            
            yield _yield_final(final_response)
