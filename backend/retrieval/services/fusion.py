class RankFusionService:
    @staticmethod
    def reciprocal_rank_fusion(list_of_ranked_lists: list[list[dict]], k: int = 60) -> list[dict]:
        """
        Fuses multiple ranked result lists using Reciprocal Rank Fusion (RRF).
        list_of_ranked_lists: A list containing lists of ranked items (nodes/chunks).
        """
        rrf_scores = {}
        item_mapping = {}

        for ranked_list in list_of_ranked_lists:
            if not ranked_list:
                continue
            for rank, item in enumerate(ranked_list):
                # Unique identifier based on text content and source
                item_id = f"{item['text']}|{item['source']}"
                
                if item_id not in rrf_scores:
                    rrf_scores[item_id] = 0.0
                    item_mapping[item_id] = item
                
                # RRF Formula
                rrf_scores[item_id] += 1.0 / (k + rank + 1)
        
        # Sort by RRF score descending
        sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return fused list of items
        fused_results = []
        for item_id, score in sorted_items:
            item = item_mapping[item_id].copy()
            item['score'] = score
            fused_results.append(item)
            
        return fused_results
