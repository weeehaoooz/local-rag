import arxiv
from typing import List, Dict

class ResearchSearcher:
    def __init__(self, max_results_per_query: int = 5):
        self.client = arxiv.Client()
        self.max_results = max_results_per_query

    def search(self, queries: List[str], sort_by: str = "relevance") -> List[Dict]:
        """
        Search ArXiv for each query and return a list of unique paper metadata.

        Args:
            queries:  List of search query strings.
            sort_by:  'relevance' (default) or 'date' (sorts by most recently submitted).
        """
        sort_criterion = (
            arxiv.SortCriterion.SubmittedDate
            if sort_by == "date"
            else arxiv.SortCriterion.Relevance
        )

        all_results: Dict[str, Dict] = {}

        for query in queries:
            search = arxiv.Search(
                query=query,
                max_results=self.max_results,
                sort_by=sort_criterion,
            )

            for result in self.client.results(search):
                if result.entry_id not in all_results:
                    all_results[result.entry_id] = {
                        "id": result.entry_id.split("/")[-1],
                        "title": result.title,
                        "summary": result.summary,
                        "pdf_url": result.pdf_url,
                        "authors": [a.name for a in result.authors],
                        "published": result.published.strftime("%Y-%m-%d"),
                        "source": "arxiv",           # ← explicit source tag
                        "result_obj": result,        # Keep for downloading
                    }

        return list(all_results.values())
