import time
from ddgs import DDGS
from ddgs.exceptions import RatelimitException
from typing import List, Dict, Callable, Any


class WebSearcher:
    def __init__(self, max_results: int = 5):
        self.max_results = max_results
        # Backoff delays (seconds) for successive rate-limit hits
        self._backoff_delays = [2, 4, 8]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _search_with_retry(self, ddgs: DDGS, method: Callable, **kwargs) -> List[Dict]:
        """
        Call a DDGS search method with exponential backoff on RatelimitException.
        Returns the raw result list, or [] if all retries are exhausted.
        """
        for attempt, delay in enumerate([0] + self._backoff_delays, start=1):
            if delay:
                time.sleep(delay)
            try:
                return list(method(**kwargs))
            except RatelimitException:
                if attempt > len(self._backoff_delays):
                    # Last attempt failed — give up on this query
                    return []
                # Otherwise loop and retry after the next delay
                continue
            except Exception as e:
                print(f"[WebSearcher] Unexpected error: {e}")
                return []
        return []

    # ------------------------------------------------------------------
    # Public search methods
    # ------------------------------------------------------------------

    def search_text(self, queries: List[str]) -> List[Dict]:
        """
        Perform a general web search for each query.
        """
        results: List[Dict] = []
        seen_links: set = set()

        with DDGS() as ddgs:
            for i, query in enumerate(queries):
                # Polite inter-query delay (skip before very first query)
                if i > 0:
                    time.sleep(1.0)

                raw = self._search_with_retry(
                    ddgs,
                    ddgs.text,
                    query=query,
                    max_results=self.max_results,
                )
                for r in raw:
                    if r["href"] not in seen_links:
                        results.append({
                            "title": r["title"],
                            "link": r["href"],
                            "snippet": r["body"],
                            "source": "web",
                        })
                        seen_links.add(r["href"])

        return results

    def search_news(self, queries: List[str]) -> List[Dict]:
        """
        Perform a news search (useful for company updates and recent events).
        """
        results: List[Dict] = []
        seen_links: set = set()

        with DDGS() as ddgs:
            for i, query in enumerate(queries):
                if i > 0:
                    time.sleep(1.0)

                raw = self._search_with_retry(
                    ddgs,
                    ddgs.news,
                    query=query,
                    max_results=self.max_results,
                )
                for r in raw:
                    if r["url"] not in seen_links:
                        results.append({
                            "title": r["title"],
                            "link": r["url"],
                            "snippet": r["body"],
                            "date": r.get("date"),
                            "source": r.get("source", "news"),
                        })
                        seen_links.add(r["url"])

        return results

    def search_definitions(self, terms: List[str]) -> List[Dict]:
        """
        Perform a targeted search for definitions of specific technical terms.
        Only the top result per term is kept.
        """
        results: List[Dict] = []
        seen_links: set = set()

        with DDGS() as ddgs:
            for i, term in enumerate(terms):
                if i > 0:
                    time.sleep(1.0)

                query = f"{term} definition meaning"
                raw = self._search_with_retry(
                    ddgs,
                    ddgs.text,
                    query=query,
                    max_results=2,
                )
                for r in raw:
                    if r["href"] not in seen_links:
                        results.append({
                            "title": f"Definition: {term}",
                            "link": r["href"],
                            "snippet": r["body"],
                            "source": "dictionary",
                        })
                        seen_links.add(r["href"])

        return results

    def search_wikipedia(self, queries: List[str]) -> List[Dict]:
        """
        Targeted search for Wikipedia articles to get authoritative summaries.
        """
        results: List[Dict] = []
        seen_links: set = set()

        with DDGS() as ddgs:
            for i, query in enumerate(queries):
                if i > 0:
                    time.sleep(1.0)

                # Force wikipedia.org results
                wiki_query = f"site:wikipedia.org {query}"
                raw = self._search_with_retry(
                    ddgs,
                    ddgs.text,
                    query=wiki_query,
                    max_results=self.max_results,
                )
                for r in raw:
                    if r["href"] not in seen_links and "wikipedia.org" in r["href"]:
                        results.append({
                            "title": r["title"].replace(" - Wikipedia", ""),
                            "link": r["href"],
                            "snippet": r["body"],
                            "source": "wikipedia",
                        })
                        seen_links.add(r["href"])

        return results
