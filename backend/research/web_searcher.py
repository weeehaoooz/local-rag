import time
from ddgs import DDGS
from ddgs.exceptions import RatelimitException
from typing import List, Dict

class WebSearcher:
    def __init__(self, max_results: int = 5):
        self.max_results = max_results

    def search_text(self, queries: List[str]) -> List[Dict]:
        """
        Perform a general web search.
        """
        results = []
        seen_links = set()
        
        with DDGS() as ddgs:
            for query in queries:
                try:
                    ddgs_gen = ddgs.text(query, max_results=self.max_results)
                    for r in ddgs_gen:
                        if r['href'] not in seen_links:
                            results.append({
                                "title": r['title'],
                                "link": r['href'],
                                "snippet": r['body'],
                                "source": "web"
                            })
                            seen_links.add(r['href'])
                    # Add delay between different queries
                    time.sleep(1.0)
                except RatelimitException:
                    print(f"Rate limited by DuckDuckGo for query: {query}")
                    continue
                except Exception as e:
                    print(f"Error searching Web: {e}")
                    continue
        return results

    def search_news(self, queries: List[str]) -> List[Dict]:
        """
        Perform a news search (useful for company updates).
        """
        results = []
        seen_links = set()
        
        with DDGS() as ddgs:
            for query in queries:
                try:
                    ddgs_gen = ddgs.news(query, max_results=self.max_results)
                    for r in ddgs_gen:
                        if r['url'] not in seen_links:
                            results.append({
                                "title": r['title'],
                                "link": r['url'],
                                "snippet": r['body'],
                                "date": r.get('date'),
                                "source": r.get('source', 'news')
                            })
                            seen_links.add(r['url'])
                    # Add delay between different queries
                    time.sleep(1.0)
                except RatelimitException:
                    print(f"Rate limit hit for query: {query}")
                    continue
                except Exception as e:
                    print(f"Error searching News: {e}")
                    continue
        return results

    def search_definitions(self, terms: List[str]) -> List[Dict]:
        """
        Perform a targeted search for definitions of specific terms.
        """
        results = []
        seen_links = set()
        
        with DDGS() as ddgs:
            for term in terms:
                query = f"{term} definition meaning"
                try:
                    # For definitions, we mostly want the top result
                    ddgs_gen = ddgs.text(query, max_results=2)
                    for r in ddgs_gen:
                        if r['href'] not in seen_links:
                            results.append({
                                "title": f"Definition: {term}",
                                "link": r['href'],
                                "snippet": r['body'],
                                "source": "dictionary"
                            })
                            seen_links.add(r['href'])
                    # Add delay
                    time.sleep(1.0)
                except Exception as e:
                    print(f"Error searching definitions for {term}: {e}")
                    continue
        return results
