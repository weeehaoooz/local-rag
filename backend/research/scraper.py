import trafilatura
import os
import re
from typing import Dict, Optional

class WebScraper:
    def __init__(self, data_dir: str = "data/web_research"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def _sanitize_filename(self, filename: str) -> str:
        # Keep alphanumeric, space, dot, and replace others with underscore
        return re.sub(r'[^\w\s\.-]', '_', filename).strip()

    def scrape_to_file(self, result: Dict) -> Optional[str]:
        """
        Scrape a URL and save its main content to a .md file.
        """
        url = result.get('link') or result.get('url')
        if not url:
            return None
            
        try:
            # Download and extract content
            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                return None
                
            # extract() can output markdown if specified. 
            # include_links=True and include_tables=True helps keep "schematic reasoning"
            content = trafilatura.extract(downloaded, output_format="markdown", include_links=True, include_tables=True)
            
            if not content:
                return None
                
            # Create a safe filename from the title
            safe_title = self._sanitize_filename(result['title'])[:80]
            # Use a hash or ID if link exists to avoid collisions
            url_hash = str(hash(url))[-6:]
            filename = f"{safe_title}_{url_hash}.md"
            filepath = os.path.join(self.data_dir, filename)
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"# {result['title']}\n\n")
                f.write(f"**SOURCE:** [{url}]({url})\n\n")
                if result.get('date'):
                    f.write(f"**DATE:** {result['date']}\n\n")
                f.write("---\n\n")
                f.write(content)
                
            return filepath
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            return None
