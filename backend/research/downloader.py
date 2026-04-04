import os
import re
import requests
from typing import Dict

class ResearchDownloader:
    def __init__(self, data_dir: str = "data/research"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def _sanitize_filename(self, filename: str) -> str:
        # Keep alphanumeric, space, dot, and replace others with underscore
        return re.sub(r'[^\w\s\.-]', '_', filename).strip()

    def download(self, paper: Dict) -> str:
        """
        Download the PDF for a paper.
        """
        try:
            safe_title = self._sanitize_filename(paper['title'])[:100]
            filename = f"{safe_title}_{paper['id']}.pdf"
            filepath = os.path.join(self.data_dir, filename)
            
            if os.path.exists(filepath):
                return filepath
                
            # Use arxiv result object's download_pdf if available, otherwise requests
            if paper.get('result_obj'):
                paper['result_obj'].download_pdf(dirpath=self.data_dir, filename=filename)
            else:
                response = requests.get(paper['pdf_url'], stream=True)
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
            
            return filepath
        except Exception as e:
            print(f"Error downloading {paper['title']}: {e}")
            return None
