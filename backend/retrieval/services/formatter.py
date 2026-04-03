import os

class ContextFormatter:
    @staticmethod
    def extract_sources(nodes: list) -> list[dict]:
        sources = []
        seen = set()
        
        for node in nodes:
            try:
                # Meta source extraction logic
                meta = getattr(node, "metadata", {})
                if not meta and hasattr(node, "node"):
                    meta = getattr(node.node, "metadata", {})
                
                title = meta.get("title", meta.get("file_name", ""))
                file_path = meta.get("file_path", "")
                if not title and file_path:
                    title = os.path.basename(file_path).rsplit(".", 1)[0]
                
                category = meta.get("category", "General")
                
                identifier = f"{title}|{category}"
                
                if identifier not in seen and title:
                    seen.add(identifier)
                    sources.append({
                        "title": title,
                        "category": category,
                        "file": os.path.basename(file_path) if file_path else ""
                    })
            except Exception:
                continue
                
        return sources
