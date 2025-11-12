import os
import requests
from pathlib import Path

def download_paper(paper_id: str, title: str, papers_dir: str = "papers") -> bool:
    Path(papers_dir).mkdir(exist_ok=True)
    
    safe_title = "".join(c for c in title[:50] if c.isalnum() or c in (' ', '-', '_')).strip()
    filename = f"{safe_title}.pdf"
    filepath = Path(papers_dir) / filename
    
    if filepath.exists():
        return True
    
    urls = [
        f"https://arxiv.org/pdf/{paper_id}.pdf",
        f"https://www.semanticscholar.org/paper/{paper_id}",
    ]
    
    for url in urls:
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200 and response.headers.get('content-type') == 'application/pdf':
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                return True
        except:
            continue
    
    return False

