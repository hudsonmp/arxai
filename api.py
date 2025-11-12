import re
import time
import requests
import numpy as np
from typing import Dict, List, Optional
try:
    import pymupdf as fitz
except ImportError:
    import fitz
from cache import PaperCache

cache = PaperCache()

def rate_limited_request(url, params=None, max_retries=3):
    for attempt in range(max_retries):
        try:
            time.sleep(0.5)
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 429:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
                continue
            return response
        except requests.exceptions.Timeout:
            if attempt == max_retries - 1:
                raise
            time.sleep(1)
    return response

def extract_doi_from_pdf(pdf_path: str) -> Optional[str]:
    try:
        doc = fitz.open(pdf_path)
        first_page = doc[0]
        page_height = first_page.rect.height
        header_rect = fitz.Rect(0, 0, first_page.rect.width, 100)
        footer_rect = fitz.Rect(0, page_height - 100, first_page.rect.width, page_height)
        text = first_page.get_text("text", clip=header_rect) + "\n" + first_page.get_text("text", clip=footer_rect)
        doi_patterns = [r'doi:\s*(10\.\d+/[^\s]+)', r'DOI:\s*(10\.\d+/[^\s]+)', r'https?://doi\.org/(10\.\d+/[^\s]+)', r'\b(10\.\d+/[^\s]+)\b']
        for pattern in doi_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                doi = re.sub(r'[.,;:\)]$', '', match.group(1))
                doc.close()
                return doi
        doc.close()
    except:
        pass
    return None

def get_paper_metadata(paper_id: str) -> Dict:
    cached = cache.get_paper(paper_id)
    if cached and 'metadata' in cached:
        return cached['metadata']
    
    fields = "title,authors,venue,publicationDate,citationCount,referenceCount,influentialCitationCount,fieldsOfStudy,embedding,externalIds,abstract"
    url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}?fields={fields}"
    response = rate_limited_request(url)
    if response.status_code != 200:
        raise Exception(f"API error ({response.status_code}): {response.text[:100]}")
    
    data = response.json()
    embedding_data = data.get("embedding") or {}
    result = {
        "title": data.get("title"),
        "venue": data.get("venue"),
        "authors": [a.get("name", "") for a in data.get("authors", [])],
        "citation_count": data.get("citationCount"),
        "publication_date": data.get("publicationDate"),
        "fields_of_study": data.get("fieldsOfStudy"),
        "reference_count": data.get("referenceCount"),
        "influential_citation_count": data.get("influentialCitationCount"),
        "doi": data.get("externalIds", {}).get("DOI", paper_id),
        "embedding": embedding_data.get("vector") if embedding_data else [],
        "abstract": data.get("abstract", "")
    }
    
    cache.save_paper(paper_id, {"metadata": result, "vector": result["embedding"]})
    return result

def get_citing_papers(paper_id: str) -> List[str]:
    url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations?fields=paperId,externalIds"
    try:
        response = rate_limited_request(url)
        if response.status_code != 200:
            return []
        data = response.json()
        dois = []
        for citation in data.get('data', []):
            citing_paper = citation.get("citingPaper")
            if citing_paper:
                doi = citing_paper.get("externalIds", {}).get("DOI")
                paper_id = citing_paper.get("paperId")
                dois.append(doi or paper_id)
        return dois
    except:
        return []

def get_reference_papers(paper_id: str) -> List[str]:
    url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references?fields=paperId,externalIds"
    try:
        response = rate_limited_request(url)
        if response.status_code != 200:
            return []
        data = response.json()
        dois = []
        for reference in data.get('data', []):
            cited_paper = reference.get("citedPaper")
            if cited_paper:
                doi = cited_paper.get("externalIds", {}).get("DOI")
                paper_id = cited_paper.get("paperId")
                dois.append(doi or paper_id)
        return dois
    except:
        return []

def search_papers(query: str, limit: int = 10) -> List[Dict]:
    url = f"https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "fields": "paperId,title,authors,year,venue,citationCount,externalIds,abstract"
    }
    response = rate_limited_request(url, params=params)
    if response.status_code != 200:
        raise Exception(f"Search error ({response.status_code}): {response.text[:100]}")
    return response.json().get('data', [])

def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    if len(vec1) == 0 or len(vec2) == 0:
        return 0.0
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(dot_product / (norm1 * norm2))

