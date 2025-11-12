#!/usr/bin/env python3
import sys
import os
import re
import textwrap
import numpy as np
from typing import List, Dict, Optional
from api import extract_doi_from_pdf, get_paper_metadata, search_papers, cosine_similarity
from search import Paper, search_level_1, search_level_2
from download import download_paper

def show_carousel(papers):
    if not papers:
        return None
    
    idx = 0
    use_raw = True
    
    try:
        import termios
        import tty
    except ImportError:
        use_raw = False
    
    while True:
        os.system('clear' if os.name == 'posix' else 'cls')
        
        print(f"\n{'='*80}")
        print(f"Select Paper ({idx+1}/{len(papers)})")
        print(f"{'='*80}\n")
        
        paper = papers[idx]
        print(f"Title: {paper.get('title', 'Unknown')}")
        print(f"\nAuthors: {', '.join([a.get('name', '') for a in paper.get('authors', [])])}")
        print(f"Year: {paper.get('year', 'N/A')} | Venue: {paper.get('venue', 'N/A')}")
        print(f"Citations: {paper.get('citationCount', 0)}")
        
        abstract = paper.get('abstract', '')
        if abstract:
            print(f"\n{'─'*80}")
            print("Abstract:")
            print(f"{'─'*80}")
            wrapped = wrap_text(abstract, width=76)
            lines = wrapped.split('\n')
            for line in lines[:12]:
                print(f"  {line}")
            if len(lines) > 12:
                print(f"  ... ({len(lines) - 12} more lines)")
        
        print(f"\n{'='*80}")
        if use_raw:
            print("[←/→] Navigate | [Enter] Select | [q] Quit")
        else:
            print("[n/p] Navigate | [Enter] Select | [q] Quit")
        
        if use_raw:
            try:
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                try:
                    tty.setraw(fd)
                    char = sys.stdin.read(1)
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                
                if char == '\x03' or char == 'q':
                    return None
                elif char == '\r':
                    return paper
                elif char == '\x1b':
                    next_chars = sys.stdin.read(2)
                    if next_chars == '[C':
                        idx = (idx + 1) % len(papers)
                    elif next_chars == '[D':
                        idx = (idx - 1) % len(papers)
            except:
                use_raw = False
        
        if not use_raw:
            try:
                choice = input("\n> ").strip().lower()
                if choice == 'q':
                    return None
                elif choice == '' or choice == 'y':
                    return paper
                elif choice == 'n':
                    idx = (idx + 1) % len(papers)
                elif choice == 'p':
                    idx = (idx - 1) % len(papers)
                elif choice.isdigit():
                    new_idx = int(choice) - 1
                    if 0 <= new_idx < len(papers):
                        idx = new_idx
            except (EOFError, KeyboardInterrupt):
                return None

def fuzzy_match(str1: str, str2: str, threshold: float = 0.7) -> bool:
    """Simple fuzzy matching for misspelling tolerance"""
    str1_lower = str1.lower()
    str2_lower = str2.lower()
    
    if str1_lower in str2_lower or str2_lower in str1_lower:
        return True
    
    if len(str1_lower) < 3 or len(str2_lower) < 3:
        return str1_lower == str2_lower
    
    common_chars = sum(1 for c in str1_lower if c in str2_lower)
    similarity = common_chars / max(len(str1_lower), len(str2_lower))
    return similarity >= threshold

def compute_initial_search_score(
    paper_data: Dict, 
    query: Optional[str], 
    query_embedding: Optional[np.ndarray] = None, 
    author_name: Optional[str] = None,
    topic: Optional[str] = None,
    title_filter: Optional[str] = None,
    institution: Optional[str] = None,
    max_citations: int = 1
) -> float:
    title = (paper_data.get('title') or '').lower()
    abstract = (paper_data.get('abstract') or '').lower()
    query_lower = (query or '').lower()
    query_words = set(query_lower.split()) if query_lower else set()
    citation_count = paper_data.get('citationCount', 0) or 0
    
    title_score = 0.0
    keyword_score = 0.0
    abstract_score = 0.0
    author_score = 0.0
    topic_score = 0.0
    title_filter_score = 0.0
    institution_score = 0.0
    citation_score = 0.0
    
    title_words = set(title.split())
    abstract_words = set(abstract.split())
    
    if query_words:
        title_matches = len(query_words.intersection(title_words))
        keyword_matches = len(query_words.intersection(abstract_words))
        
        if title_matches > 0:
            title_score = min(1.0, title_matches / max(len(query_words), 1)) * 0.2
        
        if keyword_matches > 0:
            keyword_score = min(1.0, keyword_matches / max(len(query_words), 1)) * 0.3
    
    if query_embedding is not None:
        try:
            paper_id = paper_data.get('externalIds', {}).get('DOI') or paper_data.get('paperId')
            if paper_id:
                metadata = get_paper_metadata(paper_id)
                paper_vector = np.array(metadata.get('embedding', []))
                if len(paper_vector) > 0:
                    abstract_score = cosine_similarity(paper_vector, query_embedding) * 0.2
        except:
            pass
    
    if author_name:
        authors = [a.get('name', '') for a in paper_data.get('authors', []) if a and a.get('name')]
        author_lower = author_name.lower()
        author_matched = False
        for author in authors:
            if author and fuzzy_match(author_lower, author.lower()):
                author_score = 0.85
                author_matched = True
                break
        if not author_matched:
            author_score = -0.5
    
    if topic:
        topic_lower = topic.lower()
        topic_words = set(topic_lower.split())
        title_topic_matches = len(topic_words.intersection(title_words))
        abstract_topic_matches = len(topic_words.intersection(abstract_words))
        if title_topic_matches > 0 or abstract_topic_matches > 0:
            topic_score = min(1.0, (title_topic_matches + abstract_topic_matches) / max(len(topic_words), 1)) * 0.85
        else:
            topic_score = -0.5
    
    if title_filter:
        title_filter_lower = title_filter.lower()
        if fuzzy_match(title_filter_lower, title):
            title_filter_score = 0.85
        else:
            title_filter_score = -0.5
    
    if institution:
        venue = (paper_data.get('venue') or '').lower()
        institution_lower = institution.lower()
        if fuzzy_match(institution_lower, venue):
            institution_score = 0.85
        else:
            institution_score = -0.5
    
    if max_citations > 0:
        citation_score = min(1.0, citation_count / max(max_citations, 1)) * 0.1
    
    return title_score + keyword_score + abstract_score + author_score + topic_score + title_filter_score + institution_score + citation_score

def get_paper_from_input(
    input_str: Optional[str], 
    author_name: Optional[str] = None,
    topic: Optional[str] = None,
    title_filter: Optional[str] = None,
    institution: Optional[str] = None
) -> Paper:
    if not input_str:
        query_parts = []
        if author_name:
            query_parts.append(author_name)
        if topic:
            query_parts.append(topic)
        if title_filter:
            query_parts.append(title_filter)
        if institution:
            query_parts.append(institution)
        input_str = " ".join(query_parts) if query_parts else "research papers"
    
    if os.path.exists(input_str) and input_str.lower().endswith('.pdf'):
        doi = extract_doi_from_pdf(input_str)
        if not doi:
            raise ValueError(f"DOI not found in PDF: {input_str}")
    elif re.match(r'^10\.\d+/', input_str):
        doi = input_str
    else:
        print("\nSearching Semantic Scholar...")
        results = search_papers(input_str, limit=50)
        if not results:
            raise ValueError(f"No papers found for query: {input_str}")
        
        query_embedding = None
        try:
            top_result = results[0]
            top_paper_id = top_result.get('externalIds', {}).get('DOI') or top_result.get('paperId')
            if top_paper_id:
                top_metadata = get_paper_metadata(top_paper_id)
                top_vector = np.array(top_metadata.get('embedding', []))
                if len(top_vector) > 0:
                    query_embedding = top_vector
        except:
            pass
        
        max_citations = max([r.get('citationCount', 0) or 0 for r in results], default=1)
        
        scored_results = []
        for result in results:
            score = compute_initial_search_score(
                result, input_str, query_embedding, author_name, 
                topic, title_filter, institution, max_citations
            )
            scored_results.append((score, result))
        
        scored_results.sort(key=lambda x: x[0], reverse=True)
        results = [result for _, result in scored_results][:10]
        
        selected = show_carousel(results)
        if not selected:
            raise ValueError("No paper selected")
        
        paper_id = selected.get('externalIds', {}).get('DOI') or selected.get('paperId')
        metadata = get_paper_metadata(paper_id)
        return Paper(paper_id, metadata)
    
    metadata = get_paper_metadata(doi)
    return Paper(doi, metadata)

def display_papers(papers, title, max_show=20):
    print(f"\n{'='*80}")
    print(title)
    print(f"{'='*80}\n")
    for i, paper in enumerate(papers[:max_show], 1):
        print(f"{i}. [R:{paper.relevance_score:.3f}] {paper.title[:65]}")
        venue_short = paper.venue[:35] if paper.venue else 'Unknown'
        year = paper.publication_date[:4] if paper.publication_date else 'N/A'
        print(f"   {venue_short} ({year}) | {paper.citation_count} cites")
        print()

def wrap_text(text, width=76):
    return '\n'.join(textwrap.fill(line, width=width) for line in text.split('\n'))

def show_paper_carousel(papers):
    if not papers:
        return []
    
    idx = 0
    selected = [False] * len(papers)
    use_raw = True
    
    try:
        import termios
        import tty
    except ImportError:
        use_raw = False
    
    while True:
        os.system('clear' if os.name == 'posix' else 'cls')
        
        paper = papers[idx]
        checkbox = "☑" if selected[idx] else "☐"
        
        print(f"\n{'='*80}")
        print(f"Level-1 Results ({idx+1}/{len(papers)}) {checkbox}")
        print(f"{'='*80}\n")
        
        print(f"Title: {paper.title[:75]}")
        print(f"\nRelevance: {paper.relevance_score:.4f}")
        authors_str = ', '.join(paper.authors[:3])
        if len(paper.authors) > 3:
            authors_str += f" +{len(paper.authors)-3} more"
        print(f"Authors: {authors_str}")
        year = paper.publication_date[:4] if paper.publication_date else 'N/A'
        print(f"Venue: {paper.venue[:40]} | Year: {year} | Citations: {paper.citation_count}")
        
        print(f"\n{'─'*80}")
        print("Abstract:")
        print(f"{'─'*80}")
        
        if hasattr(paper, 'abstract') and paper.abstract:
            abstract_lines = wrap_text(paper.abstract, width=76).split('\n')
            for line in abstract_lines[:12]:
                print(f"  {line}")
            if len(abstract_lines) > 12:
                print(f"  ... ({len(abstract_lines) - 12} more lines)")
        else:
            print("  (No abstract available)")
        
        print(f"\n{'='*80}")
        if use_raw:
            print(f"[←/→] Navigate | [Space] Toggle | [Enter] Dive | [q] Quit")
        else:
            print(f"[n/p] Navigate | [t] Toggle | [Enter] Done | [q] Quit")
        print(f"Selected: {sum(selected)}/{len(papers)}")
        
        if use_raw:
            try:
                fd = sys.stdin.fileno()
                old_settings = termios.tcgetattr(fd)
                try:
                    tty.setraw(fd)
                    char = sys.stdin.read(1)
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                
                if char == '\x03' or char == 'q':
                    return []
                elif char == ' ':
                    selected[idx] = not selected[idx]
                elif char == '\r' or char == '\n':
                    return [papers[i] for i, sel in enumerate(selected) if sel]
                elif char == '\x1b':
                    next_chars = sys.stdin.read(2)
                    if next_chars == '[C' or next_chars == '[6':
                        idx = (idx + 1) % len(papers)
                    elif next_chars == '[D' or next_chars == '[5':
                        idx = (idx - 1) % len(papers)
            except:
                use_raw = False
        
        if not use_raw:
            try:
                choice = input("\n> ").strip().lower()
                if choice == 'q':
                    return []
                elif choice == 't' or choice == ' ':
                    selected[idx] = not selected[idx]
                elif choice == 'n':
                    idx = (idx + 1) % len(papers)
                elif choice == 'p':
                    idx = (idx - 1) % len(papers)
                elif choice == '':
                    return [papers[i] for i, sel in enumerate(selected) if sel]
                elif choice.isdigit():
                    new_idx = int(choice) - 1
                    if 0 <= new_idx < len(papers):
                        idx = new_idx
            except (EOFError, KeyboardInterrupt):
                return []

def main():
    if len(sys.argv) < 2:
        print("Usage: arxai [\"pdf_path|doi|search_query\"] [--author \"Author\"] [--topic \"Topic\"] [--title \"Title\"] [--institution \"Institution\"]")
        sys.exit(1)
    
    input_str = None
    author_name = None
    topic = None
    title_filter = None
    institution = None
    
    i = 1
    while i < len(sys.argv):
        if sys.argv[i].startswith('--'):
            if sys.argv[i] == '--author' and i + 1 < len(sys.argv):
                author_name = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == '--topic' and i + 1 < len(sys.argv):
                topic = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == '--title' and i + 1 < len(sys.argv):
                title_filter = sys.argv[i + 1]
                i += 2
            elif sys.argv[i] == '--institution' and i + 1 < len(sys.argv):
                institution = sys.argv[i + 1]
                i += 2
            else:
                i += 1
        else:
            if input_str is None:
                input_str = sys.argv[i]
            i += 1
    
    print(f"\n{'='*80}")
    print("arxAI - Deep Research Assistant")
    print(f"{'='*80}\n")
    
    filters = []
    if author_name:
        filters.append(f"Author: {author_name}")
    if topic:
        filters.append(f"Topic: {topic}")
    if title_filter:
        filters.append(f"Title: {title_filter}")
    if institution:
        filters.append(f"Institution: {institution}")
    
    if not input_str and not filters:
        print("Usage: arxai [\"pdf_path|doi|search_query\"] [--author \"Author\"] [--topic \"Topic\"] [--title \"Title\"] [--institution \"Institution\"]")
        print("Error: Must provide either a query/path/DOI or at least one filter")
        sys.exit(1)
    
    if filters:
        print(f"Filters (weighted, not restrictive): {', '.join(filters)}")
    
    print("Loading seed paper...")
    seed_paper = get_paper_from_input(input_str, author_name, topic, title_filter, institution)
    print(f"Seed: {seed_paper.title}")
    print(f"DOI: {seed_paper.doi}")
    
    print("\nSearching level-1 papers...")
    level_1_results, all_papers = search_level_1(seed_paper)
    
    print(f"Found {len(level_1_results)} level-1 papers")
    
    selected = show_paper_carousel(level_1_results)
    
    if not selected:
        print("\nNo papers selected. Exiting.")
        return
    
    print(f"\n{'='*80}")
    print(f"Downloading {len(selected)} selected papers...")
    print(f"{'='*80}\n")
    
    for i, paper in enumerate(selected, 1):
        print(f"[{i}/{len(selected)}] {paper.title[:60]}...")
        download_paper(paper.id, paper.title)
    
    print(f"\n{'='*80}")
    print("Second Pass: Refine Search")
    print(f"{'='*80}\n")
    print("Enter a query to refine what types of papers you want to prioritize")
    print("(e.g., 'machine learning', 'neural networks', 'optimization methods')")
    print("Press Enter to skip and use citation-based search only")
    user_query = input("\nQuery: ").strip()
    
    if not user_query:
        user_query = None
    
    print("\nSearching level-2 papers...")
    final_results = search_level_2(seed_paper, selected, all_papers, user_query=user_query, max_papers=200)
    
    display_papers(final_results, "Final Results (Top 20 Papers)", max_show=20)
    
    print(f"\nDone!")

if __name__ == "__main__":
    main()

