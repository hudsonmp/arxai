#!/usr/bin/env python3
"""
Direct ranking: Score all papers by similarity + citations, no graph traversal
"""
import os
import asyncio
import numpy as np
from typing import Dict, List
from dotenv import load_dotenv

from algo import (
    Paper,
    get_parent_paper_doi,
    fetch_all_papers_from_pinecone,
    cosine_similarity,
    compute_age_normalized_citation_score,
    print_separator
)
from arxiv_annotator import process_multiple_papers, print_summary

load_dotenv(".env.local")


def rank_all_papers(parent_paper: Paper, child_papers: Dict[str, Paper]) -> List[Paper]:
    """
    Directly rank all papers by: 0.7*similarity + 0.3*citation_quality
    """
    print("\n⏳ Ranking all papers...")
    print(f"   Comparing {len(child_papers)} papers to parent")
    
    ranked_papers = []
    
    for paper_id, paper in child_papers.items():
        # Similarity score
        sim_score = cosine_similarity(paper.vector, parent_paper.vector)
        
        # Citation quality score
        citation_score = compute_age_normalized_citation_score(
            paper.citation_count,
            paper.publication_date
        )
        
        # Composite relevance
        relevance = 0.7 * sim_score + 0.3 * citation_score
        
        paper.relevance_score = relevance
        paper.depth = 1  # All are 1 hop from parent
        paper.direction = "forward"
        
        ranked_papers.append(paper)
    
    # Sort by relevance
    ranked_papers.sort(key=lambda p: p.relevance_score, reverse=True)
    
    print(f"✅ Ranked {len(ranked_papers)} papers")
    print(f"   Top score: {ranked_papers[0].relevance_score:.4f}")
    print(f"   Median score: {ranked_papers[len(ranked_papers)//2].relevance_score:.4f}")
    
    return ranked_papers


def print_top_papers(papers: List[Paper], top_n: int = 15):
    """Display top N papers"""
    print(f"\n📊 TOP {top_n} PAPERS")
    print_separator()
    
    for i, paper in enumerate(papers[:top_n], 1):
        print(f"{i}. [{paper.relevance_score:.4f}] {paper.title}")
        print(f"   Venue: {paper.venue} | Citations: {paper.citation_count}")
        print(f"   DOI: {paper.doi}")
        print()


def convert_papers_for_processing(papers: List[Paper]) -> List[Dict]:
    """Convert Paper objects to dict format"""
    return [
        {
            "id": paper.id,
            "doi": paper.doi,
            "title": paper.title,
            "authors": paper.authors,
            "venue": paper.venue,
            "publication_date": paper.publication_date,
            "citation_count": paper.citation_count,
            "relevance_score": paper.relevance_score,
            "depth": paper.depth,
            "direction": paper.direction
        }
        for paper in papers
    ]


async def main():
    """
    Direct ranking pipeline:
    1. Load all papers from Pinecone
    2. Rank by similarity + citations
    3. Annotate top 15
    4. Store in Supabase
    """
    print("\n" + "="*80)
    print("🔬 Direct Paper Ranking & Annotation Pipeline (30k TPS Rate Limited)")
    print("="*80)
    
    try:
        # Step 1: Load papers
        print("\n📚 STEP 1: Loading Papers from Pinecone")
        print("="*80)
        
        print("⏳ Extracting parent paper DOI...")
        parent_doi = get_parent_paper_doi()
        print(f"✅ Parent: {parent_doi}")
        
        print("\n⏳ Fetching all papers...")
        parent_paper, child_papers = fetch_all_papers_from_pinecone(parent_doi)
        print(f"✅ Loaded {len(child_papers)} papers")
        
        if len(child_papers) == 0:
            print("\n⚠️  No papers! Run arxai.py first.")
            return
        
        # Step 2: Rank all papers
        print("\n" + "="*80)
        print("🔬 STEP 2: Ranking Papers")
        print("="*80)
        
        ranked_papers = rank_all_papers(parent_paper, child_papers)
        
        # Display top 15
        print_top_papers(ranked_papers, top_n=15)
        
        # Step 3: Annotate top 15 (keep trying until we get 15 successful)
        print("\n" + "="*80)
        print("📄 STEP 3: Downloading & Annotating Papers (Rate Limited)")
        print("="*80)
        
        processing_results = []
        success_count = 0
        paper_index = 0
        
        while success_count < 15 and paper_index < len(ranked_papers):
            # Get next batch of papers to try
            batch_size = min(5, 15 - success_count)
            papers_to_try = convert_papers_for_processing(ranked_papers[paper_index:paper_index + batch_size])
            
            # Add parent paper info to each paper for context
            for paper in papers_to_try:
                paper['parent_title'] = parent_paper.title
                paper['parent_abstract'] = parent_paper.abstract if hasattr(parent_paper, 'abstract') else ""
            
            # Process batch
            batch_results = await process_multiple_papers(papers_to_try, top_n=batch_size, max_concurrent=3)
            
            # Count successes and mark paywalled papers
            for result in batch_results:
                if result['status'] == 'success':
                    processing_results.append(result)
                    success_count += 1
                elif result['status'] == 'failed' and 'download' in result.get('reason', '').lower():
                    print(f"   🔒 Paywall: {result.get('doi', result['paper_id'])}")
            
            paper_index += batch_size
        
        print(f"\n✅ Successfully processed {success_count} papers")
        
        # Step 4: Summary
        print("\n" + "="*80)
        print("🎉 PIPELINE COMPLETE")
        print("="*80)
        
        print_summary(processing_results)
        
        print(f"\n📊 Final Results:")
        print(f"   Papers analyzed: {len(child_papers)}")
        print(f"   Papers annotated successfully: {success_count}/15")
        print(f"   Papers stored in Supabase: {success_count}")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

