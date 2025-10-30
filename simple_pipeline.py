#!/usr/bin/env python3
"""
Simplified Pipeline: Algorithm → Top 15 → Claude Annotations → Supabase
No venue/author research, just pure ranking + annotation
"""
import os
import asyncio
from typing import Dict, List
from dotenv import load_dotenv

from algo import (
    Paper,
    BidirectionalPrioritySearch,
    get_parent_paper_doi,
    fetch_all_papers_from_pinecone,
    print_separator
)
from pdf_processor import process_top_papers, print_processing_summary

load_dotenv(".env.local")


def run_simplified_algorithm(parent_paper: Paper, child_papers: Dict[str, Paper]) -> List[Paper]:
    """
    Run algorithm with NO quality research - just embeddings + citations
    """
    print("\n⏳ Running bidirectional priority search...")
    print("   Using: Embedding similarity + Citation counts only")
    
    search = BidirectionalPrioritySearch(
        seed_paper=parent_paper,
        all_papers=child_papers,
        venue_scores=None,  # No venue research
        institution_scores=None,  # No institution research
        author_scores=None,  # No author research
        alpha=0.65,
        beta=0.15,
        epsilon=0.02,
        omega_sim=0.7,  # Higher weight on similarity since no quality scores
        omega_qual=0.3,  # Lower weight since we have no real quality data
        max_papers=10000,  # Process all papers
        top_k=5,
        boost_check_interval=10
    )
    
    results = search.run()
    print(f"✅ Found {len(results)} high-relevance papers")
    
    return results


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
    Simplified pipeline:
    1. Run algorithm (no research)
    2. Get top 15 papers
    3. Annotate with Claude
    4. Store in Supabase
    """
    print("\n" + "="*80)
    print("🔬 Simplified Paper Discovery & Annotation Pipeline")
    print("="*80)
    
    try:
        # Step 1: Load papers from Pinecone
        print("\n📚 STEP 1: Loading Papers")
        print("="*80)
        
        print("⏳ Extracting parent paper DOI...")
        parent_doi = get_parent_paper_doi()
        print(f"✅ Parent: {parent_doi}")
        
        print("\n⏳ Fetching papers from Pinecone...")
        parent_paper, child_papers = fetch_all_papers_from_pinecone(parent_doi)
        print(f"✅ Loaded {len(child_papers)} child papers")
        
        if len(child_papers) == 0:
            print("\n⚠️  No child papers! Run arxai.py first.")
            return
        
        # Step 2: Run algorithm
        print("\n" + "="*80)
        print("🔬 STEP 2: Running Algorithm")
        print("="*80)
        
        results = run_simplified_algorithm(parent_paper, child_papers)
        
        # Display top 15
        print_top_papers(results, top_n=15)
        
        # Step 3: Process top 15 papers
        print("\n" + "="*80)
        print("📄 STEP 3: Annotating Top 15 Papers with Claude")
        print("="*80)
        
        top_15_papers = convert_papers_for_processing(results[:15])
        processing_results = await process_top_papers(top_15_papers, top_n=15)
        
        # Step 4: Summary
        print("\n" + "="*80)
        print("🎉 PIPELINE COMPLETE")
        print("="*80)
        
        print_processing_summary(processing_results)
        
        print("\n📊 Results:")
        print(f"   Total papers analyzed: {len(child_papers)}")
        print(f"   High-relevance papers found: {len(results)}")
        print(f"   Papers annotated: {len([r for r in processing_results if r['status'] == 'success'])}")
        print(f"   Papers stored in Supabase: Ready for queries")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

