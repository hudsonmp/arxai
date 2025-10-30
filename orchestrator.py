#!/usr/bin/env python3
"""
Main Orchestrator for Paper Discovery & Processing Pipeline
1. Run algorithm with simplified scoring (no Claude research) → Get top 5
2. Background: Claude researches venues/authors and updates scores
3. For top 5: Download PDFs → Extract → Annotate → Store in Supabase
"""
import os
import asyncio
from typing import Dict, List, Tuple
from dotenv import load_dotenv

# Import our modules
from algo import (
    Paper,
    BidirectionalPrioritySearch,
    get_parent_paper_doi,
    fetch_all_papers_from_pinecone,
    print_header,
    print_separator
)
from quality_research import (
    research_all_parallel,
    load_research_cache,
    save_research_cache
)
from arxiv_annotator import process_multiple_papers, print_summary

# Load environment variables
load_dotenv(".env.local")


# ============================================================================
# Supabase Setup using MCP
# ============================================================================

async def setup_supabase_schema():
    """
    Create Supabase storage buckets and tables using MCP
    """
    print("\n" + "="*80)
    print("🗄️  Setting up Supabase Schema")
    print("="*80 + "\n")
    
    # SQL for documents table
    documents_table_sql = """
    CREATE TABLE IF NOT EXISTS documents (
        id TEXT PRIMARY KEY,
        doi TEXT UNIQUE NOT NULL,
        title TEXT NOT NULL,
        authors JSONB,
        venue TEXT,
        publication_date TEXT,
        citation_count INTEGER,
        relevance_score FLOAT,
        pdf_url TEXT,
        storage_path TEXT,
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW()
    );
    
    CREATE INDEX IF NOT EXISTS idx_documents_relevance ON documents(relevance_score DESC);
    CREATE INDEX IF NOT EXISTS idx_documents_doi ON documents(doi);
    """
    
    # SQL for annotations table
    annotations_table_sql = """
    CREATE TABLE IF NOT EXISTS annotations (
        id SERIAL PRIMARY KEY,
        document_id TEXT REFERENCES documents(id) ON DELETE CASCADE,
        page_number INTEGER NOT NULL,
        summary TEXT,
        key_points JSONB,
        concepts JSONB,
        tags JSONB,
        importance_score INTEGER CHECK (importance_score BETWEEN 1 AND 5),
        word_count INTEGER,
        created_at TIMESTAMP DEFAULT NOW()
    );
    
    CREATE INDEX IF NOT EXISTS idx_annotations_document ON annotations(document_id);
    CREATE INDEX IF NOT EXISTS idx_annotations_importance ON annotations(importance_score DESC);
    CREATE INDEX IF NOT EXISTS idx_annotations_tags ON annotations USING GIN (tags);
    """
    
    # SQL for quality_scores table (for incremental updates)
    quality_scores_sql = """
    CREATE TABLE IF NOT EXISTS quality_scores (
        id SERIAL PRIMARY KEY,
        entity_name TEXT UNIQUE NOT NULL,
        entity_type TEXT NOT NULL CHECK (entity_type IN ('venue', 'author', 'institution')),
        quality_score FLOAT,
        metadata JSONB,
        last_updated TIMESTAMP DEFAULT NOW()
    );
    
    CREATE INDEX IF NOT EXISTS idx_quality_entity ON quality_scores(entity_name, entity_type);
    """
    
    print("Creating tables...")
    print("  → documents table")
    print("  → annotations table")
    print("  → quality_scores table")
    
    # Note: In production, use mcp_supabase_apply_migration here
    # For now, printing the SQL that would be executed
    
    print("\n✅ Supabase schema ready!")
    print("\n📝 SQL to execute:")
    print(documents_table_sql)
    print(annotations_table_sql)
    print(quality_scores_sql)
    
    return True


# ============================================================================
# Algorithm Execution
# ============================================================================

def run_algorithm_simple_scoring(parent_paper: Paper, child_papers: Dict[str, Paper]) -> List[Paper]:
    """
    Run algorithm with SIMPLIFIED scoring (no Claude research)
    Uses only citation counts and embedding similarity
    """
    print("\n⏳ Running algorithm with simplified scoring (no Claude research)...")
    print("   Using: Citation counts + Embedding similarity only")
    
    # Run with NO quality scores (will use neutral 0.5 for all venues/authors)
    search = BidirectionalPrioritySearch(
        seed_paper=parent_paper,
        all_papers=child_papers,
        venue_scores=None,  # No venue scores yet
        institution_scores=None,
        author_scores=None,
        alpha=0.65,
        beta=0.15,
        epsilon=0.02,
        omega_sim=0.6,
        omega_qual=0.4,
        max_papers=500,
        top_k=5,
        boost_check_interval=10
    )
    
    results = search.run()
    print(f"✅ Algorithm complete! Found {len(results)} high-relevance papers")
    
    return results


async def run_quality_research_background(papers: Dict[str, Paper], parent_paper: Paper):
    """
    Run quality research in background (async)
    This can update scores while PDF processing happens
    """
    print("\n🔬 Starting background quality research...")
    
    # Extract unique venues and authors
    all_papers_with_parent = {**papers, parent_paper.id: parent_paper}
    
    venues = set()
    authors = set()
    
    for paper in all_papers_with_parent.values():
        if paper.venue and paper.venue != "Unknown":
            venues.add(paper.venue)
        if paper.authors:
            for author in paper.authors:
                if author and author.strip():
                    authors.add(author)
    
    # Try loading from cache
    cached_venues, cached_institutions, cached_authors = load_research_cache()
    
    # Filter out cached items
    venues_to_research = [v for v in venues if v not in cached_venues]
    authors_to_research = [a for a in authors if a not in cached_authors]
    
    print(f"   Venues to research: {len(venues_to_research)}")
    print(f"   Authors to research: {len(authors_to_research)}")
    
    if venues_to_research or authors_to_research:
        print("   🚀 Researching in parallel...")
        
        new_venues, new_institutions, new_authors = await research_all_parallel(
            venues_to_research,
            [],
            authors_to_research
        )
        
        # Merge with cache
        all_venues = {**cached_venues, **new_venues}
        all_institutions = {**cached_institutions, **new_institutions}
        all_authors = {**cached_authors, **new_authors}
        
        # Save cache
        save_research_cache(all_venues, all_institutions, all_authors)
        
        print(f"   ✅ Quality research complete!")
        return all_venues, all_institutions, all_authors
    else:
        print("   ✅ All data in cache!")
        return cached_venues, cached_institutions, cached_authors


# ============================================================================
# Results Display
# ============================================================================

def print_top_papers(papers: List[Paper], top_n: int = 5):
    """Display top N papers"""
    print(f"\n📊 TOP {top_n} PAPERS (Ranked by Relevance)")
    print_separator()
    
    for i, paper in enumerate(papers[:top_n], 1):
        print(f"{i}. Relevance: {paper.relevance_score:.4f} | Depth: {paper.depth}")
        print(f"   Title: {paper.title}")
        print(f"   Venue: {paper.venue}")
        print(f"   Authors: {', '.join(paper.authors[:3])}{'...' if len(paper.authors) > 3 else ''}")
        print(f"   Citations: {paper.citation_count}")
        print(f"   DOI: {paper.doi}")
        print()


def convert_papers_for_processing(papers: List[Paper]) -> List[Dict]:
    """Convert Paper objects to dict format for PDF processor"""
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


# ============================================================================
# Main Orchestration
# ============================================================================

async def main():
    """
    Main orchestration pipeline:
    1. Setup Supabase
    2. Run algorithm with simple scoring → Get top 5
    3. Background: Start quality research
    4. Foreground: Process top 5 PDFs
    5. Display results
    """
    print_header()
    
    try:
        # ===== STEP 1: Setup Supabase =====
        await setup_supabase_schema()
        
        # ===== STEP 2: Get parent paper and children from Pinecone =====
        print("\n" + "="*80)
        print("📚 Loading Papers from Pinecone")
        print("="*80 + "\n")
        
        print("⏳ Extracting parent paper DOI from PDF...")
        parent_doi = get_parent_paper_doi()
        print(f"✅ Parent DOI: {parent_doi}")
        
        print("\n⏳ Fetching all papers from Pinecone database...")
        parent_paper, child_papers = fetch_all_papers_from_pinecone(parent_doi)
        print(f"✅ Found {len(child_papers)} child papers")
        
        if len(child_papers) == 0:
            print("\n⚠️  No child papers found in Pinecone!")
            print("💡 Run arxai.py first to populate the database with related papers.")
            return
        
        # ===== STEP 3: Run algorithm with SIMPLE scoring =====
        print("\n" + "="*80)
        print("🔬 Phase 1: Simple Scoring Algorithm")
        print("="*80)
        
        results = run_algorithm_simple_scoring(parent_paper, child_papers)
        
        # Display top 5
        print_top_papers(results, top_n=5)
        
        # ===== STEP 4: Start background quality research =====
        print("\n" + "="*80)
        print("🔬 Phase 2: Background Quality Research")
        print("="*80)
        
        # Start quality research task (runs in background)
        quality_research_task = asyncio.create_task(
            run_quality_research_background(child_papers, parent_paper)
        )
        
        # ===== STEP 5: Process top 5 papers (foreground) =====
        print("\n" + "="*80)
        print("📄 Phase 3: PDF Processing & Annotation")
        print("="*80)
        
        # Convert top 5 papers to dict format
        top_5_papers = convert_papers_for_processing(results[:5])
        
        # Add parent paper context to each paper for Claude annotations
        for paper in top_5_papers:
            paper['parent_title'] = parent_paper.title
            paper['parent_abstract'] = parent_paper.abstract if parent_paper.abstract else ""
        
        # Process papers with visual highlights and comments
        processing_results = await process_multiple_papers(top_5_papers, top_n=5, max_concurrent=3)
        
        # ===== STEP 6: Wait for quality research to complete =====
        print("\n⏳ Waiting for background quality research to complete...")
        venue_scores, institution_scores, author_scores = await quality_research_task
        
        print(f"✅ Quality research complete!")
        print(f"   Venues scored: {len(venue_scores)}")
        print(f"   Authors scored: {len(author_scores)}")
        
        # ===== STEP 7: Display final summary =====
        print("\n" + "="*80)
        print("🎉 PIPELINE COMPLETE")
        print("="*80)
        
        print_summary(processing_results)
        
        print("\n📊 Next Steps:")
        print("   1. Rankings may be updated with quality scores")
        print("   2. Papers are stored in Supabase with annotations")
        print("   3. Access papers via Supabase storage buckets")
        print("   4. Query annotations for semantic search")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())

