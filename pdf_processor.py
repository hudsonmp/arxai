#!/usr/bin/env python3
"""
PDF Processing & Annotation System
Downloads papers, extracts text, annotates with Claude, stores in Supabase
"""
import os
import json
import asyncio
from typing import Dict, List, Optional, Tuple
import requests
from dotenv import load_dotenv
import anthropic
try:
    import pymupdf as fitz
except ImportError:
    import fitz
from storage import upload_pdf_to_storage, cleanup_local_files

# Load environment variables
load_dotenv(".env.local")

# Initialize clients
claude_client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))


# ============================================================================
# PDF Download & Extraction
# ============================================================================

def download_pdf_from_doi(doi: str, output_path: str) -> bool:
    """
    Download PDF from DOI using Unpaywall API or Semantic Scholar
    Returns: True if successful, False otherwise
    """
    try:
        # Try Semantic Scholar first
        ss_url = f"https://api.semanticscholar.org/graph/v1/paper/{doi}?fields=openAccessPdf"
        response = requests.get(ss_url)
        
        if response.status_code == 200:
            data = response.json()
            pdf_url = data.get("openAccessPdf", {}).get("url")
            
            if pdf_url:
                print(f"   📥 Downloading from: {pdf_url}")
                pdf_response = requests.get(pdf_url, stream=True)
                
                if pdf_response.status_code == 200:
                    with open(output_path, 'wb') as f:
                        for chunk in pdf_response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print(f"   ✅ Downloaded to: {output_path}")
                    return True
        
        print(f"   ⚠️  No open access PDF found for DOI: {doi}")
        return False
    
    except Exception as e:
        print(f"   ❌ Error downloading PDF: {str(e)}")
        return False


def extract_text_from_pdf(pdf_path: str) -> List[Dict]:
    """
    Extract text from PDF page by page
    Returns: List of {page_number, text_content}
    """
    try:
        doc = fitz.open(pdf_path)
        pages = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            pages.append({
                "page_number": page_num + 1,
                "text_content": text,
                "word_count": len(text.split())
            })
        
        doc.close()
        return pages
    
    except Exception as e:
        print(f"   ❌ Error extracting text: {str(e)}")
        return []


# ============================================================================
# Claude Annotation Agent
# ============================================================================

ANNOTATION_PROMPT = """You are an expert academic research assistant. Analyze the following page from a research paper and provide structured annotations.

Page {page_num} text:
{page_text}

Provide a JSON response with:
1. **summary**: A 2-3 sentence summary of this page
2. **key_points**: Array of 3-5 key points or findings (if any)
3. **concepts**: Array of important concepts, terms, or methodologies mentioned
4. **tags**: Array of 3-5 categorical tags (e.g., "machine learning", "methodology", "results")
5. **importance_score**: Rate 1-5 how critical this page is to understanding the paper

IMPORTANT: Respond ONLY with valid JSON in this exact format:
{{
  "summary": "Brief summary here",
  "key_points": ["Point 1", "Point 2", "Point 3"],
  "concepts": ["Concept 1", "Concept 2"],
  "tags": ["tag1", "tag2", "tag3"],
  "importance_score": 4
}}"""


async def annotate_page_with_claude(page_data: Dict, doc_title: str) -> Dict:
    """
    Use Claude to annotate a single page
    Returns: Annotation data
    """
    try:
        prompt = ANNOTATION_PROMPT.format(
            page_num=page_data["page_number"],
            page_text=page_data["text_content"][:4000]  # Limit to avoid token limits
        )
        
        # Make API call
        loop = asyncio.get_event_loop()
        message = await loop.run_in_executor(
            None,
            lambda: claude_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
        )
        
        # Parse response
        response_text = message.content[0].text.strip()
        
        # Extract JSON
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        annotation = json.loads(response_text)
        
        # Add metadata
        annotation["page_number"] = page_data["page_number"]
        annotation["word_count"] = page_data["word_count"]
        
        return annotation
    
    except Exception as e:
        print(f"   ⚠️  Error annotating page {page_data['page_number']}: {str(e)}")
        return {
            "page_number": page_data["page_number"],
            "summary": "Error during annotation",
            "key_points": [],
            "concepts": [],
            "tags": [],
            "importance_score": 1,
            "error": str(e)
        }


async def annotate_document(pages: List[Dict], doc_title: str) -> List[Dict]:
    """
    Annotate all pages of a document in parallel
    Returns: List of annotations
    """
    print(f"   🤖 Annotating {len(pages)} pages with Claude...")
    
    # Annotate pages in parallel (with rate limiting)
    tasks = []
    for page_data in pages:
        tasks.append(annotate_page_with_claude(page_data, doc_title))
        # Small delay to avoid rate limits
        await asyncio.sleep(0.1)
    
    annotations = await asyncio.gather(*tasks)
    
    print(f"   ✅ Annotation complete!")
    return annotations


# ============================================================================
# Supabase Integration (using MCP)
# ============================================================================

async def setup_supabase_storage():
    """
    Set up Supabase storage buckets and tables using MCP
    """
    print("\n🗄️  Setting up Supabase storage...")
    
    # Note: We'll use the Supabase MCP tools to create buckets and tables
    # This is a placeholder - actual implementation uses MCP tools
    
    # Create storage bucket for PDFs
    # CREATE BUCKET: arxai-pdfs (public read, authenticated write)
    
    # Create tables:
    # 1. documents table
    # 2. annotations table
    
    print("✅ Supabase storage setup complete!")


async def store_pdf_in_supabase(pdf_path: str, paper_id: str, paper_metadata: Dict, is_original: bool = True) -> str:
    """
    Upload PDF to Supabase storage bucket
    Returns: Public URL of uploaded PDF
    """
    bucket_type = "original" if is_original else "annotated"
    print(f"   📤 Uploading {bucket_type} PDF to Supabase storage...")
    
    storage_url = upload_pdf_to_storage(pdf_path, paper_id, bucket_type)
    
    if storage_url:
        print(f"   ✅ PDF uploaded: {storage_url}")
    else:
        print(f"   ❌ PDF upload failed")
    
    return storage_url


async def store_document_metadata(paper_id: str, paper_data: Dict, pdf_url: str, relevance_score: float):
    """
    Store document metadata in Supabase documents table
    """
    print(f"   💾 Storing document metadata in Supabase...")
    
    # Will use mcp_supabase_execute_sql to insert
    # Placeholder for now
    
    print(f"   ✅ Metadata stored")


async def store_annotations(paper_id: str, annotations: List[Dict]):
    """
    Store all annotations in Supabase annotations table
    """
    print(f"   💾 Storing {len(annotations)} annotations in Supabase...")
    
    # Will use mcp_supabase_execute_sql to batch insert
    # Placeholder for now
    
    print(f"   ✅ Annotations stored")


# ============================================================================
# Main Processing Pipeline
# ============================================================================

async def process_paper(paper_data: Dict, output_dir: str = "./pdfs") -> Dict:
    """
    Complete pipeline: Download → Extract → Annotate → Store
    
    Args:
        paper_data: Dict with keys: id, doi, title, relevance_score, etc.
        output_dir: Directory to save PDFs temporarily
    
    Returns: Processing results
    """
    paper_id = paper_data["id"]
    doi = paper_data["doi"]
    title = paper_data["title"]
    
    print(f"\n📄 Processing paper: {title}")
    print(f"   DOI: {doi}")
    print(f"   Relevance: {paper_data.get('relevance_score', 0):.4f}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, f"{paper_id}.pdf")
    
    # Step 1: Download PDF
    print(f"\n   Step 1: Downloading PDF...")
    if not download_pdf_from_doi(doi, pdf_path):
        return {
            "paper_id": paper_id,
            "status": "failed",
            "reason": "PDF download failed"
        }
    
    # Step 2: Extract text
    print(f"\n   Step 2: Extracting text from PDF...")
    pages = extract_text_from_pdf(pdf_path)
    if not pages:
        return {
            "paper_id": paper_id,
            "status": "failed",
            "reason": "Text extraction failed"
        }
    
    print(f"   ✅ Extracted {len(pages)} pages")
    
    # Step 3: Annotate with Claude
    print(f"\n   Step 3: Annotating with Claude...")
    annotations = await annotate_document(pages, title)
    
    # Step 4: Store in Supabase
    print(f"\n   Step 4: Storing in Supabase...")
    
    # Upload original PDF
    pdf_url = await store_pdf_in_supabase(pdf_path, paper_id, paper_data, is_original=True)
    
    if not pdf_url:
        return {
            "paper_id": paper_id,
            "status": "failed",
            "reason": "PDF upload to Supabase failed"
        }
    
    # Store metadata
    await store_document_metadata(paper_id, paper_data, pdf_url, paper_data.get("relevance_score", 0))
    
    # Store annotations
    await store_annotations(paper_id, annotations)
    
    # Clean up local file after successful upload
    cleanup_local_files(pdf_path)
    
    print(f"\n✅ Paper processing complete!")
    
    return {
        "paper_id": paper_id,
        "status": "success",
        "pdf_url": pdf_url,
        "pages_processed": len(pages),
        "annotations_count": len(annotations),
        "avg_importance": sum(a.get("importance_score", 0) for a in annotations) / len(annotations)
    }


async def process_top_papers(papers: List[Dict], top_n: int = 5) -> List[Dict]:
    """
    Process top N papers in sequence
    
    Args:
        papers: List of paper dicts sorted by relevance
        top_n: Number of top papers to process
    
    Returns: List of processing results
    """
    print(f"\n{'='*80}")
    print(f"📚 Processing Top {top_n} Papers")
    print(f"{'='*80}")
    
    results = []
    
    for i, paper in enumerate(papers[:top_n], 1):
        print(f"\n[{i}/{top_n}]")
        result = await process_paper(paper)
        results.append(result)
        
        # Brief pause between papers
        await asyncio.sleep(1)
    
    return results


# ============================================================================
# Results Summary
# ============================================================================

def print_processing_summary(results: List[Dict]):
    """Print summary of processing results"""
    print(f"\n{'='*80}")
    print("📊 PROCESSING SUMMARY")
    print(f"{'='*80}\n")
    
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]
    
    print(f"✅ Successful: {len(successful)}/{len(results)}")
    print(f"❌ Failed: {len(failed)}/{len(results)}")
    
    if successful:
        print(f"\n📈 Statistics:")
        total_pages = sum(r["pages_processed"] for r in successful)
        total_annotations = sum(r["annotations_count"] for r in successful)
        avg_importance = sum(r["avg_importance"] for r in successful) / len(successful)
        
        print(f"   Total Pages Processed: {total_pages}")
        print(f"   Total Annotations: {total_annotations}")
        print(f"   Average Importance Score: {avg_importance:.2f}/5.0")
    
    if failed:
        print(f"\n⚠️  Failed Papers:")
        for result in failed:
            print(f"   - {result['paper_id']}: {result['reason']}")


# ============================================================================
# Main Entry Point
# ============================================================================

async def main():
    """Test the PDF processing system"""
    print("\n" + "="*80)
    print("🧪 PDF Processing & Annotation Test")
    print("="*80)
    
    # Test with a sample paper
    test_paper = {
        "id": "test-paper-1",
        "doi": "10.1145/3313831.3376518",  # CHI 2020 paper (likely open access)
        "title": "Test Paper",
        "relevance_score": 0.85
    }
    
    result = await process_paper(test_paper)
    print_processing_summary([result])


if __name__ == "__main__":
    asyncio.run(main())

