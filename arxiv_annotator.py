#!/usr/bin/env python3
"""
arXiv PDF Annotator
Downloads PDFs from arXiv, uses Claude to identify key sections,
adds visual highlights directly to the PDF
"""
import os
import re
import json
import asyncio
import time
from typing import Dict, List, Tuple
import requests
try:
    import pymupdf as fitz
except ImportError:
    import fitz
from dotenv import load_dotenv
import anthropic
from storage import upload_both_pdfs, cleanup_local_files

load_dotenv(".env.local")
claude_client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))


# ============================================================================
# Rate Limiter (30k TPS)
# ============================================================================

class TokenRateLimiter:
    """Rate limiter for Claude API calls - 30k tokens per second"""
    
    def __init__(self, tokens_per_second: int = 30000):
        self.tokens_per_second = tokens_per_second
        self.tokens_used = 0
        self.window_start = time.time()
        self.lock = asyncio.Lock()
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate tokens from text (roughly 1 token per 4 characters)"""
        return len(text) // 4
    
    async def wait_if_needed(self, estimated_tokens: int):
        """Wait if adding these tokens would exceed rate limit"""
        async with self.lock:
            current_time = time.time()
            elapsed = current_time - self.window_start
            
            # Reset window if 1 second has passed
            if elapsed >= 1.0:
                self.tokens_used = 0
                self.window_start = current_time
                elapsed = 0
            
            # Check if we need to wait
            if self.tokens_used + estimated_tokens > self.tokens_per_second:
                wait_time = 1.0 - elapsed
                if wait_time > 0:
                    print(f"      ⏸️  Rate limit approaching, waiting {wait_time:.3f}s...")
                    await asyncio.sleep(wait_time)
                    # Reset after waiting
                    self.tokens_used = 0
                    self.window_start = time.time()
            
            # Add tokens to current usage
            self.tokens_used += estimated_tokens
            print(f"      📊 Token usage: {self.tokens_used}/{self.tokens_per_second} TPS")

# Global rate limiter instance
rate_limiter = TokenRateLimiter(tokens_per_second=30000)


# ============================================================================
# arXiv PDF Download
# ============================================================================

def extract_arxiv_id_from_doi(doi: str) -> str:
    """Extract arXiv ID from DOI like '10.48550/arXiv.2407.21783'"""
    if 'arxiv' in doi.lower():
        # Pattern: arXiv.YYMM.NNNNN
        match = re.search(r'arxiv[.\s]*(\d{4}\.\d{4,5})', doi, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def download_arxiv_pdf(arxiv_id: str, output_path: str) -> bool:
    """Download PDF from arXiv"""
    try:
        url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        print(f"   📥 Downloading from arXiv: {url}")
        
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"   ✅ Downloaded to: {output_path}")
            return True
        else:
            print(f"   ⚠️  arXiv returned status {response.status_code}")
            return False
    
    except Exception as e:
        print(f"   ❌ Error downloading from arXiv: {str(e)}")
        return False


def download_from_url(url: str, output_path: str, source_name: str) -> bool:
    """Generic function to download PDF from a URL"""
    try:
        print(f"   📥 Trying {source_name}: {url}")
        response = requests.get(url, stream=True, timeout=30)
        
        if response.status_code == 200:
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"   ✅ Downloaded from {source_name}!")
            return True
        else:
            print(f"   ⚠️  {source_name} returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"   ⚠️  {source_name} failed: {str(e)[:50]}")
        return False


def try_unpaywall(doi: str) -> str:
    """Try to get open access PDF URL from Unpaywall API"""
    try:
        # Unpaywall requires email in query
        email = "research@example.com"
        url = f"https://api.unpaywall.org/v2/{doi}?email={email}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("is_oa"):  # is open access
                best_location = data.get("best_oa_location")
                if best_location:
                    pdf_url = best_location.get("url_for_pdf")
                    if pdf_url:
                        return pdf_url
    except:
        pass
    return None


def try_pubmed_central(doi: str) -> str:
    """Try to get PDF from PubMed Central"""
    try:
        # Search for paper in PMC
        search_url = f"https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?ids={doi}&format=json"
        response = requests.get(search_url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            records = data.get("records", [])
            if records and len(records) > 0:
                pmcid = records[0].get("pmcid")
                if pmcid:
                    # Construct PDF URL
                    pdf_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/"
                    return pdf_url
    except:
        pass
    return None


def download_pdf_for_paper(doi: str, output_path: str) -> bool:
    """
    Try to download PDF from multiple sources with fallbacks
    Sources tried in order:
    1. arXiv (if it's an arXiv paper)
    2. Semantic Scholar open access
    3. Unpaywall API
    4. PubMed Central
    """
    print(f"   Attempting to download PDF from multiple sources...")
    
    # Source 1: Try arXiv first (fastest and most reliable for arXiv papers)
    arxiv_id = extract_arxiv_id_from_doi(doi)
    if arxiv_id:
        arxiv_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        if download_from_url(arxiv_url, output_path, "arXiv"):
            return True
    
    # Source 2: Try Semantic Scholar
    try:
        ss_url = f"https://api.semanticscholar.org/graph/v1/paper/{doi}?fields=openAccessPdf"
        response = requests.get(ss_url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            pdf_url = data.get("openAccessPdf", {}).get("url")
            
            if pdf_url:
                if download_from_url(pdf_url, output_path, "Semantic Scholar"):
                    return True
    except:
        pass
    
    # Source 3: Try Unpaywall
    print(f"   Checking Unpaywall for open access version...")
    unpaywall_url = try_unpaywall(doi)
    if unpaywall_url:
        if download_from_url(unpaywall_url, output_path, "Unpaywall"):
            return True
    
    # Source 4: Try PubMed Central (for medical/bio papers)
    print(f"   Checking PubMed Central...")
    pmc_url = try_pubmed_central(doi)
    if pmc_url:
        if download_from_url(pmc_url, output_path, "PubMed Central"):
            return True
    
    # All sources failed
    print(f"   ❌ Could not find open access PDF after trying 4 sources")
    print(f"   Paper is likely behind a paywall. Skipping...")
    return False


# ============================================================================
# Claude Highlighting Agent
# ============================================================================

HIGHLIGHT_PROMPT = """You are an expert research paper analyzer. You are analyzing a paper in relation to this PARENT PAPER:

PARENT PAPER: {parent_title}
PARENT ABSTRACT: {parent_abstract}

Now read this page from a CHILD PAPER and identify the MOST IMPORTANT sections that relate to the parent paper.

Page {page_num} text:
{page_text}

Identify 2-5 key text snippets (each 10-50 words) that contain:
- Important findings or results
- Key methodologies
- Novel contributions
- Critical insights

For EACH snippet, explain HOW IT RELATES to the parent paper (e.g., similar methodology, contradictory findings, extends the work, addresses same problem, etc.)

Return ONLY a JSON array of objects with this format:
[
  {{
    "text": "exact text snippet to highlight",
    "reason": "why this is important",
    "priority": 1-3 (1=critical, 2=important, 3=relevant),
    "relation_to_parent": "Detailed explanation of how this relates to the parent paper"
  }}
]

IMPORTANT: 
- Only include text that EXACTLY appears on the page
- Keep snippets short (10-50 words)
- Focus on the most impactful content
- Make relation_to_parent specific and insightful (2-3 sentences)
- Return valid JSON only, no other text"""


async def get_highlight_suggestions(page_num: int, page_text: str, parent_title: str, parent_abstract: str) -> List[Dict]:
    """Use Claude to identify text sections to highlight with relation to parent paper"""
    try:
        prompt = HIGHLIGHT_PROMPT.format(
            parent_title=parent_title,
            parent_abstract=parent_abstract,
            page_num=page_num, 
            page_text=page_text[:4000]
        )
        
        # Estimate tokens: input + output
        input_tokens = rate_limiter.estimate_tokens(prompt)
        output_tokens = 2048  # max_tokens
        total_estimated = input_tokens + output_tokens
        
        # Wait if needed to respect rate limit
        await rate_limiter.wait_if_needed(total_estimated)
        
        loop = asyncio.get_event_loop()
        message = await loop.run_in_executor(
            None,
            lambda: claude_client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}]
            )
        )
        
        response_text = message.content[0].text.strip()
        
        # Extract JSON
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        elif response_text.startswith('['):
            pass  # Already JSON
        else:
            # Try to find JSON array
            match = re.search(r'\[.*\]', response_text, re.DOTALL)
            if match:
                response_text = match.group(0)
        
        highlights = json.loads(response_text)
        return highlights
    
    except Exception as e:
        print(f"      ⚠️  Error getting highlights for page {page_num}: {str(e)}")
        return []


# ============================================================================
# PDF Highlighting
# ============================================================================

def add_relevance_score_header(page: fitz.Page, relevance_score: float, parent_title: str = None):
    """Add bold relevance score header to the top of the page"""
    try:
        # Create text to display
        if parent_title and len(parent_title) < 80:
            score_text = f"Relevance to '{parent_title}': {relevance_score:.4f}"
        else:
            score_text = f"Relevance Score: {relevance_score:.4f}"
        
        # Position at top center of page
        page_rect = page.rect
        text_rect = fitz.Rect(50, 10, page_rect.width - 50, 50)
        
        # Add text annotation with yellow background
        text_annot = page.add_freetext_annot(
            text_rect,
            score_text,
            fontsize=14,
            fontname="helv",
            text_color=(0, 0, 0),  # Black text
            fill_color=(1, 0.95, 0.6),  # Light yellow background
            align=fitz.TEXT_ALIGN_CENTER
        )
        
        # Make it bold by setting border
        text_annot.set_border(width=2)
        text_annot.update()
        
        return True
    
    except Exception as e:
        print(f"      ⚠️  Error adding score header: {str(e)}")
        return False


def add_highlight_to_pdf(page: fitz.Page, text_to_highlight: str, priority: int = 2, comment_text: str = None) -> bool:
    """Add yellow highlight with collapsible comment to specific text in PDF"""
    try:
        # Color based on priority
        colors = {
            1: (1, 0.8, 0),    # Bright yellow for critical
            2: (1, 1, 0),      # Yellow for important
            3: (0.8, 1, 0.8)   # Light green for relevant
        }
        color = colors.get(priority, (1, 1, 0))
        
        # Search for text
        text_instances = page.search_for(text_to_highlight)
        
        if not text_instances:
            return False
        
        # Highlight all instances (usually just one)
        for inst in text_instances:
            # Add highlight annotation
            highlight = page.add_highlight_annot(inst)
            highlight.set_colors(stroke=color)
            
            # Add collapsible comment if provided
            if comment_text:
                highlight.set_info(content=comment_text)
                # Set author and creation date for better annotation metadata
                highlight.set_info(title="ArXai Analysis")
            
            highlight.update()
        
        return True
    
    except Exception as e:
        print(f"      ⚠️  Error adding highlight: {str(e)}")
        return False


async def annotate_pdf(pdf_path: str, output_path: str, title: str, parent_title: str = "", parent_abstract: str = "", relevance_score: float = 0.0, max_pages: int = 20) -> Dict:
    """
    Annotate PDF by adding highlights with comments and relevance score header
    Uses parallel Claude calls for multiple pages
    Returns: annotation summary
    """
    print(f"   🎨 Annotating PDF with Claude (parallel processing)...")
    
    try:
        doc = fitz.open(pdf_path)
        total_pages = min(len(doc), max_pages)
        
        # Add relevance score header to first page
        if relevance_score > 0:
            first_page = doc[0]
            add_relevance_score_header(first_page, relevance_score, parent_title)
            print(f"      ✅ Added relevance score header: {relevance_score:.4f}")
        
        # Extract text from all pages first
        pages_data = []
        for page_num in range(total_pages):
            page = doc[page_num]
            text = page.get_text()
            
            if len(text.strip()) < 100:  # Skip nearly empty pages
                continue
            
            pages_data.append({
                "page_num": page_num,
                "page_obj": page,
                "text": text
            })
        
        print(f"      Processing {len(pages_data)} pages in parallel...")
        
        # Get highlights for all pages in parallel
        highlight_tasks = [
            get_highlight_suggestions(p["page_num"] + 1, p["text"], parent_title, parent_abstract)
            for p in pages_data
        ]
        
        all_highlights = await asyncio.gather(*highlight_tasks)
        
        # Add highlights with comments to PDF
        total_highlights = 0
        total_comments = 0
        page_summaries = []
        
        for page_data, highlights in zip(pages_data, all_highlights):
            if not highlights:
                continue
            
            page = page_data["page_obj"]
            page_highlight_count = 0
            
            for h in highlights:
                text_snippet = h.get("text", "")
                priority = h.get("priority", 2)
                comment = h.get("relation_to_parent", "")
                
                if add_highlight_to_pdf(page, text_snippet, priority, comment):
                    page_highlight_count += 1
                    total_highlights += 1
                    if comment:
                        total_comments += 1
            
            page_summaries.append({
                "page": page_data["page_num"] + 1,
                "highlights_added": page_highlight_count,
                "suggestions": len(highlights)
            })
        
        # Save annotated PDF
        doc.save(output_path, garbage=4, deflate=True)
        doc.close()
        
        print(f"   ✅ Added {total_highlights} highlights with {total_comments} comments across {len(page_summaries)} pages")
        
        return {
            "status": "success",
            "total_highlights": total_highlights,
            "total_comments": total_comments,
            "pages_annotated": len(page_summaries),
            "page_summaries": page_summaries
        }
    
    except Exception as e:
        print(f"   ❌ Error annotating PDF: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }


# ============================================================================
# Complete Pipeline
# ============================================================================

async def process_paper_with_highlights(paper: Dict, output_dir: str = "./annotated_pdfs") -> Dict:
    """
    Complete pipeline: Download → Annotate → Save
    """
    paper_id = paper["id"]
    doi = paper["doi"]
    title = paper["title"]
    parent_title = paper.get("parent_title", "")
    parent_abstract = paper.get("parent_abstract", "")
    relevance_score = paper.get("relevance_score", 0.0)
    
    print(f"\n📄 Processing: {title[:60]}...")
    print(f"   DOI: {doi}")
    if relevance_score > 0:
        print(f"   Relevance: {relevance_score:.4f}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Download original PDF
    original_path = os.path.join(output_dir, f"{paper_id}_original.pdf")
    
    print(f"\n   Step 1: Downloading PDF...")
    if not download_pdf_for_paper(doi, original_path):
        return {
            "paper_id": paper_id,
            "status": "failed",
            "reason": "Could not download PDF"
        }
    
    # Annotate PDF with parent paper context
    print(f"\n   Step 2: Annotating with Claude highlights and comments...")
    annotated_path = os.path.join(output_dir, f"{paper_id}_annotated.pdf")
    
    annotation_result = await annotate_pdf(
        original_path, 
        annotated_path, 
        title, 
        parent_title=parent_title, 
        parent_abstract=parent_abstract,
        relevance_score=relevance_score
    )
    
    if annotation_result["status"] == "success":
        # Upload both PDFs to Supabase storage
        print(f"\n   Step 3: Uploading to Supabase storage...")
        original_url, annotated_url = upload_both_pdfs(original_path, annotated_path, paper_id)
        
        if not original_url or not annotated_url:
            print(f"   ⚠️  Upload failed, keeping local files")
            return {
                "paper_id": paper_id,
                "status": "partial_success",
                "reason": "PDF annotation succeeded but upload failed",
                "original_pdf": original_path,
                "annotated_pdf": annotated_path,
                **annotation_result
            }
        
        # Clean up local files after successful upload
        cleanup_local_files(original_path, annotated_path)
        
        print(f"\n✅ Paper processed and uploaded successfully!")
        print(f"   Original URL: {original_url}")
        print(f"   Annotated URL: {annotated_url}")
        
        return {
            "paper_id": paper_id,
            "doi": doi,
            "title": title,
            "status": "success",
            "original_pdf_url": original_url,
            "annotated_pdf_url": annotated_url,
            **annotation_result
        }
    else:
        return {
            "paper_id": paper_id,
            "status": "failed",
            "reason": annotation_result.get("error", "Unknown error")
        }


async def process_multiple_papers(papers: List[Dict], top_n: int = 15, max_concurrent: int = 5) -> List[Dict]:
    """Process multiple papers in parallel with concurrency control"""
    print(f"\n{'='*80}")
    print(f"📚 Processing & Annotating {top_n} Papers in Parallel")
    print(f"   Max concurrent: {max_concurrent}")
    print(f"{'='*80}")
    
    # Create semaphore for rate limiting
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def process_with_semaphore(paper: Dict, index: int):
        async with semaphore:
            print(f"\n[{index}/{top_n}] Starting: {paper['title'][:60]}...")
            result = await process_paper_with_highlights(paper)
            print(f"[{index}/{top_n}] ✓ Completed")
            return result
    
    # Process all papers in parallel
    tasks = [
        process_with_semaphore(paper, i+1) 
        for i, paper in enumerate(papers[:top_n])
    ]
    
    results = await asyncio.gather(*tasks)
    
    return results


# ============================================================================
# Summary
# ============================================================================

def print_summary(results: List[Dict]):
    """Print summary of processing results"""
    print(f"\n{'='*80}")
    print("📊 ANNOTATION SUMMARY")
    print(f"{'='*80}\n")
    
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]
    
    print(f"✅ Successfully annotated: {len(successful)}/{len(results)}")
    print(f"❌ Failed: {len(failed)}/{len(results)}")
    
    if successful:
        total_highlights = sum(r.get("total_highlights", 0) for r in successful)
        total_comments = sum(r.get("total_comments", 0) for r in successful)
        total_pages = sum(r.get("pages_annotated", 0) for r in successful)
        
        print(f"\n📈 Statistics:")
        print(f"   Total highlights added: {total_highlights}")
        print(f"   Total comments added: {total_comments}")
        print(f"   Total pages annotated: {total_pages}")
        print(f"   Average highlights per paper: {total_highlights/len(successful):.1f}")
        print(f"   Average comments per paper: {total_comments/len(successful):.1f}")
        
        print(f"\n🗄️  PDFs uploaded to Supabase storage buckets:")
        print(f"   - arxai-originals (original PDFs)")
        print(f"   - arxai-annotated (highlighted PDFs with comments)")
    
    if failed:
        print(f"\n⚠️  Failed papers:")
        for r in failed:
            print(f"   - {r.get('doi', r['paper_id'])}: {r['reason']}")


# ============================================================================
# Test
# ============================================================================

async def main():
    """Test the annotation system"""
    print("\n" + "="*80)
    print("🧪 arXiv PDF Annotator Test")
    print("="*80)
    
    # Test paper (arXiv paper)
    test_paper = {
        "id": "test-arxiv",
        "doi": "10.48550/arXiv.2407.21783",
        "title": "Test arXiv Paper"
    }
    
    result = await process_paper_with_highlights(test_paper)
    print_summary([result])


if __name__ == "__main__":
    asyncio.run(main())

