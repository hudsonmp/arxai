#!/usr/bin/env python3
"""
arXiv AI Paper Processing Tool
Single-file version that reads PDF path from .env.local
"""
import os
import re
from typing import Dict, List, Optional
import requests
try:
    import pymupdf as fitz
except ImportError:
    import fitz
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables
load_dotenv(".env.local")

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Global index reference (lazy initialization)
_index = None


def get_index():
    """
    Get or create the Pinecone index for arxAI.
    Uses lazy initialization to avoid startup errors.
    """
    global _index
    
    if _index is not None:
        return _index
    
    index_name = "arxai"
    
    # Check if index exists
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    
    if index_name not in existing_indexes:
        # Create index with SPECTER v2 embedding dimensions
        print(f"Creating Pinecone index '{index_name}' with dimension 768...")
        pc.create_index(
            name=index_name,
            dimension=768,  # SPECTER v2 embedding size
            metric="cosine",
            spec={
                "serverless": {
                    "cloud": "aws",
                    "region": "us-east-1"
                }
            }
        )
        print(f"✅ Index '{index_name}' created successfully!")
    
    _index = pc.Index(index_name)
    return _index


def extract_doi_from_pdf(pdf_path: str) -> Optional[str]:
    """
    Extract DOI from PDF headers and footers.
    Looks for patterns like:
    - doi:10.xxxx/xxxxx
    - DOI: 10.xxxx/xxxxx
    - https://doi.org/10.xxxx/xxxxx
    """
    try:
        doc = fitz.open(pdf_path)
        first_page = doc[0]
        
        # Get text from header (top 100 pixels) and footer (bottom 100 pixels)
        page_height = first_page.rect.height
        header_rect = fitz.Rect(0, 0, first_page.rect.width, 100)
        footer_rect = fitz.Rect(0, page_height - 100, first_page.rect.width, page_height)
        
        header_text = first_page.get_text("text", clip=header_rect)
        footer_text = first_page.get_text("text", clip=footer_rect)
        
        # Combine header and footer text
        text = header_text + "\n" + footer_text
        
        # DOI regex patterns
        doi_patterns = [
            r'doi:\s*(10\.\d+/[^\s]+)',
            r'DOI:\s*(10\.\d+/[^\s]+)',
            r'https?://doi\.org/(10\.\d+/[^\s]+)',
            r'\b(10\.\d+/[^\s]+)\b'
        ]
        
        for pattern in doi_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                doi = match.group(1)
                # Clean up DOI (remove trailing punctuation)
                doi = re.sub(r'[.,;:\)]$', '', doi)
                doc.close()
                return doi
        
        doc.close()
        return None
        
    except Exception as e:
        print(f"❌ Error reading PDF: {str(e)}")
        return None


def get_paper_metadata(paper_id: str) -> Dict:
    """API Call #1: Get paper metadata"""
    fields = "title,authors,venue,publicationDate,publicationTypes,journal,citationCount,referenceCount,influentialCitationCount,fieldsOfStudy,authors.name,authors.affiliations"
    url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}?fields={fields}"
    
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Semantic Scholar API error: {response.text}")
    
    data = response.json()
    
    return {
        "title": data.get("title"),
        "venue": data.get("venue"),
        "authors": [a["name"] for a in data.get("authors", [])],
        "institutions": [a.get("affiliations", []) for a in data.get("authors", [])],
        "citation_count": data.get("citationCount"),
        "publication_date": data.get("publicationDate"),
        "fields_of_study": data.get("fieldsOfStudy"),
        "reference_count": data.get("referenceCount"),
        "influential_citation_count": data.get("influentialCitationCount")
    }


def get_paper_embedding(paper_id: str) -> Dict:
    """API Call #2: Get paper embedding vector"""
    fields = "title,embedding,authors,venue,citationCount"
    url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}?fields={fields}"
    
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Semantic Scholar API error: {response.text}")
    
    data = response.json()
    
    # Check if embedding exists
    embedding = data.get("embedding")
    if embedding is None:
        raise Exception(f"No embedding available for this paper. Semantic Scholar may not have computed embeddings for this paper yet.")
    
    vector = embedding.get("vector")
    if not vector:
        raise Exception("Embedding exists but vector is empty or invalid.")
    
    return {
        "title": data.get("title"),
        "embedding_model": embedding.get("model"),
        "vector": vector,
        "vector_length": len(vector),
        "first10_dims": vector[:10],
        "authors": [a.get("name") for a in data.get("authors", [])],
        "venue": data.get("venue"),
        "citation_count": data.get("citationCount")
    }


def get_citing_papers(paper_id: str, verbose: bool = False) -> List[str]:
    """API Call #3: Get citing papers' DOIs"""
    url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/citations?fields=paperId,externalIds"
    
    try:
        response = requests.get(url)
        if response.status_code != 200:
            if verbose:
                print(f"     ⚠️  API returned status {response.status_code}")
            return []
        
        data = response.json()
        if data is None or not isinstance(data, dict):
            return []
        
        citations = data.get('data', [])
        if citations is None:
            return []
        
        if verbose:
            print(f"     Retrieved {len(citations)} citing papers from API")
        
        doi_list = []
        for citation in citations:
            try:
                if citation is None:
                    continue
                    
                citing_paper = citation.get("citingPaper")
                if citing_paper is None:
                    continue
                
                external_ids = citing_paper.get("externalIds")
                if external_ids is None:
                    continue
                
                doi = external_ids.get("DOI")
                if doi:
                    doi_list.append(doi)
            except Exception:
                continue
        
        if verbose:
            print(f"     Found {len(doi_list)} DOIs")
        
        return doi_list
    
    except Exception:
        return []


def get_reference_papers(paper_id: str, verbose: bool = False) -> List[str]:
    """API Call #4: Get referenced papers' DOIs"""
    url = f"https://api.semanticscholar.org/graph/v1/paper/{paper_id}/references?fields=paperId,externalIds"
    
    try:
        response = requests.get(url)
        if response.status_code != 200:
            if verbose:
                print(f"     ⚠️  API returned status {response.status_code}")
            return []
        
        data = response.json()
        if data is None or not isinstance(data, dict):
            return []
        
        references = data.get('data', [])
        if references is None:
            return []
        
        if verbose:
            print(f"     Retrieved {len(references)} references from API")
        
        doi_list = []
        for reference in references:
            try:
                if reference is None:
                    continue
                    
                cited_paper = reference.get("citedPaper")
                if cited_paper is None:
                    continue
                
                external_ids = cited_paper.get("externalIds")
                if external_ids is None:
                    continue
                
                doi = external_ids.get("DOI")
                if doi:
                    doi_list.append(doi)
            except Exception:
                continue
        
        if verbose:
            print(f"     Found {len(doi_list)} DOIs")
        
        return doi_list
    
    except Exception:
        return []


def sanitize_doi_for_pinecone(doi: str) -> str:
    """Convert DOI to Pinecone-compatible ID"""
    return doi.lower().replace(".", "-").replace("/", "-")


def check_doi_exists_in_pinecone(doi: str) -> bool:
    """Check if a DOI already exists in Pinecone"""
    try:
        index = get_index()
        pinecone_id = sanitize_doi_for_pinecone(doi)
        
        # Try to fetch the vector
        result = index.fetch(ids=[pinecone_id])
        
        # If the ID exists, it will be in the vectors dict
        return pinecone_id in result.get('vectors', {})
    
    except Exception as e:
        # If there's an error, assume it doesn't exist
        return False


def upsert_to_pinecone(doi: str, embedding_data: Dict, metadata_dict: Dict = None, citing_dois: List[str] = None, reference_dois: List[str] = None):
    """Upsert embedding and metadata to Pinecone with citation relationships"""
    try:
        vector = embedding_data.get("vector")
        if not vector:
            raise ValueError("No embedding vector found")
        
        # Sanitize DOI for Pinecone ID
        pinecone_id = sanitize_doi_for_pinecone(doi)
        
        # Sanitize citation DOIs for storage
        sanitized_citing = [sanitize_doi_for_pinecone(d) for d in (citing_dois or [])]
        sanitized_references = [sanitize_doi_for_pinecone(d) for d in (reference_dois or [])]
        
        # Prepare metadata (excluding the large vector, but include citations)
        metadata = {
            "doi": doi,  # Store original DOI
            "title": embedding_data.get("title"),
            "authors": embedding_data.get("authors", []),
            "venue": embedding_data.get("venue"),
            "citation_count": embedding_data.get("citation_count"),
            "embedding_model": embedding_data.get("embedding_model"),
            "citing_paper_ids": sanitized_citing,  # Papers that cite this paper
            "reference_paper_ids": sanitized_references  # Papers this paper references
        }
        
        # Add additional metadata fields if provided (from get_paper_metadata)
        if metadata_dict:
            metadata["publication_date"] = metadata_dict.get("publication_date")
            metadata["fields_of_study"] = metadata_dict.get("fields_of_study", [])
            metadata["reference_count"] = metadata_dict.get("reference_count")
            metadata["influential_citation_count"] = metadata_dict.get("influential_citation_count")
        
        # Get or create index, then upsert using sanitized DOI as the ID
        index = get_index()
        index.upsert(vectors=[{
            "id": pinecone_id,
            "values": vector,
            "metadata": metadata
        }])
        
    except Exception as e:
        raise Exception(f"Error upserting to Pinecone: {str(e)}")


def print_separator():
    print("\n" + "="*80 + "\n")


def print_metadata(metadata):
    """Print paper metadata (API Call #1)"""
    print("📄 PAPER METADATA")
    print_separator()
    print(f"Title: {metadata.get('title')}")
    print(f"Venue: {metadata.get('venue')}")
    print(f"Publication Date: {metadata.get('publication_date')}")
    print(f"\nAuthors: {', '.join(metadata.get('authors', []))}")
    
    institutions = metadata.get('institutions', [])
    flat_institutions = []
    for inst_list in institutions:
        if inst_list:
            flat_institutions.extend(inst_list)
    if flat_institutions:
        print(f"Institutions: {', '.join(flat_institutions)}")
    
    print(f"\nCitation Count: {metadata.get('citation_count')}")
    print(f"Reference Count: {metadata.get('reference_count')}")
    print(f"Influential Citation Count: {metadata.get('influential_citation_count')}")
    
    fields = metadata.get('fields_of_study', [])
    if fields:
        print(f"Fields of Study: {', '.join(fields)}")


def print_citing_papers(citing_dois):
    """Print citing papers' DOIs (API Call #3)"""
    print("📚 CITING PAPERS (DOIs)")
    print_separator()
    if citing_dois:
        print(f"Found {len(citing_dois)} papers that cite this work:\n")
        for i, doi in enumerate(citing_dois, 1):
            print(f"{i}. {doi}")
    else:
        print("No citing papers found with DOIs.")


def print_reference_papers(reference_dois):
    """Print referenced papers' DOIs (API Call #4)"""
    print("📖 REFERENCED PAPERS (DOIs)")
    print_separator()
    if reference_dois:
        print(f"Found {len(reference_dois)} references with DOIs:\n")
        for i, doi in enumerate(reference_dois, 1):
            print(f"{i}. {doi}")
    else:
        print("No referenced papers found with DOIs.")


def main():
    print("\n" + "="*80)
    print("🔬 arXiv AI Paper Processing Tool")
    print("="*80 + "\n")
    
    # Get PDF path from environment variable
    pdf_path = os.getenv("PDF_PATH")
    
    if not pdf_path:
        print("❌ Error: PDF_PATH not found in .env.local")
        print("Please add: PDF_PATH=/path/to/your/paper.pdf")
        return
    
    # Validate file exists
    if not os.path.exists(pdf_path):
        print(f"❌ Error: File not found: {pdf_path}")
        return
    
    if not pdf_path.lower().endswith('.pdf'):
        print(f"❌ Error: File must be a PDF: {pdf_path}")
        return
    
    print(f"📄 Processing paper: {os.path.basename(pdf_path)}")
    print(f"📂 Path: {pdf_path}")
    
    # Step 1: Extract DOI
    print("\n⏳ Extracting DOI from PDF headers/footers...")
    doi = extract_doi_from_pdf(pdf_path)
    
    if not doi:
        print("❌ Error: DOI not found in PDF headers/footers.")
        print("Please check the PDF or try a different paper.")
        return
    
    print(f"✅ Found DOI: {doi}")
    
    # Step 2: Fetch data from Semantic Scholar
    print("\n⏳ Fetching data from Semantic Scholar API...")
    
    try:
        # Make all 4 API calls
        print("  → Getting paper metadata...")
        try:
            metadata = get_paper_metadata(doi)
            print(f"     ✓ Got metadata for: {metadata.get('title', 'Unknown')}")
        except Exception as e:
            print(f"     ❌ Failed to get metadata: {str(e)}")
            raise
        
        print("  → Getting paper embedding...")
        try:
            embedding_info = get_paper_embedding(doi)
            print(f"     ✓ Got embedding (dimension: {len(embedding_info.get('vector', []))})")
        except Exception as e:
            print(f"     ❌ Failed to get embedding: {str(e)}")
            raise
        
        print("  → Getting citing papers...")
        citing_dois = get_citing_papers(doi, verbose=True)
        print(f"     ✓ Found {len(citing_dois)} citing papers")
        
        print("  → Getting referenced papers...")
        reference_dois = get_reference_papers(doi, verbose=True)
        print(f"     ✓ Found {len(reference_dois)} referenced papers")
        
        # Step 3: Upsert to Pinecone with citation relationships
        print("\n💾 Storing embedding in Pinecone DB: arxAI...")
        try:
            upsert_to_pinecone(doi, embedding_info, metadata, citing_dois, reference_dois)
            print("     ✓ Stored in Pinecone successfully")
        except Exception as e:
            print(f"     ❌ Failed to store in Pinecone: {str(e)}")
            raise
        
        # Step 4: Process child papers with 2 levels of depth
        print("\n⏳ Building 2-level citation graph...")
        
        # Level 1: Direct citations and references
        level_1_dois = set(citing_dois + reference_dois)
        print(f"   Level 1: {len(level_1_dois)} papers ({len(citing_dois)} citing + {len(reference_dois)} referenced)")
        
        # Level 2: Citations and references of level-1 papers
        level_2_dois = set()
        print(f"\n⏳ Fetching level-2 citations (citations of citations)...")
        
        for i, l1_doi in enumerate(level_1_dois, 1):
            try:
                if i % 5 == 0:
                    print(f"   Fetching level-2 for paper {i}/{len(level_1_dois)}...")
                
                # Get citations and references for this level-1 paper
                l2_citing = get_citing_papers(l1_doi)
                l2_references = get_reference_papers(l1_doi)
                
                level_2_dois.update(l2_citing)
                level_2_dois.update(l2_references)
                
            except Exception as e:
                # Skip papers that fail
                continue
        
        # Remove duplicates with level 1 and parent
        level_2_dois -= level_1_dois
        level_2_dois.discard(doi)  # Remove parent paper
        
        print(f"   Level 2: {len(level_2_dois)} additional papers")
        
        # Combine all papers to process
        all_dois_to_process = list(level_1_dois) + list(level_2_dois)
        total_papers = len(all_dois_to_process)
        
        print(f"\n⏳ Processing {total_papers} papers total...")
        print("   Checking Pinecone for existing embeddings...")
        
        # Filter out papers already in Pinecone
        dois_to_fetch = []
        already_in_pinecone = 0
        
        for child_doi in all_dois_to_process:
            if check_doi_exists_in_pinecone(child_doi):
                already_in_pinecone += 1
            else:
                dois_to_fetch.append(child_doi)
        
        print(f"   ✓ Found {already_in_pinecone} papers already in Pinecone (skipping)")
        print(f"   → Need to fetch {len(dois_to_fetch)} new papers")
        
        if dois_to_fetch:
            print("\n⏳ Fetching embeddings for new papers...")
            print("   Note: This may take several minutes due to API rate limits...")
            
            successful_count = 0
            failed_count = 0
            
            for i, child_doi in enumerate(dois_to_fetch, 1):
                try:
                    if i % 10 == 0:
                        print(f"   Progress: {i}/{len(dois_to_fetch)}...")
                    
                    # Get embedding for child paper
                    child_embedding = get_paper_embedding(child_doi)
                    
                    # Get citations and references for proper graph construction
                    child_citing = get_citing_papers(child_doi)
                    child_references = get_reference_papers(child_doi)
                    
                    # Store with citation relationships (no extra metadata to save API calls)
                    upsert_to_pinecone(child_doi, child_embedding, None, child_citing, child_references)
                    successful_count += 1
                    
                except Exception as e:
                    failed_count += 1
                    if i % 10 == 0:
                        print(f"      ⚠️  Skipped paper {i} (no embedding available)")
            
            print(f"\n   ✅ Stored {successful_count} new papers")
            if failed_count > 0:
                print(f"   ⚠️  Skipped {failed_count} papers (no embeddings available)")
        else:
            print("\n   ✓ All papers already in Pinecone!")
        
        # Step 5: Display results
        print("\n✅ All papers processed successfully!")
        print_separator()
        
        # Print API Call #1 results
        print_metadata(metadata)
        print_separator()
        
        # Print API Call #3 results
        print_citing_papers(citing_dois)
        print_separator()
        
        # Print API Call #4 results
        print_reference_papers(reference_dois)
        print_separator()
        
        print("✅ Done!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return


if __name__ == "__main__":
    main()

