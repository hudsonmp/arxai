#!/usr/bin/env python3
"""
CLI client for arXiv AI Paper Processing Tool
"""
import requests
import sys
from pathlib import Path


BASE_URL = "http://localhost:8000"


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


def process_paper(pdf_path: str):
    """Send PDF path to FastAPI server and display results"""
    # Validate file exists
    file_path = Path(pdf_path).resolve()
    if not file_path.exists():
        print(f"❌ Error: File not found: {pdf_path}")
        sys.exit(1)
    
    if not file_path.suffix.lower() == '.pdf':
        print(f"❌ Error: File must be a PDF: {pdf_path}")
        sys.exit(1)
    
    print(f"📤 Processing paper: {file_path.name}")
    print("⏳ Extracting DOI and fetching data from Semantic Scholar...")
    
    try:
        # Make POST request to FastAPI server
        response = requests.post(
            f"{BASE_URL}/process-paper",
            json={"pdf_path": str(file_path)},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            
            print("\n✅ Paper processed successfully!")
            print("💾 Embedding stored in Pinecone DB: arxAI")
            print_separator()
            
            # Print API Call #1 results
            print_metadata(data.get("metadata", {}))
            print_separator()
            
            # Print API Call #3 results
            print_citing_papers(data.get("citing_dois", []))
            print_separator()
            
            # Print API Call #4 results
            print_reference_papers(data.get("reference_dois", []))
            print_separator()
            
            print("✅ Done!")
            
        else:
            error_detail = response.json().get("detail", "Unknown error")
            print(f"\n❌ Error: {error_detail}")
            sys.exit(1)
            
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Cannot connect to FastAPI server.")
        print("Please make sure the server is running:")
        print("  python arxai.py")
        sys.exit(1)
    except requests.exceptions.Timeout:
        print("\n❌ Error: Request timed out. The server may be busy.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1)


def main():
    print("\n" + "="*80)
    print("🔬 arXiv AI Paper Processing Tool")
    print("="*80 + "\n")
    
    # Prompt for PDF path
    pdf_path = input("Enter the path to your PDF paper: ").strip()
    
    # Remove quotes if user wrapped path in quotes
    pdf_path = pdf_path.strip('"').strip("'")
    
    if not pdf_path:
        print("❌ Error: No path provided.")
        sys.exit(1)
    
    process_paper(pdf_path)


if __name__ == "__main__":
    main()

