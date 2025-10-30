#!/usr/bin/env python3
"""
Upload local PDFs to Supabase storage buckets
"""
import os
from supabase import create_client
from dotenv import load_dotenv
import glob

load_dotenv(".env.local")

supabase_url = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
supabase_key = os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")

if not supabase_url or not supabase_key:
    print("❌ Supabase credentials not found in .env.local")
    exit(1)

supabase = create_client(supabase_url, supabase_key)

def upload_pdfs():
    """Upload annotated and original PDFs to Supabase"""
    
    # Find all annotated PDFs
    annotated_pdfs = glob.glob("/Users/hudsonmitchell-pullman/arxai/fallback/*_annotated.pdf")
    original_pdfs = glob.glob("/Users/hudsonmitchell-pullman/arxai/fallback/*_original.pdf")
    
    print(f"\n📤 Uploading {len(annotated_pdfs)} annotated PDFs...")
    
    for pdf_path in annotated_pdfs:
        filename = os.path.basename(pdf_path)
        try:
            with open(pdf_path, 'rb') as f:
                supabase.storage.from_('arxai-annotated').upload(
                    filename,
                    f,
                    file_options={"content-type": "application/pdf"}
                )
            print(f"   ✅ {filename}")
        except Exception as e:
            if "already exists" in str(e):
                print(f"   ⏭️  {filename} (already exists)")
            else:
                print(f"   ❌ {filename}: {str(e)}")
    
    print(f"\n📤 Uploading {len(original_pdfs)} original PDFs...")
    
    for pdf_path in original_pdfs:
        filename = os.path.basename(pdf_path)
        try:
            with open(pdf_path, 'rb') as f:
                supabase.storage.from_('arxai-originals').upload(
                    filename,
                    f,
                    file_options={"content-type": "application/pdf"}
                )
            print(f"   ✅ {filename}")
        except Exception as e:
            if "already exists" in str(e):
                print(f"   ⏭️  {filename} (already exists)")
            else:
                print(f"   ❌ {filename}: {str(e)}")
    
    print("\n✅ Upload complete!")

if __name__ == "__main__":
    upload_pdfs()

