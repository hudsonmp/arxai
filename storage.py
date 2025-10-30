#!/usr/bin/env python3
"""
Supabase Storage Integration
Handles PDF uploads to Supabase storage buckets
"""
import os
from typing import Tuple, Optional
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment
load_dotenv(".env.local")

SUPABASE_URL = os.getenv("NEXT_PUBLIC_SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY")

# Initialize Supabase client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def upload_both_pdfs(original_path: str, annotated_path: str, paper_id: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Upload both original and annotated PDFs to Supabase storage
    Returns: (original_url, annotated_url)
    """
    try:
        original_filename = f"{paper_id}_original.pdf"
        annotated_filename = f"{paper_id}_annotated.pdf"
        
        # Upload original PDF
        print(f"      📤 Uploading original PDF...")
        with open(original_path, 'rb') as f:
            original_data = f.read()
        
        supabase.storage.from_("arxai-originals").upload(
            original_filename,
            original_data,
            file_options={"content-type": "application/pdf", "upsert": "true"}
        )
        
        # Upload annotated PDF
        print(f"      📤 Uploading annotated PDF...")
        with open(annotated_path, 'rb') as f:
            annotated_data = f.read()
        
        supabase.storage.from_("arxai-annotated").upload(
            annotated_filename,
            annotated_data,
            file_options={"content-type": "application/pdf", "upsert": "true"}
        )
        
        # Get public URLs
        original_url = f"{SUPABASE_URL}/storage/v1/object/public/arxai-originals/{original_filename}"
        annotated_url = f"{SUPABASE_URL}/storage/v1/object/public/arxai-annotated/{annotated_filename}"
        
        print(f"      ✅ Uploaded to Supabase successfully!")
        
        return original_url, annotated_url
    
    except Exception as e:
        print(f"      ❌ Upload error: {str(e)}")
        return None, None


def cleanup_local_files(*file_paths):
    """Clean up local PDF files after successful upload"""
    try:
        for file_path in file_paths:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"      🗑️  Cleaned up: {os.path.basename(file_path)}")
    except Exception as e:
        print(f"      ⚠️  Cleanup error: {str(e)}")
