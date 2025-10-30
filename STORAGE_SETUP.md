# Supabase Storage Setup Guide

## Overview

This project now stores PDFs in Supabase storage buckets instead of local files. This provides:
- ✅ Centralized cloud storage
- ✅ Public URLs for easy sharing
- ✅ Automatic backup and redundancy
- ✅ No local disk usage

## Architecture

### Two Storage Buckets

1. **arxai-originals** - Stores original downloaded PDFs
   - Clean versions from arXiv/Semantic Scholar
   - Source material for processing

2. **arxai-annotated** - Stores processed PDFs
   - Contains Claude's highlights
   - Ready for review and analysis

### File Structure

```
Flat structure with paper IDs:
arxai-originals/
  └── {paper_id}.pdf

arxai-annotated/
  └── {paper_id}.pdf
```

## Setup Instructions

### 1. Environment Variables

Add these to your `.env.local` file:

```bash
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-service-role-key-here
```

**Getting your keys:**
1. Go to your Supabase project dashboard
2. Navigate to Settings > API
3. Copy your project URL
4. Copy your **service_role** key (not the anon key)

### 2. Buckets Created

The storage buckets are already created with this migration:
- Public read access (anyone with URL can view PDFs)
- Authenticated write access (only your app can upload)
- 50MB file size limit per PDF
- Only `application/pdf` MIME type allowed

## Usage

### Automatic Storage in Pipeline

The processing pipelines automatically handle storage:

**arxiv_annotator.py:**
```python
# Downloads → Annotates → Uploads to both buckets → Cleans up local files
result = await process_paper_with_highlights(paper)
print(result['original_pdf_url'])   # Supabase URL
print(result['annotated_pdf_url'])  # Supabase URL
```

**pdf_processor.py:**
```python
# Downloads → Extracts → Annotates → Uploads → Cleans up
result = await process_paper(paper_data)
print(result['pdf_url'])  # Supabase URL
```

### Manual Storage Operations

Use the `storage.py` utility:

```python
from storage import upload_pdf_to_storage, upload_both_pdfs, cleanup_local_files

# Upload single PDF
url = upload_pdf_to_storage(
    file_path="./my_paper.pdf",
    paper_id="my-paper-id",
    bucket_type="annotated"  # or "original"
)

# Upload both original and annotated
original_url, annotated_url = upload_both_pdfs(
    original_path="./original.pdf",
    annotated_path="./annotated.pdf",
    paper_id="my-paper-id"
)

# Clean up local files after successful upload
cleanup_local_files("./original.pdf", "./annotated.pdf")
```

## File Lifecycle

```
1. Download PDF from source → local temp file
2. Process/annotate → local annotated file
3. Upload both to Supabase → get public URLs
4. Delete local files → clean workspace
5. Store URLs in database metadata
```

## Accessing PDFs

### Public URLs

All PDFs are publicly accessible via their URLs:

```
https://cbwylfksojmasehzmpum.supabase.co/storage/v1/object/public/arxai-originals/{paper_id}.pdf
https://cbwylfksojmasehzmpum.supabase.co/storage/v1/object/public/arxai-annotated/{paper_id}.pdf
```

### Viewing in Browser

Simply paste the URL in any browser to view the PDF.

### Downloading

Use `curl` or any HTTP client:
```bash
curl -O https://your-project.supabase.co/storage/v1/object/public/arxai-annotated/paper-123.pdf
```

## Backup Considerations

### Existing Local Files

The `fallback/` and `annotated_pdfs/` directories contain PDFs from before the storage migration. Options:

1. **Upload to Supabase:**
   ```python
   from storage import upload_pdf_to_storage
   import os
   
   for file in os.listdir("./fallback"):
       if file.endswith("_annotated.pdf"):
           paper_id = file.replace("_annotated.pdf", "")
           upload_pdf_to_storage(f"./fallback/{file}", paper_id, "annotated")
   ```

2. **Keep as local backup** (add to .gitignore)

3. **Delete after verification** (saves disk space)

## Troubleshooting

### Upload Failures

If upload fails, the system keeps local files and returns a `partial_success` status:

```python
if result['status'] == 'partial_success':
    print(f"Local files: {result['original_pdf']}, {result['annotated_pdf']}")
    # Manually retry upload
```

### Missing Environment Variables

Error: `ValueError: SUPABASE_URL and SUPABASE_KEY must be set`

Solution: Add the variables to `.env.local`

### Permission Errors

Error: `403 Forbidden` or `401 Unauthorized`

Solution: Make sure you're using the **service_role** key, not the anon key

## Storage Limits

**Current Settings:**
- File size limit: 50 MB per PDF
- Buckets: Public read, authenticated write
- MIME type: application/pdf only

**To modify limits:**
```sql
-- Update file size limit (100 MB)
UPDATE storage.buckets 
SET file_size_limit = 104857600 
WHERE id IN ('arxai-originals', 'arxai-annotated');
```

## Migration Script

To upload all existing local PDFs to Supabase:

```python
#!/usr/bin/env python3
"""Upload existing local PDFs to Supabase storage"""
import os
from storage import upload_pdf_to_storage

def migrate_local_pdfs():
    directories = [
        ("./fallback", "annotated"),
        ("./annotated_pdfs", "annotated"),
    ]
    
    for directory, bucket_type in directories:
        if not os.path.exists(directory):
            continue
            
        for filename in os.listdir(directory):
            if not filename.endswith(".pdf"):
                continue
            
            # Extract paper ID
            paper_id = filename.replace("_annotated.pdf", "").replace("_original.pdf", "")
            file_path = os.path.join(directory, filename)
            
            # Determine bucket type from filename
            if "_original.pdf" in filename:
                bucket = "original"
            else:
                bucket = "annotated"
            
            print(f"Uploading {filename}...")
            url = upload_pdf_to_storage(file_path, paper_id, bucket)
            
            if url:
                print(f"  ✅ {url}")
            else:
                print(f"  ❌ Failed")

if __name__ == "__main__":
    migrate_local_pdfs()
```

## Next Steps

1. ✅ Buckets created and configured
2. ✅ Code updated to use Supabase storage
3. ✅ Local cleanup implemented
4. 🔄 (Optional) Migrate existing local PDFs
5. 🔄 (Optional) Update database schema to store URLs
6. 🔄 (Optional) Build viewer UI to browse stored PDFs

## Related Files

- `storage.py` - Storage utility functions
- `arxiv_annotator.py` - PDF annotation pipeline with storage
- `pdf_processor.py` - PDF processing pipeline with storage
- `requirements.txt` - Now includes `supabase==2.3.4`

