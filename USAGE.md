# arXiv AI Paper Processing Tool - Usage Guide

## Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Environment Variables
Make sure your `.env.local` file contains:
```
PINECONE_API_KEY=your_pinecone_api_key
```

### 3. Pinecone Index Setup
Ensure you have a Pinecone index named `arxAI` with:
- Dimension: 768 (for SPECTER v2 embeddings)
- Metric: cosine (recommended for paper embeddings)

## Running the Tool

### Step 1: Start the FastAPI Server
In one terminal window:
```bash
python server.py
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 2: Run the CLI Client
In another terminal window:
```bash
python cli.py
```

### Step 3: Enter PDF Path
When prompted, enter the full path to your PDF paper:
```
Enter the path to your PDF paper: /path/to/your/paper.pdf
```

## What It Does

1. **Extracts DOI** from PDF headers/footers
2. **Makes 4 Semantic Scholar API Calls:**
   - Call #1: Paper metadata (title, authors, citations, etc.)
   - Call #2: Paper embedding vector (SPECTER v2)
   - Call #3: Citing papers' DOIs
   - Call #4: Referenced papers' DOIs
3. **Stores in Pinecone:** Only Call #2 (embedding + metadata) is stored
4. **Prints to Terminal:** Calls #1, #3, and #4 are displayed

## Example Output

```
🔬 arXiv AI Paper Processing Tool
================================================================================

Enter the path to your PDF paper: /Users/you/papers/attention.pdf
📤 Processing paper: attention.pdf
⏳ Extracting DOI and fetching data from Semantic Scholar...

✅ Paper processed successfully!
💾 Embedding stored in Pinecone DB: arxAI
================================================================================

📄 PAPER METADATA
================================================================================
Title: Attention Is All You Need
Venue: NIPS
Authors: Ashish Vaswani, Noam Shazeer, Niki Parmar...
Citation Count: 95847
Reference Count: 42
...

📚 CITING PAPERS (DOIs)
================================================================================
Found 1234 papers that cite this work:
1. 10.xxxx/xxxxx
2. 10.yyyy/yyyyy
...

📖 REFERENCED PAPERS (DOIs)
================================================================================
Found 42 references with DOIs:
1. 10.aaaa/aaaaa
2. 10.bbbb/bbbbb
...
```

## Troubleshooting

### DOI Not Found
If the DOI cannot be extracted from the PDF headers/footers, you'll see:
```
❌ Error: DOI not found in PDF headers/footers
```

**Solutions:**
- Check if the PDF has a DOI in the header or footer of the first page
- Try a different version of the PDF
- Manually look up the paper on Semantic Scholar

### Server Connection Error
```
❌ Error: Cannot connect to FastAPI server
```

**Solution:** Make sure the server is running in another terminal:
```bash
python server.py
```

### API Rate Limits
Semantic Scholar's free tier has rate limits. If you hit them, wait a few moments and try again.

## API Endpoints

The FastAPI server exposes:

- `GET /` - Health check
- `POST /process-paper` - Main processing endpoint
  - Request body: `{"pdf_path": "/path/to/file.pdf"}`
  - Returns: metadata, embedding info, citing DOIs, reference DOIs

