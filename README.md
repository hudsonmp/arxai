# arxai
launches deep research with parallel ai and claude's agentic SDK - SDx 10/2025

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Setup
Create a `.env.local` file with your API keys:
```bash
PINECONE_API_KEY=your_pinecone_api_key_here
CLAUDE_API_KEY=your_claude_api_key_here
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_service_role_key
```

Note: The system uses two Supabase storage buckets:
- `arxai-originals` - stores original downloaded PDFs
- `arxai-annotated` - stores PDFs with Claude's highlights and comments

### 🎨 New Annotation Features

**Collapsible Comments:** Every highlight now includes a comment explaining how it relates to your parent paper!
**Relevance Scores:** Each PDF displays its relevance score in bold at the top of the first page.

See `ANNOTATION_FEATURES.md` for full documentation.

### Running the Server
```bash
python arxai.py
```

The server will start at `http://0.0.0.0:8000`
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/

### Using the CLI Client
```bash
python cli.py
```

Or make API calls directly:
```bash
curl -X POST "http://localhost:8000/process-paper" \
  -H "Content-Type: application/json" \
  -d '{"pdf_path": "/path/to/your/paper.pdf"}'
```

---

## Project Overview

When approaching research, we need to approach it from a perspective of a literature basis instead of a paper-by-paper basis. We need collaborative multi-agent research that basically simulates a lab, but instead of just doing general research, it uses a nearest neighbor paper to mine for related works and citations. LLMs are really good for finding papers, analyzing them, etc. but instead of just outputing a natural language summary or reviewing the paper, it would be cool to download them as PDFs in our workspace, have our agent highlight the sections of the paper that are relevant to our research. What should not be automated to AI is drawing conclusions, making inferences, creating research proposals, and developing novel studies, but there are a lot of barriers for researchers that hinder their ability to do this.
Here are the barriers:
1. The main indicator of an insightful and successful paper is citations because there is a very larger barrier to entry to even read an abstract and determine whether the paper is useful, but with LLMs, we can determine this very quickly.
   1. Literature reviews
   2. Present the information in a way that is optimized for synthesis
   3. Interpert data??

Human will consider the nearest neighbor paper.

Relevance this will be determined soon

Recursively discover relevant papers through the citation graph

The information should be presented in boring UI like Overleaf etc.

Create a Citation Graph (visual)

Paper metadata

Vector database

We must stream in realtime and lag is a feature not a bug


```Great question! This is essentially a **graph search problem with relevance propagation** where papers are nodes and citations are edges. Let me break down an elegant solution:

## The Core Algorithm: Priority-Weighted Best-First Search

Think of it like Dijkstra's algorithm, but instead of finding shortest paths, you're finding the most relevant papers.

### The Relevance Score Function

Each paper gets a relevance score:

**R(paper) = R(parent) × similarity(paper, query) × α**

Where:
- **R(parent)** is the relevance of the paper that cited this one
- **similarity(paper, query)** is how semantically close the paper is to your original query (0 to 1)
- **α** is a decay factor, like α = 0.85 (this is crucial!)

### Why This Works Mathematically

The decay factor α prevents exponential explosion through **geometric decay**. If you start with R₀ = 1, after k hops:

R_k ≤ α^k

Since α < 1, this converges! In fact, the sum of all possible relevance scores is bounded:

Σ(k=0 to ∞) α^k = 1/(1-α)

For α = 0.85, maximum total relevance across *all* branches = 1/0.15 ≈ 6.67 times your starting relevance. This guarantees convergence!

### The Multi-Agent Orchestra

**Agent 1: Scout** 
- Fetches papers and their citations
- Maintains a **priority queue** (max heap) ordered by relevance score
- Always explores the highest-relevance unexplored paper next

**Agent 2: Evaluator**
- Computes semantic embeddings (vector representations of papers)
- Calculates similarity scores using cosine similarity: 
  - similarity = (v₁ · v₂)/(||v₁|| ||v₂||)
- This gives you that 0-1 score for the relevance function

**Agent 3: Librarian**
- **Full storage threshold**: Papers with R > τ_high (e.g., 0.3)
- **Summary storage**: Papers with τ_low < R < τ_high
- **Discard**: Papers with R < τ_low
- This creates a tiered knowledge base

**Agent 4: Navigator (RAG)**
- Uses vector database of stored papers
- When you ask questions, retrieves relevant chunks
- Can recursively trigger Scout if you need deeper exploration of a sub-topic

### Preventing Explosion: The Three Guards

**1. Relevance Threshold (ε-cutoff)**
Stop exploring any branch where R(paper) < ε (e.g., ε = 0.05)

Since R decreases geometrically, this naturally prunes distant papers.

**2. Width Limiting**
From any paper, only explore the top-k most relevant citations (e.g., k = 5)

Expected papers explored ≤ k/(1-α) for each starting paper

With k=5 and α=0.85: ~33 papers max per branch

**3. Global Budget**
Cap total papers explored at N (e.g., 1000)

### The Convergence Proof (Simplified)

You're basically doing **weighted breadth-first search** where weights decrease geometrically. 

Let d_max be the depth where α^d_max < ε:

d_max = log(ε)/log(α)

For ε = 0.05, α = 0.85: d_max ≈ 18 hops

Maximum papers with branching factor b: Σ(i=0 to d_max) b^i, but with top-k limiting and relevance pruning, most branches die early.

**Empirically**: You'll explore ~100-500 papers but store everything, presenting only the top ~20-50 with R > τ_high.

### The Information Theory View

This is related to **information gain**. Each paper adds information I(paper):

I(paper) = -log₂(P(paper|query))

Papers with high relevance have high information gain. The decay factor represents **diminishing returns**: papers further from your query tell you less new information.

### Practical Flow

1. Start: Your seed paper/query has R = 1.0
2. Scout adds it to priority queue
3. Pop highest R paper, Evaluator scores all its citations
4. Push citations to queue with updated R scores
5. Librarian decides storage tier
6. Repeat until queue empty or budget exhausted
7. Present top-R papers to user
8. Navigator handles follow-up questions via RAG

### Why This Beats Naive Recursion

Naive recursion: **O(b^d)** - exponential!

This algorithm: **O(N log N)** where N is your budget, due to priority queue operations

The geometric decay ensures you explore broadly near the root (high relevance) and sparsely at the leaves (low relevance), naturally creating a relevance-weighted sampling of the paper space.

Does this align with what you were imagining? I can dive deeper into any part—the multi-armed bandit variation for exploration-exploitation tradeoffs, or the clustering approaches for redundancy elimination?
```