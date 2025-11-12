# arxai

Deep research assistant using citation graph traversal.

## Install

```bash
pip install -e .
```

## Usage

```bash
arxai ["pdf_path|doi|search_query"] [--author "Author"] [--topic "Topic"] [--title "Title"] [--institution "Institution"]
```

Examples:
```bash
arxai paper.pdf
arxai 10.1234/example.doi
arxai "transformer attention mechanisms"
arxai "transformer attention mechanisms" --author "Vaswani"
arxai "machine learning" --author "Bengio" --topic "neural networks" --institution "MIT"
arxai --author "koedinger" --title "algebra"
```

**Flags:**
- `--author`: Heavily weight papers by a specific author (handles misspellings, acts like a filter)
- `--topic`: Heavily weight papers matching a specific topic/keyword
- `--title`: Heavily weight papers with matching title text
- `--institution`: Heavily weight papers from a specific institution/venue

**Interactive Controls:**
- Arrow keys (←/→) to navigate
- Space to toggle select/unselect
- Enter to dive into selected papers
- q to quit

**Workflow:**
1. Search/load seed paper
2. Browse level-1 papers with interactive carousel
3. Toggle papers you want to explore (☐/☑)
4. Press Enter to dive deeper
5. Selected papers download to `papers/` directory
6. Level-2 search returns top 20 papers from selected citations
