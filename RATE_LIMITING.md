# Rate Limiting Implementation

## Overview
Added 30,000 tokens per second (TPS) rate limiting to the arXiv paper annotation pipeline to comply with Claude API usage limits.

## What Was Changed

### 1. **arxiv_annotator.py**
Added a `TokenRateLimiter` class that:
- Tracks token usage in a sliding 1-second window
- Estimates token count from text (1 token ≈ 4 characters)
- Automatically pauses requests when approaching the 30k TPS limit
- Provides visual feedback showing current token usage

**Key Features:**
- Async-safe with `asyncio.Lock` to prevent race conditions
- Estimates both input tokens (prompt) and output tokens (max_tokens=2048)
- Automatically resets the token counter every second
- Shows real-time token usage: `📊 Token usage: 15234/30000 TPS`

### 2. **direct_ranking.py**
Updated the main pipeline to:
- Pass parent paper title and abstract to child papers for contextual annotation
- Display rate limiting status in pipeline output
- Ensure parent paper context is available for all annotation calls

## How It Works

```python
# Before making an API call:
1. Estimate tokens: input_tokens + output_tokens (2048)
2. Check if adding these tokens exceeds 30k limit
3. If yes, wait until the next minute window
4. If no, proceed and track token usage
```

## Usage

Simply run the pipeline as before:
```bash
python direct_ranking.py
```

The rate limiter will automatically:
- Track token usage across all parallel requests
- Pause when necessary to stay under 30k TPM
- Display status messages when rate limiting is active

## Example Output

```
📊 Token usage: 8540/30000 TPM
📊 Token usage: 15234/30000 TPM
📊 Token usage: 22890/30000 TPM
⏸️  Rate limit approaching, waiting 12.3s...
📊 Token usage: 2340/30000 TPM  # Reset after waiting
```

## Rate Limiting Strategy

- **Conservative approach**: Estimates include full max_tokens (2048) even if actual usage is lower
- **Async-safe**: Uses locks to prevent race conditions with parallel requests
- **Automatic recovery**: Resets every 60 seconds
- **Visual feedback**: Shows token usage and wait times

## Notes

- The existing semaphore (max_concurrent=5) still controls concurrency
- Rate limiting is applied at the API call level, not at the paper level
- Token estimation is approximate but conservative to avoid exceeding limits

