#!/usr/bin/env python3
"""
Quality Research Agent using Claude API
Performs parallelized research on venues, institutions, and conferences
to provide quality scores for the algorithm
"""
import os
import json
import asyncio
import aiohttp
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
import anthropic

# Load environment variables
load_dotenv(".env.local")

# Initialize Anthropic client
client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))


# ============================================================================
# Research Agent Prompts
# ============================================================================

VENUE_RESEARCH_PROMPT = """You are an expert academic researcher specializing in conference and journal rankings.

Analyze the following academic venue: "{venue_name}"

Provide a quality score from 0.0 to 1.0 based on:
1. **Impact Factor / H-Index**: For journals, consider impact factor and h-index. For conferences, consider acceptance rates and citation metrics.
2. **Reputation**: Community recognition, selectivity, and prestige in the field.
3. **CORE Ranking**: If it's a conference, check CORE ranking (A*, A, B, C, Unranked).
4. **Historical Significance**: Longevity and influence in the field.

IMPORTANT: You must respond with ONLY a JSON object in this exact format:
{{
  "venue_name": "{venue_name}",
  "quality_score": 0.85,
  "tier": "A*",
  "reasoning": "Brief explanation of the score",
  "acceptance_rate": "20%",
  "impact_factor": "4.5"
}}

If the venue is unknown or you cannot find information, return:
{{
  "venue_name": "{venue_name}",
  "quality_score": 0.5,
  "tier": "Unknown",
  "reasoning": "Insufficient information available",
  "acceptance_rate": "Unknown",
  "impact_factor": "Unknown"
}}

Do NOT include any text before or after the JSON. Only return valid JSON."""

INSTITUTION_RESEARCH_PROMPT = """You are an expert academic researcher specializing in university and institution rankings.

Analyze the following institution: "{institution_name}"

Provide a quality score from 0.0 to 1.0 based on:
1. **Global Rankings**: QS World University Rankings, Times Higher Education, Academic Ranking of World Universities (ARWU).
2. **Research Output**: Quality and quantity of research publications.
3. **CS/AI Department Strength**: Specific strength in computer science and AI research.
4. **Notable Faculty and Alumni**: Influential researchers and Nobel laureates.

IMPORTANT: You must respond with ONLY a JSON object in this exact format:
{{
  "institution_name": "{institution_name}",
  "quality_score": 0.92,
  "global_rank": 15,
  "cs_rank": 8,
  "reasoning": "Brief explanation of the score",
  "notable_faculty": ["Name1", "Name2"]
}}

If the institution is unknown or you cannot find information, return:
{{
  "institution_name": "{institution_name}",
  "quality_score": 0.5,
  "global_rank": null,
  "cs_rank": null,
  "reasoning": "Insufficient information available",
  "notable_faculty": []
}}

Do NOT include any text before or after the JSON. Only return valid JSON."""

AUTHOR_RESEARCH_PROMPT = """You are an expert academic researcher specializing in author metrics and scholarly impact.

Analyze the following author: "{author_name}"

Provide a quality score from 0.0 to 1.0 based on:
1. **H-Index**: Author's h-index from Google Scholar or similar.
2. **Citation Count**: Total citations of their work.
3. **Notable Publications**: Influential papers and contributions to the field.
4. **Awards and Recognition**: Turing Award, ACM Fellow, IEEE Fellow, etc.

IMPORTANT: You must respond with ONLY a JSON object in this exact format:
{{
  "author_name": "{author_name}",
  "quality_score": 0.88,
  "h_index": 45,
  "total_citations": 25000,
  "reasoning": "Brief explanation of the score",
  "awards": ["ACM Fellow", "Best Paper Award"]
}}

If the author is unknown or you cannot find information, return:
{{
  "author_name": "{author_name}",
  "quality_score": 0.5,
  "h_index": null,
  "total_citations": null,
  "reasoning": "Insufficient information available",
  "awards": []
}}

Do NOT include any text before or after the JSON. Only return valid JSON."""


# ============================================================================
# Async Research Functions
# ============================================================================

async def research_venue(venue_name: str, session: aiohttp.ClientSession = None) -> Dict:
    """
    Research a single venue using Claude API
    Returns quality score and metadata
    """
    if not venue_name or venue_name.lower() in ["unknown", "none", ""]:
        return {
            "venue_name": venue_name,
            "quality_score": 0.5,
            "tier": "Unknown",
            "reasoning": "No venue specified"
        }
    
    try:
        prompt = VENUE_RESEARCH_PROMPT.format(venue_name=venue_name)
        
        # Make synchronous call to Claude API (Anthropic SDK doesn't support async yet)
        loop = asyncio.get_event_loop()
        message = await loop.run_in_executor(
            None,
            lambda: client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
        )
        
        # Parse response
        response_text = message.content[0].text.strip()
        
        # Extract JSON from response (in case there's extra text)
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        result = json.loads(response_text)
        return result
    
    except Exception as e:
        print(f"⚠️  Error researching venue '{venue_name}': {str(e)}")
        return {
            "venue_name": venue_name,
            "quality_score": 0.5,
            "tier": "Unknown",
            "reasoning": f"Error: {str(e)}"
        }


async def research_institution(institution_name: str, session: aiohttp.ClientSession = None) -> Dict:
    """
    Research a single institution using Claude API
    Returns quality score and metadata
    """
    if not institution_name or institution_name.lower() in ["unknown", "none", ""]:
        return {
            "institution_name": institution_name,
            "quality_score": 0.5,
            "reasoning": "No institution specified"
        }
    
    try:
        prompt = INSTITUTION_RESEARCH_PROMPT.format(institution_name=institution_name)
        
        # Make synchronous call to Claude API
        loop = asyncio.get_event_loop()
        message = await loop.run_in_executor(
            None,
            lambda: client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
        )
        
        # Parse response
        response_text = message.content[0].text.strip()
        
        # Extract JSON from response
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        result = json.loads(response_text)
        return result
    
    except Exception as e:
        print(f"⚠️  Error researching institution '{institution_name}': {str(e)}")
        return {
            "institution_name": institution_name,
            "quality_score": 0.5,
            "reasoning": f"Error: {str(e)}"
        }


async def research_author(author_name: str, session: aiohttp.ClientSession = None) -> Dict:
    """
    Research a single author using Claude API
    Returns quality score and metadata
    """
    if not author_name or author_name.lower() in ["unknown", "none", ""]:
        return {
            "author_name": author_name,
            "quality_score": 0.5,
            "reasoning": "No author specified"
        }
    
    try:
        prompt = AUTHOR_RESEARCH_PROMPT.format(author_name=author_name)
        
        # Make synchronous call to Claude API
        loop = asyncio.get_event_loop()
        message = await loop.run_in_executor(
            None,
            lambda: client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
        )
        
        # Parse response
        response_text = message.content[0].text.strip()
        
        # Extract JSON from response
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        result = json.loads(response_text)
        return result
    
    except Exception as e:
        print(f"⚠️  Error researching author '{author_name}': {str(e)}")
        return {
            "author_name": author_name,
            "quality_score": 0.5,
            "reasoning": f"Error: {str(e)}"
        }


# ============================================================================
# Parallel Research Functions
# ============================================================================

async def research_venues_parallel(venues: List[str]) -> Dict[str, Dict]:
    """
    Research multiple venues in parallel
    Returns: {venue_name: quality_data}
    """
    if not venues:
        return {}
    
    # Remove duplicates and empty strings
    unique_venues = list(set([v for v in venues if v and v.strip()]))
    
    print(f"🔍 Researching {len(unique_venues)} unique venues in parallel...")
    
    # Create tasks for parallel execution
    async with aiohttp.ClientSession() as session:
        tasks = [research_venue(venue, session) for venue in unique_venues]
        results = await asyncio.gather(*tasks)
    
    # Build dictionary
    venue_dict = {}
    for result in results:
        venue_name = result.get("venue_name")
        if venue_name:
            venue_dict[venue_name] = result
    
    return venue_dict


async def research_institutions_parallel(institutions: List[str]) -> Dict[str, Dict]:
    """
    Research multiple institutions in parallel
    Returns: {institution_name: quality_data}
    """
    if not institutions:
        return {}
    
    # Remove duplicates and empty strings
    unique_institutions = list(set([i for i in institutions if i and i.strip()]))
    
    print(f"🏛️  Researching {len(unique_institutions)} unique institutions in parallel...")
    
    # Create tasks for parallel execution
    async with aiohttp.ClientSession() as session:
        tasks = [research_institution(inst, session) for inst in unique_institutions]
        results = await asyncio.gather(*tasks)
    
    # Build dictionary
    institution_dict = {}
    for result in results:
        institution_name = result.get("institution_name")
        if institution_name:
            institution_dict[institution_name] = result
    
    return institution_dict


async def research_authors_parallel(authors: List[str]) -> Dict[str, Dict]:
    """
    Research multiple authors in parallel
    Returns: {author_name: quality_data}
    """
    if not authors:
        return {}
    
    # Remove duplicates and empty strings
    unique_authors = list(set([a for a in authors if a and a.strip()]))
    
    print(f"👤 Researching {len(unique_authors)} unique authors in parallel...")
    
    # Create tasks for parallel execution
    async with aiohttp.ClientSession() as session:
        tasks = [research_author(author, session) for author in unique_authors]
        results = await asyncio.gather(*tasks)
    
    # Build dictionary
    author_dict = {}
    for result in results:
        author_name = result.get("author_name")
        if author_name:
            author_dict[author_name] = result
    
    return author_dict


async def research_all_parallel(venues: List[str], institutions: List[str], authors: List[str]) -> Tuple[Dict, Dict, Dict]:
    """
    Research venues, institutions, and authors in parallel
    Returns: (venue_dict, institution_dict, author_dict)
    """
    print("🚀 Starting parallel research across all categories...")
    
    # Run all three research tasks concurrently
    venue_task = research_venues_parallel(venues)
    institution_task = research_institutions_parallel(institutions)
    author_task = research_authors_parallel(authors)
    
    venue_dict, institution_dict, author_dict = await asyncio.gather(
        venue_task,
        institution_task,
        author_task
    )
    
    print("✅ Parallel research complete!")
    return venue_dict, institution_dict, author_dict


# ============================================================================
# Cache Management
# ============================================================================

def save_research_cache(venue_dict: Dict, institution_dict: Dict, author_dict: Dict, cache_file: str = "quality_cache.json"):
    """Save research results to cache file"""
    try:
        cache_data = {
            "venues": venue_dict,
            "institutions": institution_dict,
            "authors": author_dict
        }
        
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)
        
        print(f"💾 Research cache saved to {cache_file}")
    
    except Exception as e:
        print(f"⚠️  Error saving cache: {str(e)}")


def load_research_cache(cache_file: str = "quality_cache.json") -> Tuple[Dict, Dict, Dict]:
    """Load research results from cache file"""
    try:
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            venues = cache_data.get("venues", {})
            institutions = cache_data.get("institutions", {})
            authors = cache_data.get("authors", {})
            
            print(f"📂 Loaded cache: {len(venues)} venues, {len(institutions)} institutions, {len(authors)} authors")
            return venues, institutions, authors
        else:
            return {}, {}, {}
    
    except Exception as e:
        print(f"⚠️  Error loading cache: {str(e)}")
        return {}, {}, {}


# ============================================================================
# Main Function for Testing
# ============================================================================

async def main():
    """Test the research system"""
    print("\n" + "="*80)
    print("🧪 Quality Research Agent Test")
    print("="*80 + "\n")
    
    # Test data
    test_venues = [
        "NeurIPS",
        "ICML",
        "ICLR",
        "Nature",
        "Science",
        "CVPR",
        "ACL"
    ]
    
    test_institutions = [
        "Stanford University",
        "MIT",
        "Carnegie Mellon University",
        "UC Berkeley",
        "Oxford University"
    ]
    
    test_authors = [
        "Geoffrey Hinton",
        "Yann LeCun",
        "Yoshua Bengio",
        "Andrew Ng"
    ]
    
    # Run parallel research
    venue_dict, institution_dict, author_dict = await research_all_parallel(
        test_venues,
        test_institutions,
        test_authors
    )
    
    # Display results
    print("\n" + "="*80)
    print("📊 RESEARCH RESULTS")
    print("="*80 + "\n")
    
    print("🏆 VENUE SCORES:")
    for venue, data in sorted(venue_dict.items(), key=lambda x: x[1].get("quality_score", 0), reverse=True):
        score = data.get("quality_score", 0)
        tier = data.get("tier", "Unknown")
        print(f"  {venue}: {score:.3f} (Tier: {tier})")
    
    print("\n🏛️  INSTITUTION SCORES:")
    for inst, data in sorted(institution_dict.items(), key=lambda x: x[1].get("quality_score", 0), reverse=True):
        score = data.get("quality_score", 0)
        rank = data.get("global_rank", "N/A")
        print(f"  {inst}: {score:.3f} (Global Rank: {rank})")
    
    print("\n👤 AUTHOR SCORES:")
    for author, data in sorted(author_dict.items(), key=lambda x: x[1].get("quality_score", 0), reverse=True):
        score = data.get("quality_score", 0)
        h_index = data.get("h_index", "N/A")
        print(f"  {author}: {score:.3f} (H-Index: {h_index})")
    
    # Save to cache
    save_research_cache(venue_dict, institution_dict, author_dict)


if __name__ == "__main__":
    asyncio.run(main())

