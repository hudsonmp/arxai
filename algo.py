#!/usr/bin/env python3
"""
Bidirectional Priority Search Algorithm
Implements the algorithm from algorithm.md using Pinecone data
"""
import os
import heapq
import math
import asyncio
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
from dotenv import load_dotenv
from pinecone import Pinecone
import numpy as np
from quality_research import (
    research_all_parallel,
    load_research_cache,
    save_research_cache
)

# Load environment variables
load_dotenv(".env.local")

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("arxai")


# ============================================================================
# Core Data Structures
# ============================================================================

class Paper:
    """Represents a paper in the search"""
    def __init__(self, paper_id: str, metadata: Dict, vector: List[float]):
        self.id = paper_id
        self.doi = metadata.get("doi", paper_id)
        self.title = metadata.get("title", "Unknown")
        self.abstract = metadata.get("abstract", "")
        self.authors = metadata.get("authors", [])
        self.venue = metadata.get("venue", "Unknown")
        self.citation_count = metadata.get("citation_count", 0)
        self.publication_date = metadata.get("publication_date", "")
        self.fields_of_study = metadata.get("fields_of_study", [])
        self.reference_count = metadata.get("reference_count", 0)
        self.influential_citation_count = metadata.get("influential_citation_count", 0)
        self.vector = np.array(vector)
        
        # Citation relationships from Pinecone metadata
        self.citing_paper_ids = metadata.get("citing_paper_ids", [])
        self.reference_paper_ids = metadata.get("reference_paper_ids", [])
        
        # Search state
        self.relevance_score = 0.0
        self.depth = 0
        self.parent_id = None
        self.direction = None  # "forward" or "backward"


class PriorityQueueItem:
    """Item for priority queue (max-heap by relevance)"""
    def __init__(self, paper_id: str, relevance: float, depth: int, direction: str):
        self.paper_id = paper_id
        self.relevance = relevance
        self.depth = depth
        self.direction = direction
    
    def __lt__(self, other):
        # Python's heapq is min-heap, so negate for max-heap behavior
        return self.relevance > other.relevance


class CandidateInfo:
    """Stores information about candidate papers for lazy evaluation"""
    def __init__(self, paper_id: str, base_relevance: float, composite_score: float):
        self.paper_id = paper_id
        self.base_relevance = base_relevance
        self.composite_score = composite_score
        self.citing_papers = []  # Papers that led to this candidate
        self.count = 0  # Duplication counter
        self.boosted_relevance = base_relevance


# ============================================================================
# Helper Functions
# ============================================================================

def sanitize_doi_for_pinecone(doi: str) -> str:
    """Convert DOI to Pinecone-compatible ID"""
    return doi.lower().replace(".", "-").replace("/", "-")


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors"""
    if len(vec1) == 0 or len(vec2) == 0:
        return 0.0
    
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def compute_age_normalized_citation_score(citation_count: int, publication_date: str, lambda_param: float = 0.5) -> float:
    """
    Compute age-normalized citation score to prevent old paper bias
    Q_citations(p) = log(1 + c(p)) / (log(1 + c(p)) + λ * age(p))
    """
    try:
        # Parse year from publication_date (format: "YYYY-MM-DD")
        if publication_date:
            year = int(publication_date.split("-")[0])
            current_year = 2024  # Approximate current year
            age = max(0, current_year - year)
        else:
            age = 0
    except:
        age = 0
    
    log_citations = math.log(1 + citation_count)
    denominator = log_citations + lambda_param * age
    
    if denominator == 0:
        return 0.0
    
    return log_citations / denominator


# ============================================================================
# Agent Implementations
# ============================================================================

class Agent2_Evaluator:
    """
    Computes similarity scores between papers and seed embedding
    """
    def __init__(self, seed_embedding: np.ndarray, all_papers: Dict[str, Paper], citation_graph: Dict, alpha: float = 0.65):
        self.seed_embedding = seed_embedding
        self.all_papers = all_papers
        self.citation_graph = citation_graph
        self.alpha = alpha
    
    def _aggregate_citation_embeddings(self, paper_ids: List[str], depth: int = 1) -> np.ndarray:
        """
        Aggregate embeddings from citation papers with depth weighting
        depth=1: direct citations (weight = alpha^1)
        depth=2: citations of citations (weight = alpha^2)
        Returns weighted mean of embeddings
        """
        if not paper_ids:
            return np.zeros_like(self.seed_embedding)
        
        # Weight based on depth using alpha decay
        weight = self.alpha ** depth
        
        embeddings = []
        for pid in paper_ids:
            if pid in self.all_papers:
                embeddings.append(self.all_papers[pid].vector)
        
        if not embeddings:
            return np.zeros_like(self.seed_embedding)
        
        # Mean pooling with depth weight applied
        mean_embedding = np.mean(embeddings, axis=0)
        return mean_embedding * weight
    
    def _get_two_level_citations(self, paper_id: str, direction: str) -> Tuple[List[str], List[str]]:
        """
        Get both level-1 and level-2 citations in the specified direction
        Returns: (level_1_ids, level_2_ids)
        """
        # Get level-1 citations
        level_1_ids = self.citation_graph[direction].get(paper_id, [])
        
        # Get level-2 citations (citations of citations)
        level_2_ids = []
        for l1_id in level_1_ids:
            l2_citations = self.citation_graph[direction].get(l1_id, [])
            level_2_ids.extend(l2_citations)
        
        # Remove duplicates from level-2
        level_2_ids = list(set(level_2_ids))
        
        return level_1_ids, level_2_ids
    
    def compute_similarity_score(self, paper: Paper) -> float:
        """
        Compute similarity component of composite score
        S_similarity = 0.35*abstract + 0.20*title + 0.25*fwd_cites + 0.20*bwd_cites
        
        For citations, includes 2 levels with alpha decay:
        - Level 1: direct citations (weight = alpha)
        - Level 2: citations of citations (weight = alpha^2)
        """
        # Abstract similarity (0.35 weight)
        abstract_sim = cosine_similarity(paper.vector, self.seed_embedding)
        
        # Title similarity (0.20 weight)
        # Note: Currently using same vector as abstract since we don't have separate title embeddings
        title_sim = abstract_sim
        
        # Forward citations (0.25 weight) - papers this paper references
        fwd_l1_ids, fwd_l2_ids = self._get_two_level_citations(paper.id, "forward")
        
        fwd_l1_emb = self._aggregate_citation_embeddings(fwd_l1_ids, depth=1)
        fwd_l2_emb = self._aggregate_citation_embeddings(fwd_l2_ids, depth=2)
        
        # Combine level-1 and level-2 with normalized weights
        if np.any(fwd_l1_emb) or np.any(fwd_l2_emb):
            # Normalize by total weight to get proper mean
            total_fwd_weight = self.alpha + (self.alpha ** 2)
            fwd_combined = (fwd_l1_emb + fwd_l2_emb) / total_fwd_weight
            fwd_sim = cosine_similarity(fwd_combined, self.seed_embedding)
        else:
            fwd_sim = 0.0
        
        # Backward citations (0.20 weight) - papers that cite this paper
        bwd_l1_ids, bwd_l2_ids = self._get_two_level_citations(paper.id, "backward")
        
        bwd_l1_emb = self._aggregate_citation_embeddings(bwd_l1_ids, depth=1)
        bwd_l2_emb = self._aggregate_citation_embeddings(bwd_l2_ids, depth=2)
        
        # Combine level-1 and level-2 with normalized weights
        if np.any(bwd_l1_emb) or np.any(bwd_l2_emb):
            total_bwd_weight = self.alpha + (self.alpha ** 2)
            bwd_combined = (bwd_l1_emb + bwd_l2_emb) / total_bwd_weight
            bwd_sim = cosine_similarity(bwd_combined, self.seed_embedding)
        else:
            bwd_sim = 0.0
        
        # Composite similarity score with weights from algorithm.md
        similarity_score = (
            0.35 * abstract_sim +
            0.20 * title_sim +
            0.25 * fwd_sim +
            0.20 * bwd_sim
        )
        
        return similarity_score


class Agent3_QualityAssessor:
    """
    Assesses paper quality based on venue, authors, institutions, citations
    Uses Claude API research for accurate quality scoring
    """
    def __init__(self, venue_scores: Dict[str, float] = None, 
                 institution_scores: Dict[str, float] = None, 
                 author_scores: Dict[str, float] = None):
        """
        Initialize with pre-researched quality scores
        If not provided, will use simplified citation-based scoring
        """
        self.venue_scores = venue_scores or {}
        self.institution_scores = institution_scores or {}
        self.author_scores = author_scores or {}
        
        # Weights from algorithm.md
        self.w_venue = 0.30
        self.w_authors = 0.25
        self.w_institutions = 0.25
        self.w_citations = 0.20
    
    def _get_venue_quality(self, venue: str) -> float:
        """Get normalized venue quality score"""
        if not venue or venue == "Unknown":
            return 0.5
        
        # Check if we have researched data
        if venue in self.venue_scores:
            return self.venue_scores[venue]
        
        # Fallback: return neutral score
        return 0.5
    
    def _get_author_quality(self, authors: List[str]) -> float:
        """Get average author quality score with square root normalization"""
        if not authors:
            return 0.5
        
        scores = []
        for author in authors:
            if author in self.author_scores:
                scores.append(self.author_scores[author])
            else:
                scores.append(0.5)  # Neutral score for unknown authors
        
        # Average and apply square root normalization (as per algorithm.md)
        avg_score = sum(scores) / len(scores)
        return avg_score ** 0.5
    
    def _get_institution_quality(self, authors_list: List[str]) -> float:
        """
        Get institution quality score
        Note: Paper.authors is a list of author names, institutions would need to be extracted
        For now, returning neutral score as institution data isn't in Paper object
        """
        # TODO: Extract institution data from paper metadata
        return 0.5
    
    def compute_quality_score(self, paper: Paper) -> float:
        """
        Compute composite quality score:
        If we have research data: Q_quality(p) = 0.30*Q_venue + 0.25*Q_authors + 0.25*Q_institutions + 0.20*Q_citations
        If no research data: Just use citation quality
        """
        # Check if we have any research data
        has_research_data = bool(self.venue_scores or self.author_scores or self.institution_scores)
        
        if has_research_data:
            venue_quality = self._get_venue_quality(paper.venue)
            author_quality = self._get_author_quality(paper.authors)
            institution_quality = self._get_institution_quality(paper.authors)
            citation_quality = compute_age_normalized_citation_score(
                paper.citation_count,
                paper.publication_date
            )
            
            # Composite score with weights
            quality_score = (
                self.w_venue * venue_quality +
                self.w_authors * author_quality +
                self.w_institutions * institution_quality +
                self.w_citations * citation_quality
            )
        else:
            # Simplified: just use citation quality
            quality_score = compute_age_normalized_citation_score(
                paper.citation_count,
                paper.publication_date
            )
        
        return quality_score


class Agent4_Librarian:
    """
    Manages candidate pool with lazy evaluation and duplication boost
    """
    def __init__(self, beta: float = 0.15):
        self.candidate_pool: Dict[str, CandidateInfo] = {}
        self.beta = beta  # Boost weight parameter
    
    def add_candidate(self, paper_id: str, base_relevance: float, composite_score: float, citing_paper_id: str):
        """Add or update a candidate in the pool"""
        if paper_id in self.candidate_pool:
            # Update existing candidate
            candidate = self.candidate_pool[paper_id]
            candidate.citing_papers.append(citing_paper_id)
            candidate.count += 1
        else:
            # Create new candidate
            candidate = CandidateInfo(paper_id, base_relevance, composite_score)
            candidate.citing_papers.append(citing_paper_id)
            candidate.count = 1
            self.candidate_pool[paper_id] = candidate
    
    def compute_duplication_boost(self, candidate: CandidateInfo, explored_papers: Dict[str, Paper]) -> float:
        """
        Compute duplication boost: B(p) = log2(1 + sum(R(c) for c in citers))
        """
        total_relevance = sum(
            explored_papers[citer_id].relevance_score 
            for citer_id in candidate.citing_papers 
            if citer_id in explored_papers
        )
        return math.log2(1 + total_relevance)
    
    def apply_duplication_boosts(self, explored_papers: Dict[str, Paper], threshold: float = 0.5):
        """
        Apply duplication boost to all candidates that appear multiple times
        R_boosted = R_base * (1 + β * boost)
        """
        for candidate in self.candidate_pool.values():
            if candidate.count > 1:
                boost = self.compute_duplication_boost(candidate, explored_papers)
                candidate.boosted_relevance = candidate.base_relevance * (1 + self.beta * boost)
    
    def get_top_candidates(self, k: int, explored_ids: Set[str]) -> List[CandidateInfo]:
        """Get top-k candidates by boosted relevance, excluding already explored"""
        available = [c for c in self.candidate_pool.values() if c.paper_id not in explored_ids]
        return sorted(available, key=lambda c: c.boosted_relevance, reverse=True)[:k]


# ============================================================================
# Main Algorithm
# ============================================================================

class BidirectionalPrioritySearch:
    """
    Implements the bidirectional priority search algorithm from algorithm.md
    """
    def __init__(
        self,
        seed_paper: Paper,
        all_papers: Dict[str, Paper],
        venue_scores: Dict[str, float] = None,
        institution_scores: Dict[str, float] = None,
        author_scores: Dict[str, float] = None,
        alpha: float = 0.65,
        beta: float = 0.15,
        epsilon: float = 0.02,
        omega_sim: float = 0.6,
        omega_qual: float = 0.4,
        max_papers: int = 500,
        top_k: int = 5,
        boost_check_interval: int = 10
    ):
        self.seed_paper = seed_paper
        self.all_papers = all_papers
        
        # Algorithm parameters
        self.alpha = alpha  # Decay factor
        self.beta = beta  # Boost weight
        self.epsilon = epsilon  # Relevance threshold
        self.omega_sim = omega_sim  # Similarity weight
        self.omega_qual = omega_qual  # Quality weight
        self.max_papers = max_papers  # Budget
        self.top_k = top_k  # Number of children to explore per iteration
        self.boost_check_interval = boost_check_interval
        
        # Citation graph (extracted from Pinecone metadata if available)
        self.citation_graph = self._build_citation_graph()
        
        # Data structures
        self.q_forward = []  # Priority queue for forward search (references)
        self.q_backward = []  # Priority queue for backward search (citing papers)
        self.explored: Dict[str, Paper] = {}
        self.librarian = Agent4_Librarian(beta=beta)
        
        # Agents with quality research data
        self.evaluator = Agent2_Evaluator(
            seed_embedding=seed_paper.vector,
            all_papers=all_papers,
            citation_graph=self.citation_graph,
            alpha=alpha
        )
        self.quality_assessor = Agent3_QualityAssessor(
            venue_scores=venue_scores,
            institution_scores=institution_scores,
            author_scores=author_scores
        )
    
    def _build_citation_graph(self) -> Dict[str, Dict[str, List[str]]]:
        """
        Build citation graph from paper metadata stored in Pinecone
        Returns: {"forward": {paper_id: [referenced_paper_ids]}, "backward": {paper_id: [citing_paper_ids]}}
        """
        graph = {"forward": defaultdict(list), "backward": defaultdict(list)}
        
        # Build graph from all papers including seed
        all_papers_with_seed = {**self.all_papers, self.seed_paper.id: self.seed_paper}
        
        for paper_id, paper in all_papers_with_seed.items():
            # Forward citations (papers this paper references)
            if hasattr(paper, 'reference_paper_ids'):
                graph["forward"][paper_id] = getattr(paper, 'reference_paper_ids', [])
            
            # Backward citations (papers that cite this paper)
            if hasattr(paper, 'citing_paper_ids'):
                graph["backward"][paper_id] = getattr(paper, 'citing_paper_ids', [])
        
        return graph
    
    def _get_citations(self, paper_id: str, direction: str) -> List[str]:
        """
        Get citations for a paper in the specified direction
        direction: "forward" (references) or "backward" (citing papers)
        """
        if direction == "forward":
            return self.citation_graph["forward"].get(paper_id, [])
        else:
            return self.citation_graph["backward"].get(paper_id, [])
    
    def _compute_composite_score(self, paper: Paper) -> float:
        """
        Compute composite score: S(p) = ω_sim * S_similarity + ω_qual * S_quality
        """
        sim_score = self.evaluator.compute_similarity_score(paper)
        qual_score = self.quality_assessor.compute_quality_score(paper)
        
        return self.omega_sim * sim_score + self.omega_qual * qual_score
    
    def _propagate_relevance(self, parent_relevance: float, composite_score: float, depth_modifier: int) -> float:
        """
        Compute child relevance: R_child = R_parent × α^depth × S_composite
        depth_modifier: +1 for forward, -1 for backward (stored as absolute value)
        """
        return parent_relevance * (self.alpha ** abs(depth_modifier)) * composite_score
    
    def run(self) -> List[Paper]:
        """
        Execute the bidirectional priority search algorithm
        Returns: List of papers sorted by relevance score
        """
        # Initialize: seed paper has relevance = 1.0 at depth 0
        self.seed_paper.relevance_score = 1.0
        self.seed_paper.depth = 0
        self.explored[self.seed_paper.id] = self.seed_paper
        
        # Push seed to forward queue
        heapq.heappush(self.q_forward, PriorityQueueItem(
            self.seed_paper.id, 1.0, 0, "forward"
        ))
        
        iteration = 0
        
        while (self.q_forward or self.q_backward) and len(self.explored) < self.max_papers:
            iteration += 1
            
            # Select highest relevance paper from either queue
            current_item = None
            
            if self.q_forward and self.q_backward:
                if self.q_forward[0].relevance >= self.q_backward[0].relevance:
                    current_item = heapq.heappop(self.q_forward)
                else:
                    current_item = heapq.heappop(self.q_backward)
            elif self.q_forward:
                current_item = heapq.heappop(self.q_forward)
            elif self.q_backward:
                current_item = heapq.heappop(self.q_backward)
            
            if not current_item or current_item.relevance < self.epsilon:
                continue
            
            # Skip if already explored
            if current_item.paper_id in self.explored and iteration > 1:
                continue
            
            current_paper = self.all_papers.get(current_item.paper_id)
            if not current_paper:
                continue
            
            # Mark as explored
            current_paper.relevance_score = current_item.relevance
            current_paper.depth = current_item.depth
            current_paper.direction = current_item.direction
            
            if current_item.paper_id not in self.explored:
                self.explored[current_item.paper_id] = current_paper
            
            # Get citations in current direction
            citations = self._get_citations(current_item.paper_id, current_item.direction)
            
            # Score all citations and add to candidate pool
            depth_modifier = 1 if current_item.direction == "forward" else -1
            
            for cited_paper_id in citations:
                if cited_paper_id not in self.all_papers:
                    continue
                
                cited_paper = self.all_papers[cited_paper_id]
                
                # Compute composite score
                composite_score = self._compute_composite_score(cited_paper)
                
                # Compute base relevance
                base_relevance = self._propagate_relevance(
                    current_item.relevance,
                    composite_score,
                    depth_modifier
                )
                
                # Add to candidate pool
                self.librarian.add_candidate(
                    cited_paper_id,
                    base_relevance,
                    composite_score,
                    current_item.paper_id
                )
            
            # Select top-k candidates for immediate exploration
            top_candidates = self.librarian.get_top_candidates(self.top_k, set(self.explored.keys()))
            
            for candidate in top_candidates[:self.top_k]:
                if candidate.boosted_relevance > self.epsilon:
                    new_depth = abs(current_item.depth) + 1
                    
                    # Add to appropriate queue
                    if current_item.direction == "forward":
                        heapq.heappush(self.q_forward, PriorityQueueItem(
                            candidate.paper_id,
                            candidate.boosted_relevance,
                            new_depth,
                            "forward"
                        ))
                    else:
                        heapq.heappush(self.q_backward, PriorityQueueItem(
                            candidate.paper_id,
                            candidate.boosted_relevance,
                            new_depth,
                            "backward"
                        ))
            
            # Periodic duplication boost check
            if iteration % self.boost_check_interval == 0:
                self.librarian.apply_duplication_boosts(self.explored)
                
                # Resurrect high-relevance boosted papers
                resurrected = self.librarian.get_top_candidates(20, set(self.explored.keys()))
                for candidate in resurrected:
                    if candidate.boosted_relevance > self.epsilon and candidate.count > 1:
                        # Infer depth and direction from citing papers
                        avg_depth = sum(
                            self.explored[citer].depth 
                            for citer in candidate.citing_papers 
                            if citer in self.explored
                        ) / len(candidate.citing_papers)
                        
                        heapq.heappush(self.q_forward, PriorityQueueItem(
                            candidate.paper_id,
                            candidate.boosted_relevance,
                            int(avg_depth) + 1,
                            "forward"
                        ))
        
        # Return high-relevance papers sorted by relevance
        high_relevance = [p for p in self.explored.values() if p.relevance_score > 0.1]
        return sorted(high_relevance, key=lambda p: p.relevance_score, reverse=True)


# ============================================================================
# Pinecone Data Retrieval
# ============================================================================

def get_parent_paper_doi() -> str:
    """Extract DOI from the parent paper PDF"""
    from arxai import extract_doi_from_pdf
    
    pdf_path = os.getenv("PDF_PATH")
    if not pdf_path:
        raise ValueError("PDF_PATH not found in .env.local")
    
    if not os.path.exists(pdf_path):
        raise ValueError(f"PDF file not found: {pdf_path}")
    
    doi = extract_doi_from_pdf(pdf_path)
    if not doi:
        raise ValueError(f"DOI not found in PDF: {pdf_path}")
    
    return doi


def fetch_all_papers_from_pinecone(parent_doi: str) -> Tuple[Paper, Dict[str, Paper]]:
    """
    Fetch all papers from Pinecone, separating parent and children
    Returns: (parent_paper, dict of child papers)
    """
    # Sanitize parent DOI for Pinecone lookup
    parent_id = sanitize_doi_for_pinecone(parent_doi)
    
    # Fetch all vectors from Pinecone
    # Note: Pinecone query with top_k=10000 to get as many as possible
    # For production, use pagination with describe_index_stats and fetch in batches
    
    try:
        # Query for vectors similar to a zero vector to get all papers
        # (This is a workaround; ideally use list/scan operations)
        dummy_vector = [0.0] * 768
        results = index.query(
            vector=dummy_vector,
            top_k=10000,
            include_metadata=True,
            include_values=True
        )
        
        papers = {}
        parent_paper = None
        
        for match in results.matches:
            paper = Paper(
                paper_id=match.id,
                metadata=match.metadata,
                vector=match.values
            )
            
            if match.id == parent_id:
                parent_paper = paper
            else:
                papers[match.id] = paper
        
        if not parent_paper:
            raise ValueError(f"Parent paper not found in Pinecone: {parent_doi}")
        
        return parent_paper, papers
    
    except Exception as e:
        raise ValueError(f"Error fetching papers from Pinecone: {str(e)}")


# ============================================================================
# CLI Display Functions
# ============================================================================

def print_separator():
    print("\n" + "="*80 + "\n")


def print_header():
    print("\n" + "="*80)
    print("🔬 Bidirectional Priority Search Algorithm")
    print("="*80 + "\n")


def print_paper_hierarchy(papers: List[Paper], seed_paper: Paper):
    """Display the paper hierarchy sorted by relevance"""
    print("📊 PAPER HIERARCHY (Sorted by Relevance Score)")
    print_separator()
    
    print(f"🌱 SEED PAPER (Relevance: 1.000)")
    print(f"   Title: {seed_paper.title}")
    print(f"   DOI: {seed_paper.doi}")
    print(f"   Citations: {seed_paper.citation_count}")
    print_separator()
    
    print(f"📚 DISCOVERED PAPERS (Top {len(papers)})")
    print_separator()
    
    for i, paper in enumerate(papers[:50], 1):  # Show top 50
        direction_icon = "→" if paper.direction == "forward" else "←"
        print(f"{i}. {direction_icon} Relevance: {paper.relevance_score:.4f} | Depth: {paper.depth}")
        print(f"   Title: {paper.title}")
        print(f"   Venue: {paper.venue}")
        print(f"   Citations: {paper.citation_count}")
        print(f"   Date: {paper.publication_date}")
        print()
    
    if len(papers) > 50:
        print(f"... and {len(papers) - 50} more papers")


def print_statistics(papers: List[Paper], explored_count: int):
    """Print algorithm statistics"""
    print("📈 ALGORITHM STATISTICS")
    print_separator()
    
    print(f"Total Papers Explored: {explored_count}")
    print(f"High-Relevance Papers (R > 0.1): {len(papers)}")
    
    if papers:
        avg_relevance = sum(p.relevance_score for p in papers) / len(papers)
        max_depth = max(p.depth for p in papers)
        
        forward_count = sum(1 for p in papers if p.direction == "forward")
        backward_count = sum(1 for p in papers if p.direction == "backward")
        
        print(f"Average Relevance Score: {avg_relevance:.4f}")
        print(f"Maximum Depth Reached: {max_depth}")
        print(f"Forward Search (References): {forward_count} papers")
        print(f"Backward Search (Citations): {backward_count} papers")


# ============================================================================
# Main Execution
# ============================================================================

async def perform_quality_research(papers: Dict[str, Paper], parent_paper: Paper) -> Tuple[Dict, Dict, Dict]:
    """
    Extract unique venues, institutions, and authors, then research them in parallel
    Returns: (venue_scores, institution_scores, author_scores)
    """
    print("\n⏳ Step 3: Performing quality research with Claude API...")
    
    # Try loading from cache first
    cached_venues, cached_institutions, cached_authors = load_research_cache()
    
    # Extract unique venues, institutions, and authors from all papers
    all_papers_with_parent = {**papers, parent_paper.id: parent_paper}
    
    venues = set()
    institutions = set()
    authors = set()
    
    for paper in all_papers_with_parent.values():
        if paper.venue and paper.venue != "Unknown":
            venues.add(paper.venue)
        
        if paper.authors:
            for author in paper.authors:
                if author and author.strip():
                    authors.add(author)
    
    # Filter out already cached items
    venues_to_research = [v for v in venues if v not in cached_venues]
    authors_to_research = [a for a in authors if a not in cached_authors]
    
    print(f"   Found {len(venues)} unique venues ({len(venues_to_research)} new)")
    print(f"   Found {len(authors)} unique authors ({len(authors_to_research)} new)")
    
    # Only research if there are new items
    if venues_to_research or authors_to_research:
        print(f"   🚀 Researching {len(venues_to_research)} venues and {len(authors_to_research)} authors...")
        
        new_venues, new_institutions, new_authors = await research_all_parallel(
            venues_to_research,
            [],  # No institution research for now
            authors_to_research
        )
        
        # Merge with cache
        all_venues = {**cached_venues, **new_venues}
        all_institutions = {**cached_institutions, **new_institutions}
        all_authors = {**cached_authors, **new_authors}
        
        # Save updated cache
        save_research_cache(all_venues, all_institutions, all_authors)
    else:
        print("   ✅ All quality data found in cache!")
        all_venues = cached_venues
        all_institutions = cached_institutions
        all_authors = cached_authors
    
    # Convert to score dictionaries
    venue_scores = {v: data.get("quality_score", 0.5) for v, data in all_venues.items()}
    institution_scores = {i: data.get("quality_score", 0.5) for i, data in all_institutions.items()}
    author_scores = {a: data.get("quality_score", 0.5) for a, data in all_authors.items()}
    
    print(f"✅ Quality research complete! Scored {len(venue_scores)} venues, {len(author_scores)} authors")
    
    return venue_scores, institution_scores, author_scores


def main():
    """Main execution function"""
    print_header()
    
    try:
        # Step 1: Get parent paper DOI
        print("⏳ Step 1: Extracting parent paper DOI from PDF...")
        parent_doi = get_parent_paper_doi()
        print(f"✅ Parent DOI: {parent_doi}")
        
        # Step 2: Fetch all papers from Pinecone
        print("\n⏳ Step 2: Fetching all papers from Pinecone database...")
        parent_paper, child_papers = fetch_all_papers_from_pinecone(parent_doi)
        print(f"✅ Found {len(child_papers)} child papers")
        
        # Step 3: Perform quality research with Claude API (parallelized)
        venue_scores, institution_scores, author_scores = asyncio.run(
            perform_quality_research(child_papers, parent_paper)
        )
        
        # Step 4: Run bidirectional priority search with quality scores
        print("\n⏳ Step 4: Running bidirectional priority search algorithm...")
        print(f"   Parameters: α={0.65}, β={0.15}, ε={0.02}")
        print(f"   Quality scoring enabled with {len(venue_scores)} venue scores, {len(author_scores)} author scores")
        
        search = BidirectionalPrioritySearch(
            seed_paper=parent_paper,
            all_papers=child_papers,
            venue_scores=venue_scores,
            institution_scores=institution_scores,
            author_scores=author_scores,
            alpha=0.65,
            beta=0.15,
            epsilon=0.02,
            omega_sim=0.6,
            omega_qual=0.4,
            max_papers=500,
            top_k=5,
            boost_check_interval=10
        )
        
        results = search.run()
        print(f"✅ Algorithm complete! Found {len(results)} high-relevance papers")
        
        # Step 5: Display results
        print_separator()
        print_paper_hierarchy(results, parent_paper)
        print_separator()
        print_statistics(results, len(search.explored))
        print_separator()
        
        print("✅ All done!")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

