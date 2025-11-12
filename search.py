import math
import heapq
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
from api import get_paper_metadata, get_citing_papers, get_reference_papers, cosine_similarity, search_papers

class Paper:
    def __init__(self, paper_id: str, metadata: Dict):
        self.id = paper_id
        self.doi = metadata.get("doi", paper_id)
        self.title = metadata.get("title", "Unknown")
        self.authors = metadata.get("authors", [])
        self.venue = metadata.get("venue", "Unknown")
        self.citation_count = metadata.get("citation_count", 0)
        self.publication_date = metadata.get("publication_date", "")
        self.fields_of_study = metadata.get("fields_of_study", [])
        self.vector = np.array(metadata.get("embedding", []))
        self.abstract = metadata.get("abstract", "")
        self.relevance_score = 0.0
        self.depth = 0
        self.direction = None
        self.parent_id = None

class PriorityQueueItem:
    def __init__(self, paper_id: str, relevance: float, depth: int, direction: str):
        self.paper_id = paper_id
        self.relevance = relevance
        self.depth = depth
        self.direction = direction
    
    def __lt__(self, other):
        return self.relevance > other.relevance

class CandidateInfo:
    def __init__(self, paper_id: str, base_relevance: float, composite_score: float):
        self.paper_id = paper_id
        self.base_relevance = base_relevance
        self.composite_score = composite_score
        self.citing_papers = []
        self.count = 0
        self.boosted_relevance = base_relevance

def compute_quality_score(citation_count: int, publication_date: str) -> float:
    try:
        if publication_date:
            year = int(publication_date.split("-")[0])
            age = max(0, 2024 - year)
        else:
            age = 0
    except:
        age = 0
    log_citations = math.log(1 + citation_count)
    denominator = log_citations + 0.5 * age
    return log_citations / denominator if denominator > 0 else 0.0

def aggregate_citation_embeddings(paper_ids: List[str], all_papers: Dict[str, Paper], depth: int, alpha: float) -> np.ndarray:
    if not paper_ids:
        return np.array([])
    
    weight = alpha ** depth
    embeddings = []
    for pid in paper_ids:
        if pid in all_papers and len(all_papers[pid].vector) > 0:
            embeddings.append(all_papers[pid].vector)
    
    if not embeddings:
        return np.array([])
    
    mean_embedding = np.mean(embeddings, axis=0)
    return mean_embedding * weight

def get_two_level_citations(paper_id: str, citation_graph: Dict, direction: str) -> Tuple[List[str], List[str]]:
    level_1_ids = citation_graph[direction].get(paper_id, [])
    level_2_ids = []
    for l1_id in level_1_ids:
        l2_citations = citation_graph[direction].get(l1_id, [])
        level_2_ids.extend(l2_citations)
    level_2_ids = list(set(level_2_ids))
    return level_1_ids, level_2_ids

def compute_similarity_score(paper: Paper, seed_vector: np.ndarray, all_papers: Dict[str, Paper], citation_graph: Dict, alpha: float = 0.65) -> float:
    abstract_sim = cosine_similarity(paper.vector, seed_vector)
    title_sim = abstract_sim
    
    fwd_l1_ids, fwd_l2_ids = get_two_level_citations(paper.id, citation_graph, "forward")
    fwd_l1_emb = aggregate_citation_embeddings(fwd_l1_ids, all_papers, depth=1, alpha=alpha)
    fwd_l2_emb = aggregate_citation_embeddings(fwd_l2_ids, all_papers, depth=2, alpha=alpha)
    
    if len(fwd_l1_emb) > 0 or len(fwd_l2_emb) > 0:
        total_fwd_weight = alpha + (alpha ** 2)
        if len(fwd_l1_emb) > 0 and len(fwd_l2_emb) > 0:
            fwd_combined = (fwd_l1_emb + fwd_l2_emb) / total_fwd_weight
        elif len(fwd_l1_emb) > 0:
            fwd_combined = fwd_l1_emb / alpha
        else:
            fwd_combined = fwd_l2_emb / (alpha ** 2)
        fwd_sim = cosine_similarity(fwd_combined, seed_vector)
    else:
        fwd_sim = 0.0
    
    bwd_l1_ids, bwd_l2_ids = get_two_level_citations(paper.id, citation_graph, "backward")
    bwd_l1_emb = aggregate_citation_embeddings(bwd_l1_ids, all_papers, depth=1, alpha=alpha)
    bwd_l2_emb = aggregate_citation_embeddings(bwd_l2_ids, all_papers, depth=2, alpha=alpha)
    
    if len(bwd_l1_emb) > 0 or len(bwd_l2_emb) > 0:
        total_bwd_weight = alpha + (alpha ** 2)
        if len(bwd_l1_emb) > 0 and len(bwd_l2_emb) > 0:
            bwd_combined = (bwd_l1_emb + bwd_l2_emb) / total_bwd_weight
        elif len(bwd_l1_emb) > 0:
            bwd_combined = bwd_l1_emb / alpha
        else:
            bwd_combined = bwd_l2_emb / (alpha ** 2)
        bwd_sim = cosine_similarity(bwd_combined, seed_vector)
    else:
        bwd_sim = 0.0
    
    return 0.35 * abstract_sim + 0.20 * title_sim + 0.25 * fwd_sim + 0.20 * bwd_sim

def build_citation_graph(papers: Dict[str, Paper]) -> Dict:
    graph = {"forward": defaultdict(list), "backward": defaultdict(list)}
    for paper_id, paper in papers.items():
        try:
            refs = get_reference_papers(paper_id)
            cites = get_citing_papers(paper_id)
            graph["forward"][paper_id] = refs
            graph["backward"][paper_id] = cites
        except:
            pass
    return graph

def compute_duplication_boost(candidate: CandidateInfo, explored_papers: Dict[str, Paper]) -> float:
    total_relevance = sum(
        explored_papers[citer_id].relevance_score 
        for citer_id in candidate.citing_papers 
        if citer_id in explored_papers
    )
    return math.log2(1 + total_relevance)

class BidirectionalPrioritySearch:
    def __init__(
        self,
        seed_paper: Paper,
        all_papers: Dict[str, Paper],
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
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.omega_sim = omega_sim
        self.omega_qual = omega_qual
        self.max_papers = max_papers
        self.top_k = top_k
        self.boost_check_interval = boost_check_interval
        
        self.citation_graph = build_citation_graph(all_papers)
        self.q_forward = []
        self.q_backward = []
        self.explored: Dict[str, Paper] = {}
        self.candidate_pool: Dict[str, CandidateInfo] = {}
    
    def _compute_composite_score(self, paper: Paper) -> float:
        sim_score = compute_similarity_score(paper, self.seed_paper.vector, self.all_papers, self.citation_graph, self.alpha)
        qual_score = compute_quality_score(paper.citation_count, paper.publication_date)
        return self.omega_sim * sim_score + self.omega_qual * qual_score
    
    def _propagate_relevance(self, parent_relevance: float, composite_score: float, depth_modifier: int, boost: float = 0.0) -> float:
        return parent_relevance * (self.alpha ** abs(depth_modifier)) * (1 + self.beta * boost) * composite_score
    
    def _get_citations(self, paper_id: str, direction: str) -> List[str]:
        if direction == "forward":
            return self.citation_graph["forward"].get(paper_id, [])
        else:
            return self.citation_graph["backward"].get(paper_id, [])
    
    def run(self) -> List[Paper]:
        self.seed_paper.relevance_score = 1.0
        self.seed_paper.depth = 0
        self.explored[self.seed_paper.id] = self.seed_paper
        
        if not self.citation_graph["forward"].get(self.seed_paper.id) and not self.citation_graph["backward"].get(self.seed_paper.id):
            refs = get_reference_papers(self.seed_paper.id)
            cites = get_citing_papers(self.seed_paper.id)
            self.citation_graph["forward"][self.seed_paper.id] = refs
            self.citation_graph["backward"][self.seed_paper.id] = cites
        
        # Initialize bidirectional search: push seed to both queues
        heapq.heappush(self.q_forward, PriorityQueueItem(self.seed_paper.id, 1.0, 0, "forward"))
        heapq.heappush(self.q_backward, PriorityQueueItem(self.seed_paper.id, 1.0, 0, "backward"))
        
        print(f"Starting bidirectional search from seed: {self.seed_paper.title[:60]}")
        
        # Check initial citation counts
        forward_count = len(self.citation_graph["forward"].get(self.seed_paper.id, []))
        backward_count = len(self.citation_graph["backward"].get(self.seed_paper.id, []))
        print(f"Found {forward_count} references (forward) and {backward_count} citations (backward)")
        print("Exploring papers...\n", flush=True)
        
        iteration = 0
        
        while (self.q_forward or self.q_backward) and len(self.explored) < self.max_papers:
            iteration += 1
            
            # Periodic progress update
            if iteration % 50 == 0:
                print(f"\n  Progress: {len(self.explored)} papers explored, {len(self.candidate_pool)} candidates queued\n")
            
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
            
            if not current_item:
                break
            
            if current_item.relevance < self.epsilon:
                continue
            
            # Allow seed paper to be processed from both directions for true bidirectional search
            # Skip other papers that are already explored (unless it's the seed paper)
            if current_item.paper_id in self.explored and iteration > 1 and current_item.paper_id != self.seed_paper.id:
                continue
            
            current_paper = self.all_papers.get(current_item.paper_id)
            if not current_paper:
                continue
            
            current_paper.relevance_score = current_item.relevance
            current_paper.depth = abs(current_item.depth)
            current_paper.direction = current_item.direction
            
            if current_item.paper_id not in self.explored:
                self.explored[current_item.paper_id] = current_paper
                # Stream output: show paper being explored
                if current_item.paper_id != self.seed_paper.id:
                    title_short = current_paper.title[:70] if len(current_paper.title) > 70 else current_paper.title
                    print(f"  [{current_item.direction:8s}] Score: {current_item.relevance:.4f} | {title_short}", flush=True)
            
            citations = self._get_citations(current_item.paper_id, current_item.direction)
            depth_modifier = 1 if current_item.direction == "forward" else -1
            
            if citations:
                # Show that we're processing citations
                if current_item.paper_id == self.seed_paper.id:
                    print(f"  Processing {len(citations)} citations from seed paper ({current_item.direction} direction)...", flush=True)
            
            papers_fetched = 0
            for cited_paper_id in citations:
                if cited_paper_id not in self.all_papers:
                    try:
                        metadata = get_paper_metadata(cited_paper_id)
                        self.all_papers[cited_paper_id] = Paper(cited_paper_id, metadata)
                        refs = get_reference_papers(cited_paper_id)
                        cites = get_citing_papers(cited_paper_id)
                        self.citation_graph["forward"][cited_paper_id] = refs
                        self.citation_graph["backward"][cited_paper_id] = cites
                        papers_fetched += 1
                    except Exception as e:
                        # Silently continue on errors (API failures, etc.)
                        continue

                cited_paper = self.all_papers.get(cited_paper_id)
                if not cited_paper:
                    continue
                
                composite_score = self._compute_composite_score(cited_paper)
                base_relevance = current_item.relevance * (self.alpha ** abs(depth_modifier))
                
                if cited_paper_id in self.candidate_pool:
                    self.candidate_pool[cited_paper_id].citing_papers.append(current_item.paper_id)
                    self.candidate_pool[cited_paper_id].count += 1
                    self.candidate_pool[cited_paper_id].base_relevance = max(
                        self.candidate_pool[cited_paper_id].base_relevance,
                        base_relevance
                    )
                    self.candidate_pool[cited_paper_id].composite_score = composite_score
                else:
                    self.candidate_pool[cited_paper_id] = CandidateInfo(
                        cited_paper_id, base_relevance, composite_score
                    )
                    self.candidate_pool[cited_paper_id].citing_papers.append(current_item.paper_id)
                    self.candidate_pool[cited_paper_id].count = 1
            
            top_candidates = sorted(
                [c for c in self.candidate_pool.values() if c.paper_id not in self.explored],
                key=lambda c: c.base_relevance * c.composite_score,
                reverse=True
            )[:self.top_k]
            
            queued_count = 0
            for candidate in top_candidates:
                boost = compute_duplication_boost(candidate, self.explored) if candidate.count > 1 else 0.0
                child_relevance = self._propagate_relevance(
                    current_item.relevance,
                    candidate.composite_score,
                    depth_modifier,
                    boost
                )
                
                if child_relevance > self.epsilon:
                    new_depth = abs(current_item.depth) + 1
                    if current_item.direction == "forward":
                        heapq.heappush(self.q_forward, PriorityQueueItem(
                            candidate.paper_id, child_relevance, new_depth, "forward"
                        ))
                    else:
                        heapq.heappush(self.q_backward, PriorityQueueItem(
                            candidate.paper_id, child_relevance, new_depth, "backward"
                        ))
                    queued_count += 1
            
            # Show summary after processing seed paper citations
            if current_item.paper_id == self.seed_paper.id and citations:
                print(f"  Fetched {papers_fetched} new papers, queued {queued_count} candidates\n", flush=True)
            
            if iteration % self.boost_check_interval == 0:
                for candidate in self.candidate_pool.values():
                    if candidate.count > 1:
                        boost = compute_duplication_boost(candidate, self.explored)
                        candidate.boosted_relevance = candidate.base_relevance * (1 + self.beta * boost)
                        
                        if candidate.boosted_relevance > self.epsilon and candidate.paper_id not in self.explored:
                            avg_depth = sum(
                                self.explored[citer].depth 
                                for citer in candidate.citing_papers 
                                if citer in self.explored
                            ) / len(candidate.citing_papers) if candidate.citing_papers else 1
                            
                            heapq.heappush(self.q_forward, PriorityQueueItem(
                                candidate.paper_id,
                                candidate.boosted_relevance,
                                int(avg_depth) + 1,
                                "forward"
                            ))
        
        print(f"\n  Search complete: explored {len(self.explored)} papers total", flush=True)
        
        high_relevance = [p for p in self.explored.values() if p.id != self.seed_paper.id and p.relevance_score > self.epsilon]
        if len(high_relevance) < 5:
            high_relevance = [p for p in self.explored.values() if p.id != self.seed_paper.id and p.relevance_score > 0]
        
        if not high_relevance:
            print("  Warning: No papers found above relevance threshold. This might indicate:")
            print("    - API rate limiting or connection issues")
            print("    - Seed paper has no citations/references")
            print("    - Relevance scores are too low")
            print(f"    - Epsilon threshold: {self.epsilon}")
        
        return sorted(high_relevance, key=lambda p: p.relevance_score, reverse=True)[:15]

def search_level_1(seed_paper: Paper, alpha: float = 0.65, epsilon: float = 0.02, max_papers: int = 500) -> Tuple[List[Paper], Dict[str, Paper]]:
    all_papers = {seed_paper.id: seed_paper}
    
    print("Running bidirectional priority search...")
    search = BidirectionalPrioritySearch(
        seed_paper=seed_paper,
        all_papers=all_papers,
        alpha=alpha,
        beta=0.15,
        epsilon=epsilon,
        max_papers=max_papers,
        top_k=5,
        boost_check_interval=10
    )
    
    results = search.run()
    filtered_results = [p for p in results if p.id != seed_paper.id]
    top_15 = filtered_results[:15]
    print(f"Found {len(top_15)} high-relevance papers (top 15)")
    
    return top_15, search.all_papers

def search_level_2(
    seed_paper: Paper,
    selected_papers: List[Paper],
    all_papers: Dict[str, Paper],
    user_query: Optional[str] = None,
    alpha: float = 0.65,
    epsilon: float = 0.02,
    max_papers: int = 500,
    semantic_weight: float = 0.4,
    citation_weight: float = 0.6
) -> List[Paper]:
    explored = {p.id: p for p in selected_papers}
    explored[seed_paper.id] = seed_paper
    
    print("Running second pass with user query refinement...")
    
    seed_vector = seed_paper.vector
    user_query_vector = None
    
    if user_query:
        print(f"Processing user query: {user_query}")
        semantic_results = search_papers(user_query, limit=50)
        if semantic_results:
            query_papers = []
            for result in semantic_results[:10]:
                paper_id = result.get("externalIds", {}).get("DOI") or result.get("paperId")
                if paper_id:
                    try:
                        metadata = get_paper_metadata(paper_id)
                        paper = Paper(paper_id, metadata)
                        all_papers[paper_id] = paper
                        query_papers.append(paper)
                    except:
                        continue
            
            if query_papers:
                query_embeddings = [p.vector for p in query_papers if len(p.vector) > 0]
                if query_embeddings:
                    user_query_vector = np.mean(query_embeddings, axis=0)
                    print(f"Created user query embedding from {len(query_papers)} papers")
    
    citation_graph = build_citation_graph(all_papers)
    
    citation_candidates = set()
    for paper in selected_papers:
        citing = get_citing_papers(paper.id)
        refs = get_reference_papers(paper.id)
        citation_candidates.update(citing + refs)
    
    citation_candidates -= set(explored.keys())
    print(f"Found {len(citation_candidates)} citation-based candidates")
    
    semantic_candidates = []
    if user_query and user_query_vector is not None:
        semantic_results = search_papers(user_query, limit=100)
        for result in semantic_results:
            paper_id = result.get("externalIds", {}).get("DOI") or result.get("paperId")
            if paper_id and paper_id not in explored:
                semantic_candidates.append(paper_id)
        print(f"Found {len(semantic_candidates)} semantic search candidates")
    
    all_candidates = list(citation_candidates) + semantic_candidates
    
    print(f"Processing {len(all_candidates)} total candidates...")
    
    for i, paper_id in enumerate(all_candidates):
        if i >= max_papers:
            break
        if paper_id not in all_papers:
            try:
                metadata = get_paper_metadata(paper_id)
                all_papers[paper_id] = Paper(paper_id, metadata)
            except:
                continue
    
    citation_graph = build_citation_graph(all_papers)
    
    for paper_id, paper in all_papers.items():
        if paper_id in explored:
            continue
        
        citation_sim = compute_similarity_score(paper, seed_vector, all_papers, citation_graph, alpha)
        
        semantic_sim = 0.0
        if user_query_vector is not None and len(paper.vector) > 0:
            semantic_sim = cosine_similarity(paper.vector, user_query_vector)
        
        is_citation_connected = paper_id in citation_candidates
        
        if is_citation_connected:
            combined_sim = citation_weight * citation_sim + semantic_weight * semantic_sim
        else:
            combined_sim = semantic_sim
        
        qual_score = compute_quality_score(paper.citation_count, paper.publication_date)
        composite = 0.6 * combined_sim + 0.4 * qual_score
        
        citation_boost = 1.2 if is_citation_connected else 1.0
        
        paper.relevance_score = (alpha ** 2) * composite * citation_boost
        paper.depth = 2
        
        if paper.relevance_score > epsilon:
            explored[paper_id] = paper
    
    results = sorted(
        [p for p in explored.values() if p.id != seed_paper.id and p.relevance_score > epsilon],
        key=lambda p: p.relevance_score,
        reverse=True
    )
    
    print(f"Found {len(results)} papers in second pass")
    return results
