# Bidirectional Priority Search with Duplication Boost and Real-Time Human Feedback

## Core Mathematical Framework

### Relevance Propagation Function

\[
R_{\text{child}} = R_{\text{parent}} \times \alpha^{d} \times (1 + \beta \cdot B(\text{child}))
\]

Where:
- \( d \) = depth (forward: positive, backward: negative to differentiate)
- \( \alpha \in [0.5, 0.9] \) = decay factor (to be optimized)
- \( B(\text{child}) \) = duplication boost function
- \( \beta \) = boost weight parameter

### Duplication Boost Function (Lazy Evaluation)

\[
B(p) = \log_2(1 + \sum_{c \in \text{citers}(p)} R(c))
\]

Where:
- Papers cited by high-relevance papers get weighted boost
- Logarithmic to prevent survey paper dominance
- Only triggered when \( \sum R(\text{citers}) > \tau_{\text{boost}} \)

### Composite Score Function

\[
S(p) = \omega_{\text{sim}} \cdot S_{\text{similarity}}(p) + \omega_{\text{qual}} \cdot S_{\text{quality}}(p)
\]

**Similarity Component** (compared to seed query + paper + annotations):

\[
S_{\text{similarity}}(p) = 0.35 \cdot \sigma_{\text{abstract}}(p) + 0.20 \cdot \sigma_{\text{title}}(p) + 0.25 \cdot \sigma_{\text{fwd\_cites}}(p) + 0.20 \cdot \sigma_{\text{bwd\_cites}}(p)
\]

Where \( \sigma(p) = \cos(\text{emb}(p), \text{emb}(\text{seed})) \)

**Two-Level Citation Embedding**:

For citation embeddings, we aggregate embeddings across 2 levels with alpha-based decay weighting:

\[
\sigma_{\text{fwd\_cites}}(p) = \cos\left(\frac{\alpha \cdot E_1^{\text{fwd}} + \alpha^2 \cdot E_2^{\text{fwd}}}{\alpha + \alpha^2}, \text{emb}(\text{seed})\right)
\]

\[
\sigma_{\text{bwd\_cites}}(p) = \cos\left(\frac{\alpha \cdot E_1^{\text{bwd}} + \alpha^2 \cdot E_2^{\text{bwd}}}{\alpha + \alpha^2}, \text{emb}(\text{seed})\right)
\]

Where:
- \( E_1^{\text{fwd}} \) = mean embedding of papers referenced by \( p \) (level-1 forward)
- \( E_2^{\text{fwd}} \) = mean embedding of papers referenced by \( p \)'s references (level-2 forward)
- \( E_1^{\text{bwd}} \) = mean embedding of papers citing \( p \) (level-1 backward)
- \( E_2^{\text{bwd}} \) = mean embedding of papers citing \( p \)'s citers (level-2 backward)
- \( \alpha \) = decay factor (same as relevance propagation, typically 0.65)

This captures the broader semantic context by including citations of citations, weighted by distance.

**Quality Component**:

\[
S_{\text{quality}}(p) = 0.30 \cdot Q_{\text{venue}}(p) + 0.25 \cdot Q_{\text{authors}}(p) + 0.25 \cdot Q_{\text{institutions}}(p) + 0.20 \cdot Q_{\text{citations}}(p)
\]

### Citation Count Normalization (Prevents old paper bias)

\[
Q_{\text{citations}}(p) = \frac{\log(1 + c(p))}{\log(1 + c(p)) + \lambda \cdot \text{age}(p)}
\]

Where:
- \( c(p) \) = citation count
- \( \text{age}(p) \) = years since publication
- \( \lambda \approx 0.5 \) = age penalty factor

### Author/Institution Quality Score

\[
Q_{\text{authors}}(p) = \frac{1}{|A(p)|} \sum_{a \in A(p)} h\text{-index}(a)^{0.5}
\]

\[
Q_{\text{institutions}}(p) = \frac{1}{|I(p)|} \sum_{i \in I(p)} \text{rank}(i)^{-0.3}
\]

Normalized to [0,1] via min-max scaling across explored papers.

## Algorithm Structure

### Data Structures

```
PriorityQueue Q_forward    // Max-heap by R(p), forward citations
PriorityQueue Q_backward   // Max-heap by R(p), backward citations
HashMap explored           // paper_id → {metadata, R_score, depth}
HashMap candidate_pool     // paper_id → {scores[], citing_papers[], count}
HashMap human_feedback     // timestamp → {paper_id, annotations, relevance_signal}
Vector seed_embedding      // Updated dynamically with human feedback
```

### Bidirectional Agent Architecture

**Agent 1A - Forward Scout**
- Explores papers cited BY current paper (older literature)
- Tracks forward edges in citation graph

**Agent 1B - Backward Scout**  
- Explores papers that CITE current paper (newer literature)
- Tracks backward edges in citation graph

**Agent 2 - Evaluator**
- Computes embeddings for abstract, title
- Aggregates citation embeddings (mean pooling) across 2 levels with alpha decay
- Level 1: Direct citations (weight = α)
- Level 2: Citations of citations (weight = α²)
- Scores via cosine similarity to dynamic seed

**Agent 3 - Quality Assessor**
- Fetches venue rankings (CORE, impact factor)
- Queries author h-indices (Semantic Scholar API)
- Looks up institution rankings (cached database)
- Normalizes scores to [0,1]

**Agent 4 - Librarian (Lazy Evaluation)**
- Stores ALL scored papers in candidate_pool (not just top-k)
- Increments duplication counters as papers reappear
- Periodically re-scores candidates with boost applied
- Promotes to queue if \( R_{\text{boosted}} > \epsilon \)

**Agent 5 - Annotator**
- Uses extractive summarization on full text
- Highlights sentences with high similarity to seed
- Generates natural language context for user

**Agent 6 - Feedback Integrator**
- Listens to human comment stream
- Updates seed_embedding = \( 0.7 \cdot \text{seed} + 0.3 \cdot \text{feedback\_emb} \)
- Triggers re-evaluation of candidate_pool
- Adjusts \( \alpha, \beta \) based on user signals

### Main Loop (Pseudocode)

```
Initialize:
    seed_embedding ← embed(seed_query + seed_paper + annotations)
    Q_forward.push((seed_paper, R=1.0, depth=0))
    papers_streamed ← 0
    
While (Q_forward not empty OR Q_backward not empty) AND budget < N:
    
    // Interleaved bidirectional exploration
    current ← max(Q_forward.peek(), Q_backward.peek())  // Higher R wins
    direction ← "forward" if current from Q_forward else "backward"
    
    If papers_streamed < 3:
        stream_to_user(current, with_annotations=True)
        papers_streamed++
    
    // Fetch citations (bidirectional)
    If direction == "forward":
        citations ← API.get_references(current.id)
        depth_modifier ← 1
    Else:
        citations ← API.get_citing_papers(current.id)
        depth_modifier ← -1
    
    // Score ALL citations
    For each paper p in citations:
        
        // Check duplication
        If p in candidate_pool:
            candidate_pool[p].citing_papers.append(current.id)
            candidate_pool[p].count++
        
        // Fetch abstract (not full PDF yet)
        abstract ← API.get_abstract(p)
        
        // Compute similarity
        S_sim ← similarity_score(p, seed_embedding)
        S_qual ← quality_score(p)
        S_total ← 0.6 × S_sim + 0.4 × S_qual
        
        // Store in candidate pool
        candidate_pool[p] ← {
            score: S_total,
            citing_papers: [current.id],
            count: 1,
            R_base: R(current) × α^(depth_modifier)
        }
    
    // Select top-k for immediate exploration
    top_k ← select_top(candidate_pool, k=5, by="R_base × S_total")
    
    For each p in top_k:
        R_child ← R(current) × α × S_total(p)
        If R_child > ε:
            If direction == "forward":
                Q_forward.push((p, R_child, current.depth+1))
            Else:
                Q_backward.push((p, R_child, current.depth-1))
    
    // Lazy duplication boost check (every M iterations)
    If iteration % M == 0:
        For each p in candidate_pool where p.count > 1:
            boost ← log2(1 + sum(R(c) for c in p.citing_papers))
            R_boosted ← p.R_base × (1 + β × boost)
            If R_boosted > ε AND p not in explored:
                // Resurrect paper
                Q_forward.push((p, R_boosted, infer_depth(p)))
    
    // Human feedback integration
    If new_feedback_available():
        annotations ← get_latest_feedback()
        seed_embedding ← update_embedding(seed_embedding, annotations)
        
        // Re-score top candidates
        For each p in candidate_pool.top(50):
            S_sim_new ← similarity_score(p, seed_embedding)
            If S_sim_new significantly increased:
                R_adjusted ← p.R_base × S_sim_new
                If R_adjusted > ε:
                    Q_forward.push((p, R_adjusted, p.depth))
    
    explored[current.id] ← current
    budget++

// Present results
high_relevance ← filter(explored, R > 0.3)
sort(high_relevance, by=R, descending)
return high_relevance
```

## Parameter Research: Alpha Selection

### Theoretical Bounds

**Convergence Requirement**: \( \alpha < 1 \)

**Depth-Relevance Relationship**:

At depth \( d \), maximum relevance: \( R_{\max}(d) = \alpha^d \)

Setting exploration cutoff at \( R < \epsilon = 0.02 \):

\[
d_{\max} = \frac{\log(\epsilon)}{\log(\alpha)}
\]

| α | d_max (ε=0.02) | Expected Papers | Exploration Pattern |
|-----|----------------|-----------------|---------------------|
| 0.5 | 5.6 hops | ~150 | Narrow, deep |
| 0.6 | 6.8 hops | ~250 | Balanced |
| 0.7 | 8.3 hops | ~400 | Broad |
| 0.85 | 12.6 hops | ~800 | Very broad |

### Recommended Strategy

**Phase 1**: Start with \( \alpha = 0.65 \)
- Provides ~7 hops depth
- Explores ~300 papers at k=5
- Balanced breadth/depth

**Phase 2**: Adaptive tuning based on metrics

\[
\alpha_{\text{optimal}} = \arg\max_{\alpha} \left[ F_1(\text{relevance}) - \lambda_{\text{cost}} \cdot |E| \right]
\]

Where:
- \( F_1 \) = harmonic mean of precision/recall (user labels subset)
- \( |E| \) = number of papers explored (cost)
- \( \lambda_{\text{cost}} \approx 0.001 \) = cost penalty

## Streaming Protocol

### Three-Phase Streaming

**Phase 1: Immediate (First 3 papers)**
```
For papers 1-3:
    - Full PDF download
    - Annotator highlights key passages
    - Stream to UI within 10 seconds
    - User begins reading, providing feedback
```

**Phase 2: Background Exploration**
```
While user reads:
    - Continue priority search
    - Build candidate_pool
    - Apply duplication boosts
    - Queue high-R papers for Phase 3
```

**Phase 3: Continuous Delivery**
```
Every 30 seconds:
    - Stream next highest-R unexplored paper
    - Include annotations
    - Incorporate any user feedback from Phase 1 papers
    - Adjust seed_embedding
```

### Feedback Signal Processing

**User provides annotations on Paper i at time t**:

```
annotation_embedding ← embed(user_text)
relevance_signal ← user_rating ∈ [-1, 1]  // -1=irrelevant, 0=neutral, 1=highly relevant

// Update seed (exponential moving average)
seed_embedding ← (1-γ) × seed_embedding + γ × annotation_embedding

// Adjust parameters
If relevance_signal < -0.5:
    // User finds papers irrelevant → explore more conservatively
    α ← α × 0.95  // Decay faster
    ε ← ε × 1.1   // Raise threshold
    
If relevance_signal > 0.5:
    // User finds papers relevant → explore more aggressively  
    α ← min(α × 1.05, 0.85)  // Decay slower
    ε ← max(ε × 0.9, 0.01)   // Lower threshold
```

Where \( \gamma \approx 0.3 \) = feedback learning rate

## Complexity Analysis

### Time Complexity

**Per iteration**: \( O(C \log N + C \cdot E) \)
- \( C \) = citations per paper ≈ 30
- \( N \) = queue size
- \( E \) = embedding dimension ≈ 768

**Duplication boost check** (every M iterations): \( O(P \log P) \)
- \( P \) = candidate pool size

**Total**: \( O(N \cdot C \log N) \approx O(N \log N) \) for N papers explored

### Space Complexity

\( O(N + P + E) \)
- \( N \) = explored papers (full metadata)
- \( P \) = candidate pool (metadata only, no PDFs)
- \( E \) = edges in citation graph

**Estimate**: For N=500, P=2000, E=15000 → ~50MB in-memory

## Open Questions Requiring Empirical Testing

1. **Optimal \( \beta \) for duplication boost**: Test range [0.05, 0.3]

2. **Forward/backward balance**: Should \( \alpha_{\text{forward}} \neq \alpha_{\text{backward}} \)?

3. **Feedback integration rate \( \gamma \)**: Too high = unstable, too low = unresponsive

4. **Re-evaluation frequency M**: Trade-off between boost effectiveness and computational cost

5. **Similarity weight \( \omega_{\text{sim}} \) vs quality weight \( \omega_{\text{qual}} \)**: Domain-dependent (ML vs biology vs physics)

## Implementation Priority

1. Core bidirectional search (Agents 1A, 1B, 2)
2. Lazy evaluation candidate pool (Agent 4)
3. Streaming + annotation (Agent 5)
4. Duplication boost mechanism
5. Human feedback loop (Agent 6)
6. Parameter tuning infrastructure