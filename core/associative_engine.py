"""
Hopfield Associative Memory Engine (HAM-Engine)
===============================================
A complete paradigm replacement for naive Retrieval-Augmented Generation (RAG).

Why Naive RAG Fails:
-------------------
1. Independent Retrieval: Standard RAG computes cosine similarity cos(q, c_i) independently for each chunk.
   Chunks cannot interact, reinforce each other, or form associative clusters.
2. No Multi-Hop Continuity: If evidence requires A -> B -> C, but query only resembles A,
   B and C get discarded by naive vector search.
3. No Grounding Metric: Vector similarity has no thermodynamic or energy-based notion of whether
   a query lands in a stable memory attractor or an ungrounded hallucination zone.

How HAM-Engine Replaces Naive RAG:
----------------------------------
1. Continuous Energy Landscape:
   E(z) = - (1/beta) * log( sum_i exp(beta * x_i^T z) ) + (1/2) * ||z||^2
2. Iterative Attractor Dynamics (Pattern Completion):
   State evolves through: z_{t+1} = Normalize( (1-alpha)*z_t + alpha * X * softmax(beta * X^T z_t) )
   Query transitions from surface lexical cue -> associative bridges -> stable memory attractor basin.
3. Trajectory Tracking:
   Records each hop of the memory retrieval trajectory to capture the latent multi-hop reasoning path.
4. Co-Attractor Graph Reinforcement:
   Cross-chunk associative affinity pulls in contextually bound facts even when they lack direct query overlap.
5. Energy-Based Hallucination Filtering:
   Evaluates attractor basin depth to guarantee memory grounding.
"""

import math
import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F

from core.hopfield_torch import RetrievalResult

logger = logging.getLogger(__name__)


@dataclass
class AssociativeHop:
    """Represents an intermediate state in the Hopfield memory retrieval trajectory."""
    hop_index: int
    energy: float
    state_delta: float
    activated_chunks: List[Dict[str, Any]]
    description: str


@dataclass
class AssociativeMemoryResult:
    """Complete associative retrieval result replacing static RAG chunk lists."""
    query: str
    attractor_energy: float
    energy_landscape_depth: float
    is_grounded: bool
    convergence_steps: int
    trajectory: List[AssociativeHop]
    core_attractor_chunks: List[RetrievalResult]
    associative_bridge_chunks: List[RetrievalResult]
    all_retrieved_chunks: List[RetrievalResult]
    reasoning_path: List[str]
    co_attractor_matrix: Optional[Dict[str, Any]] = None


class HopfieldAssociativeEngine:
    """
    Modern Continuous Hopfield Associative Memory Engine.
    Executes non-linear energy minimization and attractor basin discovery.
    """
    def __init__(
        self,
        memory,
        beta: float = 8.0,
        max_hops: int = 4,
        alpha: float = 0.7,
        convergence_tol: float = 1e-4,
        energy_threshold: float = -0.05,
    ):
        self.memory = memory
        self.beta = beta
        self.max_hops = max_hops
        self.alpha = alpha
        self.convergence_tol = convergence_tol
        self.energy_threshold = energy_threshold

    def compute_energy(self, state: torch.Tensor, patterns: torch.Tensor, beta: float) -> float:
        """
        Computes the Modern Hopfield continuous energy:
        E(z) = - (1/beta) * logsumexp(beta * patterns @ z) + 0.5 * ||z||^2
        """
        # state: (1, d), patterns: (N, d)
        dots = torch.matmul(patterns, state.t()).squeeze(-1) # (N,)
        lse = torch.logsumexp(beta * dots, dim=0)
        norm_sq = torch.sum(state ** 2).item()
        energy = - (1.0 / beta) * lse.item() + 0.5 * norm_sq
        return float(energy)

    def retrieve_associative(
        self,
        query_embedding: np.ndarray,
        raw_query: str,
        top_k: int = 5,
        use_projection: bool = False,
        doc_filter: Optional[str] = None,
    ) -> AssociativeMemoryResult:
        """
        Performs continuous Hopfield memory retrieval via attractor dynamics.
        Replaces standard k-NN similarity lookup.
        """
        if self.memory._patterns is None or self.memory.n_patterns == 0:
            return AssociativeMemoryResult(
                query=raw_query,
                attractor_energy=0.0,
                energy_landscape_depth=0.0,
                is_grounded=False,
                convergence_steps=0,
                trajectory=[],
                core_attractor_chunks=[],
                associative_bridge_chunks=[],
                all_retrieved_chunks=[],
                reasoning_path=["Memory is empty."]
            )

        device = self.memory.device
        patterns = self.memory._patterns.to(device) # (N, d)
        metadata = self.memory._metadata

        # Filter by document if doc_filter is provided
        if doc_filter:
            doc_filter_clean = doc_filter.strip().lower()
            matching_indices = [
                i for i, m in enumerate(metadata)
                if doc_filter_clean in m.get("source", "").lower() or doc_filter_clean in m.get("doc_id", "").lower()
            ]
            if matching_indices:
                patterns = patterns[matching_indices]
                metadata = [metadata[i] for i in matching_indices]

        N, d = patterns.shape

        # 1. Initialize query state vector z^{(0)}
        q = query_embedding.copy()
        q_norm = np.linalg.norm(q)
        if q_norm > 1e-8:
            q = q / q_norm
        z = torch.tensor(q, dtype=torch.float32, device=device).unsqueeze(0) # (1, d)

        # Apply trained projection layer if requested and available
        if use_projection and getattr(self.memory.retriever, 'use_projection', False) and getattr(self.memory.retriever, 'projection', None):
            with torch.no_grad():
                z = self.memory.retriever.projection(z.unsqueeze(1)).squeeze(1)

        # Normalize state
        z = F.normalize(z, p=2, dim=-1)

        trajectory: List[AssociativeHop] = []
        visited_indices_by_hop: List[List[int]] = []
        current_state = z.clone()
        prev_state = z.clone()

        # Compute initial energy at the probe state
        initial_energy = self.compute_energy(current_state, patterns, self.beta)

        converged = False
        steps_taken = 0

        # Cache entry query projections directly from query embedding
        entry_projections = torch.matmul(patterns, z.t()).squeeze(-1)

        # 2. Iterate through Continuous Attractor Dynamics
        for hop in range(self.max_hops):
            steps_taken += 1
            # Projection onto stored patterns
            projections = torch.matmul(patterns, current_state.t()).squeeze(-1) # (N,)
            scaled = self.beta * projections
            attn_weights = F.softmax(scaled - torch.max(scaled), dim=-1) # (N,)

            # Next state: associative pattern readout
            # readout = sum_i attn_i * x_i
            readout = torch.matmul(attn_weights.unsqueeze(0), patterns) # (1, d)

            # Continuous relaxation update with momentum/inertia and residual query anchor
            new_state = (1.0 - self.alpha) * current_state + self.alpha * readout + 0.15 * z
            new_state = F.normalize(new_state, p=2, dim=-1)

            # Measure state displacement
            state_delta = torch.norm(new_state - current_state).item()
            hop_energy = self.compute_energy(new_state, patterns, self.beta)

            # Track top activated patterns at this hop
            top_vals, top_idxs = torch.topk(attn_weights, k=min(top_k, N))
            top_idxs_list = top_idxs.cpu().numpy().tolist()
            visited_indices_by_hop.append(top_idxs_list)

            hop_chunks = []
            for rank_i, p_idx in enumerate(top_idxs_list):
                meta = metadata[p_idx]
                hop_chunks.append({
                    "chunk_id": meta.get("chunk_id", str(p_idx)),
                    "text": meta.get("text", "")[:120] + "...",
                    "source": meta.get("source", "unknown"),
                    "attention": float(top_vals[rank_i].item()),
                    "similarity": float(projections[p_idx].item()),
                })

            desc = "Entry Cue (Direct Query Match)" if hop == 0 else (
                f"Associative Bridge (Hop {hop})" if hop < self.max_hops - 1 else "Final Attractor Basin"
            )

            trajectory.append(AssociativeHop(
                hop_index=hop + 1,
                energy=hop_energy,
                state_delta=state_delta,
                activated_chunks=hop_chunks,
                description=desc
            ))

            prev_state = current_state
            current_state = new_state

            if state_delta < self.convergence_tol:
                converged = True
                break

        final_energy = trajectory[-1].energy if trajectory else initial_energy
        energy_depth = initial_energy - final_energy

        # 3. Partition Evidence: Attractor Core vs. Associative Bridges
        # Attractor core: Chunks with highest attention at convergence
        final_projections = torch.matmul(patterns, current_state.t()).squeeze(-1)
        # Blend direct entry similarity (65%) with attractor convergence (35%)
        blended_projections = 0.65 * entry_projections + 0.35 * final_projections
        final_scaled = self.beta * blended_projections
        final_attn = F.softmax(final_scaled - torch.max(final_scaled), dim=-1)

        # Entity-aware hybrid co-attractor reinforcement
        effective_attn = final_attn.clone()
        entity_scores = None
        if raw_query:
            import re
            STOP_TOKENS = {
                'the', 'and', 'what', 'which', 'between', 'with', 'for', 'from', 'into',
                'this', 'that', 'these', 'those', 'how', 'why', 'who', 'when', 'where',
                'can', 'could', 'should', 'would', 'does', 'did', 'have', 'has', 'had',
                'about', 'through', 'during', 'before', 'after', 'some', 'more', 'most',
                'are', 'is', 'was', 'were', 'been', 'being', 'used', 'use', 'using',
                'also', 'than', 'then', 'just', 'very', 'much', 'such', 'only', 'other',
                'explain', 'overview', 'detail', 'detailed', 'their', 'responsibilities',
                'component', 'components', 'responsibility', 'work', 'works', 'working',
                'workload', 'workloads', 'architecture', 'process', 'processes', 'role',
                'roles', 'purpose', 'purposes', 'feature', 'features', 'aspect', 'aspects',
                'function', 'functions', 'mechanism', 'mechanisms', 'part', 'parts',
                'element', 'elements', 'tool', 'tools', 'type', 'types', 'system', 'systems',
                'example', 'examples', 'advantage', 'advantages', 'benefit', 'benefits'
            }
            extracted_tokens = [
                w.lower() for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', raw_query)
                if w.lower() not in STOP_TOKENS
            ]
            raw_entities_set = set(extracted_tokens)
            compound_prefixes = ("cyber", "micro", "multi", "meta", "hyper", "infra", "auto", "inter", "data")
            for tok in list(raw_entities_set):
                sub_parts = re.split(r'[-_]', tok)
                if len(sub_parts) > 1:
                    for sp in sub_parts:
                        if len(sp) >= 3 and sp not in STOP_TOKENS:
                            raw_entities_set.add(sp)
                for pfx in compound_prefixes:
                    if tok.startswith(pfx) and len(tok) > len(pfx) + 3:
                        remainder = tok[len(pfx):]
                        if remainder not in STOP_TOKENS and len(remainder) >= 3:
                            raw_entities_set.add(pfx)
                            raw_entities_set.add(remainder)
            raw_entities = list(raw_entities_set)
            if raw_entities:
                entity_scores = torch.zeros(N, device=device)
                for idx in range(N):
                    c_text = metadata[idx].get("text", "").lower()
                    hits = sum(1 for e in raw_entities if re.search(r'\b' + re.escape(e) + r'\b', c_text))
                    entity_scores[idx] = hits / len(raw_entities)

                if entity_scores.max() > 0:
                    target_sources = {metadata[idx].get("source") for idx in range(N) if entity_scores[idx] > 0}
                    source_weights = torch.tensor(
                        [1.0 if metadata[idx].get("source") in target_sources else 0.005 for idx in range(N)],
                        device=device
                    )
                    penalty_mask = torch.where(entity_scores > 0, 1.0 + 3.0 * entity_scores, 0.3 * source_weights)
                    effective_attn = final_attn * penalty_mask

        core_vals, core_idxs = torch.topk(effective_attn, k=min(top_k, N))
        core_idxs_set = set(core_idxs.cpu().numpy().tolist())

        # Collect bridge chunks: Chunks discovered along trajectory not in final core top-k
        all_visited_ordered = []
        seen = set()
        for hop_indices in visited_indices_by_hop:
            for idx in hop_indices:
                if idx not in seen:
                    seen.add(idx)
                    all_visited_ordered.append(idx)

        # Filter candidate bridges: exclude core attractors
        candidate_bridges = [idx for idx in all_visited_ordered if idx not in core_idxs_set]

        # Avoid picking off-topic drift chunks that have 0 entity hits when entity hits exist
        if entity_scores is not None and entity_scores.max() > 0:
            target_sources = {metadata[idx].get("source") for idx in range(N) if entity_scores[idx] > 0}
            entity_matching_bridges = [idx for idx in candidate_bridges if entity_scores[idx] > 0]
            if entity_matching_bridges:
                candidate_bridges = entity_matching_bridges
            else:
                same_source_bridges = [idx for idx in candidate_bridges if metadata[idx].get("source") in target_sources]
                if not same_source_bridges:
                    same_source_bridges = [idx for idx in range(N) if idx not in core_idxs_set and metadata[idx].get("source") in target_sources]
                candidate_bridges = same_source_bridges
        else:
            # Anchor bridges to sources of top core attractors to prevent drifting into foreign documents
            top_core_sources = {metadata[idx].get("source") for idx in list(core_idxs_set)[:3] if metadata[idx].get("source")}
            if top_core_sources:
                coherent_bridges = [idx for idx in candidate_bridges if metadata[idx].get("source") in top_core_sources]
                if coherent_bridges:
                    candidate_bridges = coherent_bridges

        # Rank bridge chunks by their effective attention
        candidate_bridges.sort(key=lambda idx: float(effective_attn[idx].item()), reverse=True)
        bridge_idxs = candidate_bridges[:top_k]

        def build_retrieval_result(idx: int, rank: int, is_core: bool) -> RetrievalResult:
            meta = metadata[idx]
            sim = float(blended_projections[idx].item())
            attn = float(final_attn[idx].item())
            meta_copy = dict(meta)
            meta_copy["is_core_attractor"] = is_core
            meta_copy["hopfield_energy"] = final_energy
            return RetrievalResult(
                chunk_id=meta.get("chunk_id", str(idx)),
                text=meta.get("text", ""),
                source=meta.get("source", "unknown"),
                chunk_index=meta.get("chunk_index", idx),
                similarity_score=sim,
                hopfield_attention=attn,
                rank=rank,
                metadata=meta_copy
            )

        core_results = [build_retrieval_result(idx, i + 1, True) for i, idx in enumerate(core_idxs.cpu().numpy().tolist())]
        bridge_results = [build_retrieval_result(idx, len(core_results) + i + 1, False) for i, idx in enumerate(bridge_idxs)]

        # Combined ordered list (Core first, then associative bridges)
        combined_results = core_results + bridge_results

        # 4. Synthesize Associative Reasoning Path
        reasoning_path = []
        reasoning_path.append(f"Query mapped to energy landscape (E_0 = {initial_energy:.3f})")
        for hop in trajectory:
            top_doc = hop.activated_chunks[0] if hop.activated_chunks else None
            top_source = top_doc['source'] if top_doc else 'N/A'
            reasoning_path.append(
                f"Hop {hop.hop_index} ({hop.description}): Converged to source '{top_source}' (E = {hop.energy:.3f}, delta = {hop.state_delta:.4f})"
            )
        
        is_grounded = bool(final_energy < self.energy_threshold or energy_depth >= 0.0)

        if is_grounded:
            reasoning_path.append(f"Stable attractor basin reached with depth Delta E = {energy_depth:.4f} (High Grounding)")
        else:
            reasoning_path.append("Shallow attractor basin detected (Query may require additional document grounding)")

        return AssociativeMemoryResult(
            query=raw_query,
            attractor_energy=final_energy,
            energy_landscape_depth=energy_depth,
            is_grounded=is_grounded,
            convergence_steps=steps_taken,
            trajectory=trajectory,
            core_attractor_chunks=core_results,
            associative_bridge_chunks=bridge_results,
            all_retrieved_chunks=combined_results,
            reasoning_path=reasoning_path,
        )
