"""
Core Schemas for Evidence-Grounded HopfieldQA RAG Pipeline
==========================================================
Defines typed, immutable data contracts for:
- Evidence chunks and collections
- Claim-level citation tracking
- Structured grounded answers
- Engine and provider configurations
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class EvidenceChunk:
    """A standardized evidence passage retrieved from document or web memory."""
    chunk_id: str
    doc_id: str
    source: str
    text: str
    similarity_score: float
    hopfield_attention: float = 0.0
    rank: int = 1
    page_number: Optional[int] = None
    char_offset_start: Optional[int] = None
    char_offset_end: Optional[int] = None
    is_core_attractor: bool = True
    source_type: str = "document"  # "document" | "web" | "cached_knowledge"
    url: Optional[str] = None
    title: Optional[str] = None
    published_date: Optional[str] = None
    content_hash: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def citation_header(self, cite_idx: int) -> str:
        """Citation header formatted with source distinction [D#] vs [W#]."""
        if self.source_type == "web":
            url_part = f" ({self.url})" if self.url else ""
            return f"[W{cite_idx}] Web: {self.source}{url_part}"
        elif self.source_type == "cached_knowledge":
            return f"[K{cite_idx}] Cached: {self.source}"
        return f"[D{cite_idx}] Source: {self.source}"

    def to_citation_header(self, cite_idx: int) -> str:
        """Standardized citation string for prompts and UI."""
        return self.citation_header(cite_idx)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize chunk to dictionary with all web and provenance fields."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "source": self.source,
            "text": self.text,
            "similarity_score": self.similarity_score,
            "hopfield_attention": self.hopfield_attention,
            "rank": self.rank,
            "page_number": self.page_number,
            "char_offset_start": self.char_offset_start,
            "char_offset_end": self.char_offset_end,
            "is_core_attractor": self.is_core_attractor,
            "source_type": self.source_type,
            "url": self.url,
            "title": self.title,
            "published_date": self.published_date,
            "content_hash": self.content_hash,
            "metadata": self.metadata
        }


@dataclass
class EvidenceCollection:
    """Collection of retrieved evidence with query-level diagnostics."""
    query: str
    chunks: List[EvidenceChunk]
    retriever_name: str
    retrieval_time_ms: float = 0.0
    attractor_energy: Optional[float] = None
    energy_depth: Optional[float] = None
    is_grounded: bool = True
    trajectory_hops: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return len(self.chunks) == 0

    @property
    def sources(self) -> List[str]:
        return list(dict.fromkeys(c.source for c in self.chunks))

    @property
    def top_score(self) -> float:
        return self.chunks[0].similarity_score if self.chunks else 0.0

    def get_chunk_by_id(self, chunk_id: str) -> Optional[EvidenceChunk]:
        for c in self.chunks:
            if c.chunk_id == chunk_id:
                return c
        return None


@dataclass
class ClaimCitation:
    """Represents a claim and the evidence chunk IDs supporting it."""
    claim_text: str
    evidence_ids: List[str]
    is_verified: bool = True
    unresolved_ids: List[str] = field(default_factory=list)
    is_supported: bool = True
    is_relevant: bool = True
    is_unique: bool = True
    verification_rationale: Optional[str] = None


@dataclass
class GroundedAnswer:
    """Complete, evidence-grounded answer produced by the synthesis layer."""
    query: str
    answer: str
    claims: List[ClaimCitation]
    cited_evidence: List[EvidenceChunk]
    abstained: bool
    abstention_reason: Optional[str]
    confidence: float
    provider_used: str
    model_name: str
    generation_time_ms: float
    style: str = "conversational"
    images: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def fallback_occurred(self) -> bool:
        return self.metadata.get("fallback_occurred", False)

    @property
    def fallback_reason(self) -> Optional[str]:
        return self.metadata.get("fallback_reason")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "images": self.images,
            "abstained": self.abstained,
            "abstention_reason": self.abstention_reason,
            "confidence": round(self.confidence, 4),
            "provider_used": self.provider_used,
            "model_name": self.model_name,
            "generation_time_ms": round(self.generation_time_ms, 2),
            "style": self.style,
            "claims": [
                {
                    "claim_text": c.claim_text,
                    "evidence_ids": c.evidence_ids,
                    "is_verified": c.is_verified,
                    "unresolved_ids": c.unresolved_ids,
                    "is_supported": getattr(c, 'is_supported', c.is_verified),
                    "is_relevant": getattr(c, 'is_relevant', True),
                    "is_unique": getattr(c, 'is_unique', True),
                    "verification_rationale": getattr(c, 'verification_rationale', None)
                } for c in self.claims
            ],
            "cited_evidence": [
                {
                    "chunk_id": c.chunk_id,
                    "source": c.source,
                    "source_type": getattr(c, "source_type", "document"),
                    "url": getattr(c, "url", None),
                    "title": getattr(c, "title", None),
                    "published_date": getattr(c, "published_date", None),
                    "text": c.text,
                    "similarity_score": round(c.similarity_score, 4),
                    "hopfield_attention": round(c.hopfield_attention, 4),
                    "is_core_attractor": c.is_core_attractor
                } for c in self.cited_evidence
            ],
            "metadata": self.metadata
        }
