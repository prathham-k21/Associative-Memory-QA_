"""
Main Pipeline Orchestrator
===========================
Ties together: Embeddings → Hopfield Memory → Query Processing → Answer Generation
"""

import numpy as np
import time
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import re
import hashlib
import logging

from core.embeddings import DocumentChunker, EmbeddingEngine, Chunk
from core.hopfield_torch import PyTorchHopfieldMemory, RetrievalResult
from core.query_handler import QueryProcessor, ProcessedQuery
from core.answer_generator import AnswerGenerator, GeneratedAnswer
from core.associative_engine import HopfieldAssociativeEngine, AssociativeMemoryResult
from core.schemas import EvidenceChunk, EvidenceCollection, GroundedAnswer
from core.retrievers import HopfieldRetrieverAdapter, DenseRetrieverAdapter, HybridRetrieverAdapter

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the full pipeline."""
    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_size: int = 50

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_batch_size: int = 32

    # Hopfield
    hopfield_beta: float = 8.0
    hopfield_n_iter: int = 1
    similarity_threshold: float = 0.05
    use_projection: bool = False
    projection_path: str = None
    use_associative_engine: bool = True
    default_retriever: str = "hopfield"  # "hopfield" | "dense" | "hybrid"
    persistence_dir: str = "data"

    # Retrieval
    top_k: int = 5
    use_fusion: bool = True
    n_query_variants: int = 3

    # Answer
    max_answer_length: int = 16000
    min_confidence: float = 0.1

    # LLM Answer Generation
    llm_provider: str = "ollama"  # "ollama" | "openai" | "mock" | "deterministic"
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "llama3.2"
    llm_api_key: Optional[str] = None
    llm_timeout_seconds: float = 60.0
    llm_max_tokens: int = 8192
    llm_temperature: float = 0.2
    llm_fallback_to_deterministic: bool = True

    # Web Augmentation & Persistent Knowledge Memory
    web_search_enabled: bool = False
    web_search_provider: str = "mock"  # "mock" | "duckduckgo"
    web_fetch_timeout_seconds: float = 8.0
    web_cache_path: str = "data/web_memory_store.json"
    sufficiency_min_chunks: int = 2
    sufficiency_min_similarity: float = 0.35
    sufficiency_min_entity_coverage: float = 0.5

    def to_dict(self) -> Dict:
        return self.__dict__.copy()


@dataclass
class QueryResult:
    """Complete result of a pipeline query."""
    query: str
    processed_query: ProcessedQuery
    retrieved_chunks: List[RetrievalResult]
    answer: GeneratedAnswer
    retrieval_time_ms: float
    total_time_ms: float
    attention_landscape: Optional[Dict] = None
    associative_result: Optional[AssociativeMemoryResult] = None
    grounded_answer: Optional[GroundedAnswer] = None
    evidence_collection: Optional[EvidenceCollection] = None
    stage_latencies_ms: Dict[str, float] = field(default_factory=dict)
    deduplication_diagnostics: Dict[str, Any] = field(default_factory=dict)
    web_search_used: bool = False
    cached_knowledge_used: bool = False
    format_detected: Optional[str] = None
    format_requested: Optional[str] = None
    sufficiency_decision: Optional[Dict[str, Any]] = None
    web_sources: List[Dict[str, Any]] = field(default_factory=list)


class HopfieldQAPipeline:
    """
    End-to-end QA pipeline using Modern Hopfield Networks.

    Architecture:
    ┌────────────────────────────────────────────────────────┐
    │  Document Ingestion                                    │
    │  ─────────────────────────────────────────────────     │
    │  Text → Chunks → Embeddings → Hopfield Memory Store   │
    └────────────────────────────────────────────────────────┘
                          ↓
    ┌────────────────────────────────────────────────────────┐
    │  Query Processing                                      │
    │  ─────────────────────────────────────────────────     │
    │  Raw Query → Clean → Expand → Variants               │
    └────────────────────────────────────────────────────────┘
                          ↓
    ┌────────────────────────────────────────────────────────┐
    │  Hopfield Retrieval                                    │
    │  ─────────────────────────────────────────────────     │
    │  Query Embeddings → Energy Minimization → Top-K      │
    └────────────────────────────────────────────────────────┘
                          ↓
    ┌────────────────────────────────────────────────────────┐
    │  Answer Generation                                     │
    │  ─────────────────────────────────────────────────     │
    │  Chunks → Sentence Extraction → Answer + Confidence  │
    └────────────────────────────────────────────────────────┘
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()

        # Initialize components
        self.chunker = DocumentChunker(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            min_chunk_size=self.config.min_chunk_size,
        )
        self.embedder = EmbeddingEngine(
            model_name=self.config.embedding_model,
            batch_size=self.config.embedding_batch_size,
        )
        self.memory = PyTorchHopfieldMemory(
            beta=self.config.hopfield_beta,
            n_iter=self.config.hopfield_n_iter,
            similarity_threshold=self.config.similarity_threshold,
            energy_type=getattr(self.config, 'energy_type', 'softmax'),
            use_projection=self.config.use_projection
        )
        if self.config.use_projection and self.config.projection_path:
            import torch
            try:
                self.memory.retriever.projection.load_state_dict(torch.load(self.config.projection_path, map_location=self.memory.device, weights_only=True))
                print(f"Loaded trained projection layer from {self.config.projection_path}")
            except Exception as e:
                print(f"Could not load projection weights: {e}")
        self.query_processor = QueryProcessor()
        from core.generative_synthesizer import GenerativeRAGSynthesizer
        from core.llm_provider import LLMConfig
        llm_cfg = LLMConfig(
            provider=getattr(self.config, 'llm_provider', 'ollama'),
            base_url=getattr(self.config, 'llm_base_url', 'http://localhost:11434/v1'),
            model=getattr(self.config, 'llm_model', 'llama3.2'),
            api_key=getattr(self.config, 'llm_api_key', None),
            timeout_seconds=getattr(self.config, 'llm_timeout_seconds', 60.0),
            max_tokens=getattr(self.config, 'llm_max_tokens', 4096),
            temperature=getattr(self.config, 'llm_temperature', 0.2),
            fallback_to_deterministic=getattr(self.config, 'llm_fallback_to_deterministic', True)
        )
        self.answer_generator = GenerativeRAGSynthesizer(
            llm_config=llm_cfg
        )
        self.associative_engine = HopfieldAssociativeEngine(
            memory=self.memory,
            beta=self.config.hopfield_beta,
            max_hops=4,
            alpha=0.7,
        )
        self.hopfield_retriever = HopfieldRetrieverAdapter(self.associative_engine)
        self.dense_retriever = DenseRetrieverAdapter(self.memory)
        self.hybrid_retriever = HybridRetrieverAdapter(self.hopfield_retriever, self.memory)

        # Web Search & Persistent Knowledge Memory
        from core.knowledge_memory import PersistentKnowledgeMemory
        from core.web_search import DuckDuckGoSearchProvider, MockWebSearchProvider, WebPageExtractor, WebResearchConfig
        from core.retrieval_coordinator import RetrievalCoordinator, SufficiencyEvaluator
        from core.format_detector import FormatDetector

        self.knowledge_memory = PersistentKnowledgeMemory(
            filepath=getattr(self.config, 'web_cache_path', 'data/web_memory_store.json')
        )
        provider_name = getattr(self.config, 'web_search_provider', 'duckduckgo').lower()
        if provider_name == "mock":
            self.web_search_provider = MockWebSearchProvider()
        else:
            self.web_search_provider = DuckDuckGoSearchProvider()

        self.web_extractor = WebPageExtractor(
            mock_provider=self.web_search_provider if isinstance(self.web_search_provider, MockWebSearchProvider) else None,
            timeout=getattr(self.config, 'web_fetch_timeout_seconds', 8.0)
        )
        self.web_config = WebResearchConfig(
            enabled=getattr(self.config, 'web_search_enabled', False),
            provider=provider_name,
            fetch_timeout_seconds=getattr(self.config, 'web_fetch_timeout_seconds', 8.0)
        )
        self.retrieval_coordinator = RetrievalCoordinator(
            doc_retriever=self.hopfield_retriever,
            embedder=self.embedder,
            chunker=self.chunker,
            knowledge_memory=self.knowledge_memory,
            web_search_provider=self.web_search_provider,
            web_extractor=self.web_extractor,
            web_config=self.web_config
        )
        self.format_detector = FormatDetector()

        # State
        self._all_chunks: List[Chunk] = []
        self._documents: Dict[str, str] = {}  # source → original text
        self._document_images: Dict[str, List[Dict[str, Any]]] = {}  # source → list of image metadata dicts
        self._is_ready = False

        # Document Persistence recovery
        self._recover_persisted_index()

    def _recover_persisted_index(self):
        """Recover indexed documents and their visual figures from disk if available."""
        pers_path = Path(self.config.persistence_dir) / "memory_store.json"
        if not pers_path.exists() and self.config.persistence_dir == "data":
            pers_path = Path(__file__).resolve().parent.parent / "data" / "memory_store.json"
        if pers_path.exists():
            try:
                with open(pers_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    docs = data.get("documents", {})
                    doc_images = data.get("document_images", {})
                    if docs:
                        logger.info(f"Recovering {len(docs)} persisted documents from {pers_path}...")
                        print(f"Recovering {len(docs)} persisted documents from {pers_path}...")
                        for source, text in docs.items():
                            imgs = doc_images.get(source, [])
                            self.add_document(text, source=source, persist=False, images=imgs)
            except Exception as e:
                logger.warning(f"Could not recover persisted document index: {e}")
                print(f"Could not recover persisted document index: {e}")

    @property
    def is_ready(self) -> bool:
        return self._is_ready and self.memory.n_patterns > 0

    @property
    def n_documents(self) -> int:
        return len(self._documents)

    @property
    def n_chunks(self) -> int:
        return len(self._all_chunks)

    def add_document(
        self,
        text: str,
        source: str = "document",
        progress_callback=None,
        persist: bool = True,
        images: Optional[List[Any]] = None,
    ) -> Dict:
        """
        Add a document to the Hopfield memory.

        Process:
        1. Chunk the text
        2. Embed chunks + any extracted visual figure chunks
        3. Store in Hopfield network

        Returns stats about the added document.
        """
        if not text.strip() and not images:
            return {"error": "Empty document text and no images"}

        t0 = time.time()

        # Chunk text if text is provided
        chunks = []
        if text.strip():
            if progress_callback:
                progress_callback("Chunking document...", 0.2)
            chunks = self.chunker.chunk_text(text, source=source)

        # Create image-anchored chunks for extracted visual figures/diagrams
        if images:
            img_meta_list = []
            for idx, img in enumerate(images):
                img_data = img.to_dict() if hasattr(img, "to_dict") else img
                img_meta_list.append(img_data)
                img_url = img_data.get("image_url", "")
                caption = img_data.get("caption", f"Figure {idx+1}")
                context_snippet = img_data.get("context_text", "")
                page_num = img_data.get("page_number", 1)

                img_text = f"[Visual Figure: {caption}]\nDocument: {source} (Page {page_num})\nVisual Content & Context: {context_snippet}"
                chunk_id = f"img_{hashlib.md5((source + str(page_num) + img_url).encode()).hexdigest()[:12]}"
                img_chunk = Chunk(
                    chunk_id=chunk_id,
                    text=img_text,
                    source=source,
                    chunk_index=len(chunks) + idx,
                    start_char=0,
                    end_char=len(img_text),
                    metadata={
                        "has_image": True,
                        "image_url": img_url,
                        "image_caption": caption,
                        "page_number": page_num,
                        "source_type": "document",
                        "width": img_data.get("width"),
                        "height": img_data.get("height"),
                    }
                )
                chunks.append(img_chunk)
            self._document_images[source] = img_meta_list

        # Embed all chunks (text + visual)
        if progress_callback:
            progress_callback("Generating embeddings...", 0.5)
        chunks = self.embedder.embed_chunks(chunks, show_progress=False)

        # Store original
        self._documents[source] = text
        self._all_chunks.extend(chunks)

        # Rebuild Hopfield memory with all chunks
        if progress_callback:
            progress_callback("Storing in Hopfield memory...", 0.8)
        self._rebuild_memory()
        
        logger.info(f"Added document '{source}' with {len(chunks)} chunks.")

        # Persist documents to disk if requested
        if persist:
            try:
                p_dir = Path(self.config.persistence_dir)
                p_dir.mkdir(parents=True, exist_ok=True)
                with open(p_dir / "memory_store.json", "w", encoding="utf-8") as f:
                    json.dump({"documents": self._documents, "document_images": self._document_images}, f, indent=2)
            except Exception as e:
                logger.warning(f"Could not persist document store: {e}")

        elapsed = (time.time() - t0) * 1000
        self._is_ready = True

        if progress_callback:
            progress_callback("Done!", 1.0)

        return {
            "source": source,
            "n_chunks": len(chunks),
            "n_total_chunks": self.n_chunks,
            "embedding_dim": self.embedder.embedding_dim,
            "processing_time_ms": elapsed,
        }

    def add_documents(
        self,
        documents: List[Tuple[str, str]],  # [(text, source), ...]
        progress_callback=None,
    ) -> List[Dict]:
        """Add multiple documents efficiently."""
        results = []
        for i, (text, source) in enumerate(documents):
            if progress_callback:
                progress_callback(
                    f"Processing {source}...",
                    (i + 1) / len(documents)
                )
            result = self.add_document(text, source)
            results.append(result)
        return results

    def remove_document(self, source: str) -> bool:
        """
        Remove a document and its chunks from memory and persist changes to disk.
        """
        if source not in self._documents:
            return False

        del self._documents[source]
        if source in self._document_images:
            del self._document_images[source]
        self._all_chunks = [c for c in self._all_chunks if c.source != source]

        if self._all_chunks:
            self._rebuild_memory()
        else:
            if hasattr(self.memory, "clear"):
                self.memory.clear()
            self._is_ready = False

        # Persist updated documents to disk
        try:
            p_dir = Path(self.config.persistence_dir)
            p_dir.mkdir(parents=True, exist_ok=True)
            with open(p_dir / "memory_store.json", "w", encoding="utf-8") as f:
                json.dump({"documents": self._documents, "document_images": self._document_images}, f, indent=2)
            logger.info(f"Removed document '{source}' from Hopfield memory store.")
        except Exception as e:
            logger.warning(f"Could not persist document store after removal: {e}")

        return True

    def _rebuild_memory(self):
        """Rebuild Hopfield memory from all stored chunks."""
        if not self._all_chunks:
            return

        # Stack all embeddings
        patterns = np.vstack([c.embedding for c in self._all_chunks])

        # Build metadata list preserving visual attributes
        metadata = [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "source": c.source,
                "chunk_index": c.chunk_index,
                "start_char": c.start_char,
                "end_char": c.end_char,
                **(c.metadata or {})
            }
            for c in self._all_chunks
        ]

        self.memory.store(patterns, metadata)

    def query(
        self,
        raw_query: str,
        top_k: Optional[int] = None,
        include_attention_landscape: bool = False,
        use_associative_mode: bool = True,
        answer_style: str = "conversational",
        retriever_mode: Optional[str] = None,
        llm_provider_override: Optional[Any] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        web_search_enabled: Optional[bool] = None,
        force_web_search: bool = False,
        answer_format: Optional[str] = None,
        active_doc_id: Optional[str] = None,
        doc_filter: Optional[str] = None,
    ) -> QueryResult:
        """
        Execute a complete query through the pipeline with multi-source retrieval coordinator.

        Args:
            raw_query: User's raw query string.
            top_k: Number of chunks to retrieve (defaults to config).
            include_attention_landscape: Whether to include full attention map.
            use_associative_mode: Whether to execute Hopfield continuous attractor dynamics
                                 (replacing static naive RAG similarity).
            answer_style: "conversational", "detailed", "executive", "concise", etc.
            retriever_mode: "hopfield" (Continuous Associative Memory), "dense" (Cosine Baseline),
                           or "hybrid" (BM25 + Neural RRF).
            web_search_enabled: Whether web search augmentation is active.
            force_web_search: Whether to force web research even if documents are present.
            answer_format: Examination mark tier or genre (2_mark, 5_mark, 8_mark, 10_mark, comparison, etc.).
            active_doc_id: Optional document ID or source name to restrict/filter retrieval.
            doc_filter: Alias for active_doc_id.

        Returns:
            QueryResult with answer, evidence, and performance stats.
        """
        effective_doc_filter = active_doc_id or doc_filter
        effective_web_search = self.config.web_search_enabled if web_search_enabled is None else web_search_enabled
        if not self.is_ready and not (effective_web_search or force_web_search):
            raise RuntimeError("Pipeline not ready. Please add documents first.")

        k = top_k or self.config.top_k
        t0 = time.perf_counter()

        # Format & Intent Detection
        explicit_fmt = answer_format or answer_style
        detected_fmt, _ = self.format_detector.detect_format(raw_query, explicit_preference=explicit_fmt)
        active_style = detected_fmt or answer_style or "conversational"
        
        # Determine active retriever mode
        if retriever_mode is None:
            if not use_associative_mode:
                mode = "dense"
            else:
                mode = getattr(self.config, 'default_retriever', 'hopfield')
        else:
            mode = retriever_mode.lower()

        logger.info(f"Processing query: '{raw_query}' (mode={mode}, top_k={k}, style={active_style}, doc_filter={effective_doc_filter}, web_search={effective_web_search})")

        # 1. Process query (with multi-turn coreference contextualization)
        processed = self.query_processor.process(raw_query, conversation_history=conversation_history)

        # 2. Embed query
        t_embed_start = time.perf_counter()
        query_text = getattr(processed, 'cleaned', None) or raw_query
        query_embedding = self.embedder.embed_query(query_text)
        embed_time_ms = max(0.05, (time.perf_counter() - t_embed_start) * 1000)

        # 3. Select retriever adapter
        if mode == "dense":
            selected_retriever = self.dense_retriever
        elif mode == "hybrid":
            selected_retriever = self.hybrid_retriever
        else:
            self.associative_engine.beta = self.config.hopfield_beta
            selected_retriever = self.hopfield_retriever

        # 4. Multi-Source Retrieval via RetrievalCoordinator
        t_ret_start = time.perf_counter()
        self.retrieval_coordinator.doc_retriever = selected_retriever
        coord_res = self.retrieval_coordinator.retrieve(
            query=query_text,
            query_embedding=query_embedding,
            top_k=k,
            format_id=active_style,
            web_search_enabled=effective_web_search,
            force_web_search=force_web_search,
            doc_filter=effective_doc_filter,
        )
        retrieval_time_ms = max(0.1, (time.perf_counter() - t_ret_start) * 1000)

        dedup_evidence_collection = coord_res.evidence_collection
        dedup_diagnostics = coord_res.dedup_diagnostics or {"duplicates_removed": 0}
        dedup_time_ms = 0.1

        associative_res = getattr(self.hopfield_retriever, 'last_result', None) if mode in ("hopfield", "hybrid") else None

        # 5. Optional: attention landscape
        landscape = None
        if include_attention_landscape:
            landscape = self.memory.get_attention_landscape(query_embedding)

        # 6. Generate answer using GenerativeRAGSynthesizer
        t_synth_start = time.perf_counter()
        grounded_answer = self.answer_generator.synthesize(
            query=query_text,
            evidence=dedup_evidence_collection,
            style=active_style,
            provider_override=llm_provider_override,
            conversation_history=conversation_history
        )
        synth_time_ms = max(0.1, (time.perf_counter() - t_synth_start) * 1000)

        # 7. Convert for backward-compatibility with legacy consumers
        legacy_retrieved: List[RetrievalResult] = []
        for i, c in enumerate(dedup_evidence_collection.chunks):
            meta = c.metadata or {}
            legacy_retrieved.append(
                RetrievalResult(
                    chunk_id=c.chunk_id,
                    text=c.text,
                    source=c.source,
                    chunk_index=int(meta.get("chunk_index", i)),
                    similarity_score=float(c.similarity_score),
                    rank=c.rank or (i + 1),
                    metadata=meta,
                    hopfield_attention=float(c.hopfield_attention)
                )
            )

        legacy_answer = GeneratedAnswer(
            query=raw_query,
            answer=grounded_answer.answer,
            confidence=grounded_answer.confidence,
            evidence_chunks=legacy_retrieved,
            highlighted_passages=[
                {"text": c.text, "chunk_id": c.chunk_id, "relevance": c.similarity_score}
                for c in dedup_evidence_collection.chunks
            ],
            sources=dedup_evidence_collection.sources,
            answer_type="grounded_rag",
            generation_stats={
                "provider": grounded_answer.provider_used,
                "model": grounded_answer.model_name,
                "time_ms": grounded_answer.generation_time_ms,
                "abstained": grounded_answer.abstained,
                "abstention_reason": grounded_answer.abstention_reason,
            }
        )

        total_time = max(0.1, (time.perf_counter() - t0) * 1000)
        logger.info(f"Query completed in {total_time:.1f}ms with {len(dedup_evidence_collection.chunks)} unique chunks (web_used={coord_res.web_search_performed}, cached_used={coord_res.cached_knowledge_used}, abstained={grounded_answer.abstained}).")

        synth_stages = (grounded_answer.metadata.get("stage_latencies_ms") or {}) if grounded_answer else {}
        stage_latencies = {
            "embed_ms": round(embed_time_ms, 2),
            "retrieval_ms": round(retrieval_time_ms, 2),
            "dedup_ms": round(dedup_time_ms, 2),
            "synthesis_ms": round(synth_time_ms, 2),
            "total_ms": round(total_time, 2),
            **synth_stages
        }

        return QueryResult(
            query=raw_query,
            processed_query=processed,
            retrieved_chunks=legacy_retrieved,
            answer=legacy_answer,
            retrieval_time_ms=round(retrieval_time_ms, 2),
            total_time_ms=round(total_time, 2),
            attention_landscape=landscape,
            associative_result=associative_res,
            grounded_answer=grounded_answer,
            evidence_collection=dedup_evidence_collection,
            stage_latencies_ms=stage_latencies,
            deduplication_diagnostics=dedup_diagnostics,
            web_search_used=coord_res.web_search_performed,
            cached_knowledge_used=coord_res.cached_knowledge_used,
            format_detected=detected_fmt,
            format_requested=answer_format or answer_style,
            sufficiency_decision=coord_res.sufficiency_decision.__dict__ if coord_res.sufficiency_decision else None,
            web_sources=coord_res.web_sources
        )

    def get_stats(self) -> Dict:
        """Get pipeline statistics."""
        stats = {
            "pipeline": {
                "n_documents": self.n_documents,
                "n_chunks": self.n_chunks,
                "is_ready": self.is_ready,
                "embedding_model": self.config.embedding_model,
            },
            "hopfield_network": self.memory.get_network_stats(),
            "embedding_cache": self.embedder.cache_stats(),
            "config": self.config.to_dict(),
        }
        return stats

    def update_hopfield_beta(self, new_beta: float):
        """Dynamically update Hopfield temperature parameter."""
        self.memory.beta = new_beta

    def clear(self):
        """Clear all documents and reset the pipeline."""
        self._all_chunks = []
        self._documents = {}
        self._is_ready = False
        self.memory = PyTorchHopfieldMemory(
            beta=self.config.hopfield_beta,
            n_iter=self.config.hopfield_n_iter,
            similarity_threshold=self.config.similarity_threshold,
            energy_type=getattr(self.config, 'energy_type', 'softmax'),
            use_projection=self.config.use_projection
        )
        if self.config.use_projection and self.config.projection_path:
            import torch
            try:
                self.memory.retriever.projection.load_state_dict(torch.load(self.config.projection_path, map_location=self.memory.device, weights_only=True))
                print(f"Loaded trained projection layer from {self.config.projection_path}")
            except Exception as e:
                print(f"Could not load projection weights: {e}")
        self.embedder.clear_cache()

    def load_sample_documents(self) -> List[Dict]:
        """Load built-in sample documents for demo purposes."""
        samples = [
            (SAMPLE_DOC_AI, "AI_Overview.txt"),
            (SAMPLE_DOC_HOPFIELD, "Hopfield_Networks.txt"),
            (SAMPLE_DOC_RAG, "RAG_Systems.txt"),
        ]
        return self.add_documents(samples)


# ─────────────────────────────────────────────────────────────
# Sample documents for demonstration
# ─────────────────────────────────────────────────────────────

SAMPLE_DOC_AI = """
Artificial Intelligence: An Overview

Artificial intelligence (AI) refers to the simulation of human intelligence processes by
computer systems. These processes include learning (the acquisition of information and rules
for using the information), reasoning (using rules to reach approximate or definite
conclusions), and self-correction.

Machine Learning is a subset of AI that provides systems the ability to automatically
learn and improve from experience without being explicitly programmed. Machine learning
focuses on the development of computer programs that can access data and use it to learn
for themselves.

Deep Learning is part of a broader family of machine learning methods based on artificial
neural networks with representation learning. Learning can be supervised, semi-supervised
or unsupervised. Deep learning architectures such as deep neural networks, recurrent neural
networks, convolutional neural networks, and transformers have been applied to fields
including computer vision, natural language processing, and speech recognition.

Natural Language Processing (NLP) is a subfield of linguistics, computer science, and
artificial intelligence concerned with the interactions between computers and human
language, in particular how to program computers to process and analyze large amounts of
natural language data. NLP combines computational linguistics—rule-based modeling of
human language—with statistical, machine learning, and deep learning models.

Large Language Models (LLMs) are a type of AI system trained on vast amounts of text data.
They can generate text, answer questions, translate languages, and perform many other
language tasks. Examples include GPT-4, Claude, and LLaMA. These models have billions of
parameters and are trained using transformer architectures with attention mechanisms.

Transformers, introduced in "Attention is All You Need" (Vaswani et al., 2017), are
sequence-to-sequence models that use self-attention mechanisms. They have become the
dominant architecture for NLP tasks and have been extended to vision, audio, and
multimodal applications.

Reinforcement Learning from Human Feedback (RLHF) is a technique used to align language
models with human preferences. The model generates outputs, human raters evaluate them,
and the ratings are used to train a reward model. The language model is then fine-tuned
using reinforcement learning to maximize the reward.
"""

SAMPLE_DOC_HOPFIELD = """
Hopfield Networks and Associative Memory

Hopfield Networks are a form of recurrent neural network that serve as content-addressable
memory systems with binary threshold nodes. They were introduced by John Hopfield in 1982
and are used to model associative memory in biological neural systems.

Classical Hopfield Networks store binary patterns and can retrieve a stored pattern when
presented with a partial or noisy version. The network has a storage capacity of approximately
0.14 * N patterns, where N is the number of neurons. Retrieval works by minimizing an energy
function through iterative state updates.

The energy function of a Hopfield network is defined as:
E = -0.5 * Σᵢⱼ wᵢⱼ sᵢ sⱼ
where wᵢⱼ are the weights (connection strengths) and sᵢ, sⱼ are the neuron states.

Modern Hopfield Networks (MHN), introduced by Ramsauer et al. in 2021, extend classical
Hopfield networks to continuous states and exponentially increase storage capacity.
The storage capacity of modern Hopfield networks scales as exp(d/2) where d is the
dimensionality of the stored patterns — an exponential improvement over classical networks.

The energy function for modern Hopfield networks is:
E = -lse(β, X^T ξ) + 0.5 ξ^T ξ + (1/β) log N + 0.5 M²

where:
- X is the matrix of stored patterns (d × N)
- ξ is the current state vector
- β is the inverse temperature controlling retrieval sharpness
- lse is the log-sum-exp function
- N is the number of patterns, M is the largest norm

The update rule for modern Hopfield networks is:
ξ_new = X · softmax(β · X^T · ξ)

This is equivalent to a single attention head in the transformer attention mechanism,
establishing a deep connection between associative memory and attention.

A key insight is that the Hopfield update rule converges in ONE STEP for well-separated
patterns, making retrieval extremely efficient. The inverse temperature β controls the
trade-off between retrieval specificity and generalization.

Applications of Modern Hopfield Networks include:
1. Drug discovery and molecular property prediction
2. Document retrieval and question answering
3. Time series classification
4. Immune repertoire classification
5. Replacing traditional nearest-neighbor search in recommendation systems

The connection to transformers is particularly important: the self-attention mechanism in
transformers can be understood as performing associative memory retrieval in a modern
Hopfield network. Keys and values in attention correspond to stored patterns and their
associated outputs.
"""

SAMPLE_DOC_RAG = """
Retrieval-Augmented Generation (RAG) Systems

Retrieval-Augmented Generation (RAG) is a technique that combines the power of
large language models with information retrieval to produce more accurate, up-to-date,
and verifiable responses. RAG was introduced by Lewis et al. in 2020.

Traditional RAG Pipeline:
1. Document Ingestion: Documents are split into chunks, typically 256-512 tokens.
2. Embedding: Each chunk is encoded into a dense vector using an embedding model.
3. Vector Store: Embeddings are stored in a vector database (e.g., FAISS, Pinecone, Weaviate).
4. Retrieval: For a given query, the most similar chunks are retrieved using approximate
   nearest neighbor search (ANN).
5. Generation: Retrieved chunks are provided as context to an LLM which generates an answer.

Limitations of Traditional RAG:
- Chunking artifacts: Splitting documents can lose context that spans chunk boundaries.
- Fixed retrieval: Standard cosine similarity doesn't model query-document interactions.
- Sensitivity to query phrasing: Small changes in query wording can dramatically change results.
- No noise handling: Typos and ambiguous queries are not explicitly handled.
- Memory inefficiency: All embeddings must be stored and searched.

Advanced RAG Techniques:
- HyDE (Hypothetical Document Embeddings): Generate a hypothetical answer and use it for retrieval.
- Multi-Query Retrieval: Generate multiple query variants and fuse results.
- Parent-Child Chunking: Retrieve small chunks but return their larger parent for context.
- Contextual Compression: Extract only the relevant portion of a retrieved chunk.
- Re-ranking: Use a cross-encoder to re-rank initially retrieved chunks.
- Reciprocal Rank Fusion: Combine rankings from multiple retrieval systems.

Associative Memory as an Alternative to Traditional RAG:
Modern Hopfield Networks offer a compelling alternative to traditional vector similarity search.
Rather than finding the globally nearest neighbor, Hopfield retrieval performs energy
minimization that naturally handles:
- Noisy queries: The energy landscape smoothly handles perturbations
- Multi-pattern retrieval: Attention weights distributed across relevant patterns
- Context-aware similarity: The iterative update incorporates pattern relationships

Evaluation Metrics for RAG Systems:
- Faithfulness: Is the answer supported by the retrieved context?
- Answer Relevance: Does the answer address the query?
- Context Precision: Are the retrieved chunks relevant?
- Context Recall: Are all relevant chunks retrieved?
- RAGAS (RAG Assessment) is a popular evaluation framework.

Vector Databases for RAG:
- FAISS: Facebook AI Similarity Search, highly optimized for GPU
- Pinecone: Managed vector database with filtering
- Weaviate: Open-source with GraphQL interface
- Chroma: Lightweight, easy to use for prototyping
- Milvus: Distributed vector database for production scale
"""
