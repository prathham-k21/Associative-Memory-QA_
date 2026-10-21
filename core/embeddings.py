"""
Embeddings Module
=================
Handles document chunking, embedding generation, and vector storage
using sentence-transformers for dense semantic representations.
"""

import numpy as np
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import hashlib


@dataclass
class Chunk:
    """Represents a document chunk with metadata."""
    chunk_id: str
    text: str
    source: str
    chunk_index: int
    start_char: int
    end_char: int
    embedding: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source": self.source,
            "chunk_index": self.chunk_index,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "metadata": self.metadata,
        }


class DocumentChunker:
    """
    Splits documents into overlapping chunks for embedding.
    Supports sentence-aware splitting to avoid mid-sentence cuts.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_size: int = 50,
        split_by_sentence: bool = True,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.split_by_sentence = split_by_sentence

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex, preserving boundaries from headings."""
        sentence_pattern = re.compile(
            r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s'
        )
        sentences = sentence_pattern.split(text)
        return [s.strip() for s in sentences if s.strip()]

    def _create_chunk_id(self, source: str, index: int, text: str) -> str:
        content = f"{source}_{index}_{text[:50]}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def chunk_text(self, text: str, source: str = "document") -> List[Chunk]:
        """Split text into overlapping chunks with heading and paragraph boundary preservation."""
        orig_text = text
        text = re.sub(r'\r\n|\r', '\n', text)
        text = re.sub(r'\n\s*\n+', '. ', text)
        text = re.sub(r'(?<=[a-zA-Z0-9\)])\s*\n\s*(?=[A-Z0-9])', '. ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        if len(text) <= self.chunk_size:
            if len(text) < self.min_chunk_size:
                return []
            chunk_id = self._create_chunk_id(source, 0, text)
            return [Chunk(
                chunk_id=chunk_id,
                text=text,
                source=source,
                chunk_index=0,
                start_char=0,
                end_char=len(text),
            )]

        chunks = []
        if self.split_by_sentence:
            sentences = self._split_sentences(text)
            chunks = self._chunk_by_sentences(sentences, text, source)
        else:
            chunks = self._chunk_by_chars(text, source)

        return [c for c in chunks if len(c.text) >= self.min_chunk_size]

    def _chunk_by_sentences(
        self, sentences: List[str], original_text: str, source: str
    ) -> List[Chunk]:
        chunks = []
        current_sentences = []
        current_length = 0
        chunk_index = 0

        for sentence in sentences:
            sentence_len = len(sentence) + 1  # +1 for space

            # Separate institutional/course cover headers from body content
            is_sec_break = bool(re.search(r'^(?:unit\s+\d+|chapter\s+\d+|module\s+\d+|section\s+\d+|concept\s+of\b)', sentence.strip().lower()))
            has_inst_header = any(
                any(k in s.lower() for k in ["college of", "university", "department of", "course code", "autonomous", "syllabus"])
                for s in current_sentences
            )
            should_split = (current_length + sentence_len > self.chunk_size) or (is_sec_break and has_inst_header)

            if should_split and current_sentences:
                chunk_text = ' '.join(current_sentences)
                start = original_text.find(chunk_text[:30])
                chunk_id = self._create_chunk_id(source, chunk_index, chunk_text)

                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    source=source,
                    chunk_index=chunk_index,
                    start_char=max(0, start),
                    end_char=max(0, start) + len(chunk_text),
                ))
                chunk_index += 1

                # If cutting after institutional header, do not carry header into body chunk
                if is_sec_break and has_inst_header:
                    current_sentences = []
                    current_length = 0
                else:
                    # Overlap: keep last few sentences
                    overlap_sentences = []
                    overlap_len = 0
                    for s in reversed(current_sentences):
                        if overlap_len + len(s) < self.chunk_overlap:
                            overlap_sentences.insert(0, s)
                            overlap_len += len(s) + 1
                        else:
                            break
                    current_sentences = overlap_sentences
                    current_length = overlap_len

            current_sentences.append(sentence)
            current_length += sentence_len

        # Last chunk
        if current_sentences:
            chunk_text = ' '.join(current_sentences)
            if len(chunk_text) >= self.min_chunk_size:
                chunk_id = self._create_chunk_id(source, chunk_index, chunk_text)
                start = original_text.find(chunk_text[:30])
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    source=source,
                    chunk_index=chunk_index,
                    start_char=max(0, start),
                    end_char=max(0, start) + len(chunk_text),
                ))

        return chunks

    def _chunk_by_chars(self, text: str, source: str) -> List[Chunk]:
        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]
            chunk_id = self._create_chunk_id(source, chunk_index, chunk_text)

            chunks.append(Chunk(
                chunk_id=chunk_id,
                text=chunk_text,
                source=source,
                chunk_index=chunk_index,
                start_char=start,
                end_char=end,
            ))
            chunk_index += 1
            start += self.chunk_size - self.chunk_overlap

        return chunks


class EmbeddingEngine:
    """
    Generates dense vector embeddings using sentence-transformers.
    Supports batching and caching for efficiency.
    """

    SUPPORTED_MODELS = {
        "all-MiniLM-L6-v2": {"dim": 384, "speed": "fast", "quality": "good"},
        "all-mpnet-base-v2": {"dim": 768, "speed": "medium", "quality": "excellent"},
        "multi-qa-MiniLM-L6-cos-v1": {"dim": 384, "speed": "fast", "quality": "good"},
        "paraphrase-multilingual-MiniLM-L12-v2": {"dim": 384, "speed": "medium", "quality": "multilingual"},
    }

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 32):
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self._cache: Dict[str, np.ndarray] = {}

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def embedding_dim(self) -> int:
        return self.SUPPORTED_MODELS.get(self.model_name, {}).get("dim", 384)

    def embed_texts(
        self,
        texts: List[str],
        show_progress: bool = False,
        normalize: bool = True,
    ) -> np.ndarray:
        """Embed a list of texts, returning a 2D numpy array."""
        # Check cache
        uncached_indices = []
        uncached_texts = []
        results = [None] * len(texts)

        for i, text in enumerate(texts):
            cache_key = hashlib.md5(text.encode()).hexdigest()
            if cache_key in self._cache:
                results[i] = self._cache[cache_key]
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Embed uncached texts
        if uncached_texts:
            embeddings = self.model.encode(
                uncached_texts,
                batch_size=self.batch_size,
                show_progress_bar=show_progress,
                normalize_embeddings=normalize,
                convert_to_numpy=True,
            )
            for i, (orig_idx, text) in enumerate(zip(uncached_indices, uncached_texts)):
                cache_key = hashlib.md5(text.encode()).hexdigest()
                self._cache[cache_key] = embeddings[i]
                results[orig_idx] = embeddings[i]

        if not results:
            return np.empty((0, self.embedding_dim))
        return np.vstack(results)

    def embed_chunks(
        self,
        chunks: List[Chunk],
        show_progress: bool = True,
    ) -> List[Chunk]:
        """Embed chunks in-place, setting the embedding attribute."""
        texts = [c.text for c in chunks]
        embeddings = self.embed_texts(texts, show_progress=show_progress)

        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb

        return chunks

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string."""
        return self.embed_texts([query], normalize=True)[0]

    def clear_cache(self):
        self._cache.clear()

    def cache_stats(self) -> Dict:
        return {"cached_embeddings": len(self._cache)}
