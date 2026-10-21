"""
Query Handling Module
=====================
Robust query processing with:
  - Query cleaning & normalization
  - Query expansion for noisy/vague inputs
  - Multi-variant generation for fusion retrieval
  - Noise detection and handling
"""

import re
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field


from collections import Counter
from typing import List, Dict, Tuple, Optional, Set, Union


@dataclass
class ProcessedQuery:
    """Result of query processing pipeline."""
    original: str
    cleaned: str
    variants: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    is_noisy: bool = False
    noise_score: float = 0.0
    query_type: str = "factual"  # factual | conceptual | procedural | comparative
    metadata: Dict = field(default_factory=dict)


class QueryProcessor:
    """
    Multi-stage query processing pipeline designed to handle:
    - Typos and spelling errors
    - Vague or underspecified queries
    - Long verbose questions
    - Technical jargon
    - Noisy real-world inputs
    """

    # Common stopwords to strip from keyword extraction
    STOPWORDS = {
        "a", "an", "the", "is", "it", "in", "on", "at", "to", "for",
        "of", "and", "or", "but", "not", "with", "this", "that", "be",
        "are", "was", "were", "been", "have", "has", "had", "do", "does",
        "did", "will", "would", "could", "should", "may", "might", "can",
        "what", "when", "where", "who", "why", "how", "which", "about",
        "from", "into", "through", "during", "before", "after", "above",
        "below", "between", "each", "than", "its", "their", "our", "your",
        "my", "his", "her", "we", "they", "you", "i", "me", "him", "us",
        "just", "also", "some", "any", "all", "more", "much", "many",
        "most", "other", "then", "so", "such", "no", "only", "same",
        "both", "while", "use", "get", "make", "go", "see", "tell",
        "know", "think", "come", "want", "give", "take", "say", "look",
        "even", "still", "need", "feel", "try", "find", "like"
    }

    # Query type signals
    QUERY_TYPE_PATTERNS = {
        "factual": [r"\bwhat\s+is\b", r"\bwho\s+is\b", r"\bwhen\s+did\b",
                    r"\bwhere\s+is\b", r"\bdefine\b", r"\bmeaning\s+of\b"],
        "procedural": [r"\bhow\s+to\b", r"\bhow\s+do\b", r"\bsteps\s+to\b",
                       r"\bprocess\s+of\b", r"\bprocedure\b"],
        "comparative": [r"\bvs\b", r"\bversus\b", r"\bdifference\s+between\b",
                        r"\bcompare\b", r"\bbetter\b", r"\badvantage\b"],
        "conceptual": [r"\bexplain\b", r"\bdescribe\b", r"\bwhy\b",
                       r"\bconcept\b", r"\btheory\b", r"\bprinciple\b"],
    }

    # Common abbreviations to expand
    ABBREVIATIONS = {
        "ml": "machine learning",
        "ai": "artificial intelligence",
        "nlp": "natural language processing",
        "dl": "deep learning",
        "nn": "neural network",
        "rag": "retrieval augmented generation",
        "llm": "large language model",
        "mhn": "modern hopfield network",
        "qa": "question answering",
        "ir": "information retrieval",
        "bert": "bidirectional encoder representations from transformers",
        "gpt": "generative pre-trained transformer",
        "api": "application programming interface",
    }

    def __init__(self, expand_abbreviations: bool = True):
        self.expand_abbreviations = expand_abbreviations
        self.vocabulary: Counter = Counter()

    def update_vocabulary(self, text_or_words: Union[str, List[str]]) -> None:
        """Register domain tokens from ingested documents for spelling correction."""
        if isinstance(text_or_words, str):
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text_or_words.lower())
        else:
            words = [w.lower() for w in text_or_words if len(w) >= 3]
        self.vocabulary.update(words)

    def _detect_noise(self, text: str) -> Tuple[bool, float]:
        """
        Detect noise in query: typos, fragments, excessive brevity.
        Returns (is_noisy, noise_score in [0,1])
        """
        score = 0.0
        tokens = text.split()

        # Too short
        if len(tokens) < 2:
            score += 0.4
        elif len(tokens) < 4:
            score += 0.15

        # Very long (may be copy-pasted, noisy)
        if len(tokens) > 60:
            score += 0.2

        # All caps
        if text.isupper() and len(text) > 5:
            score += 0.2

        # Missing spaces between words (runon)
        avg_word_len = np.mean([len(t) for t in tokens]) if tokens else 0
        if avg_word_len > 15:
            score += 0.3

        # Lots of punctuation or special characters
        special_char_ratio = len(re.findall(r'[^a-zA-Z0-9\s]', text)) / max(len(text), 1)
        if special_char_ratio > 0.2:
            score += 0.2

        # Repeated characters (typoooo)
        repeated = re.findall(r'(.)\1{3,}', text)
        if repeated:
            score += 0.3

        score = min(score, 1.0)
        return score > 0.3, score

    def _clean_text(self, text: str) -> str:
        """Clean and normalize query text."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Remove leading/trailing punctuation (but keep question mark context)
        text = re.sub(r'^[^\w\s]+', '', text)
        text = re.sub(r'[^\w\s?!.]+$', '', text)

        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace("'", "'").replace("'", "'")

        # Remove repeated punctuation
        text = re.sub(r'[!?]{2,}', '?', text)
        text = re.sub(r'\.{3,}', '...', text)

        return text.strip()

    def _expand_abbreviations(self, text: str) -> str:
        """Expand known abbreviations in the query."""
        words = text.split()
        expanded = []
        for word in words:
            lower = word.lower().rstrip('s')  # handle plural
            if lower in self.ABBREVIATIONS:
                expanded.append(self.ABBREVIATIONS[lower])
            else:
                expanded.append(word)
        return ' '.join(expanded)

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract meaningful keywords from query."""
        tokens = re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())
        keywords = [t for t in tokens if t not in self.STOPWORDS]

        # Score by length (longer = more specific)
        keywords = sorted(set(keywords), key=lambda x: -len(x))
        return keywords[:10]

    def _detect_query_type(self, text: str) -> str:
        """Classify query type based on linguistic patterns."""
        text_lower = text.lower()
        for qtype, patterns in self.QUERY_TYPE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return qtype
        return "factual"

    def _generate_variants(
        self, query: str, keywords: List[str], query_type: str
    ) -> List[str]:
        """
        Generate query variants for robust multi-query retrieval.
        Handles vague/noisy queries by reformulating.
        """
        variants = [query]  # original always included

        # Keyword-only variant (robust to stopwords/noise)
        if keywords:
            kw_variant = ' '.join(keywords[:6])
            if kw_variant != query.lower() and len(kw_variant) > 5:
                variants.append(kw_variant)

        # Type-specific reformulations
        if query_type == "factual":
            # Add "explain" prefix for broader semantic coverage
            if not query.lower().startswith("explain"):
                variants.append(f"explain {query}")

        elif query_type == "procedural":
            # Add "steps" for more grounded retrieval
            clean = re.sub(r'^how\s+to\s+', '', query.lower())
            variants.append(f"steps for {clean}")
            variants.append(f"process of {clean}")

        elif query_type == "comparative":
            # Extract the compared entities
            parts = re.split(r'\s+vs\.?\s+|\s+versus\s+|\s+compared\s+to\s+',
                             query, flags=re.IGNORECASE)
            if len(parts) == 2:
                variants.append(f"advantages of {parts[0].strip()}")
                variants.append(f"advantages of {parts[1].strip()}")

        elif query_type == "conceptual":
            clean = re.sub(r'^(explain|describe|what\s+is)\s+', '', query.lower())
            variants.append(f"definition of {clean}")
            variants.append(f"{clean} overview")

        # Deduplication while preserving order
        seen = set()
        unique_variants = []
        for v in variants:
            v_norm = v.strip().lower()
            if v_norm not in seen and len(v_norm) > 3:
                seen.add(v_norm)
                unique_variants.append(v.strip())

        return unique_variants[:5]  # cap at 5 variants

    @classmethod
    def contextualize_query(
        cls,
        raw_query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Multi-turn coreference resolution for query embedding and retrieval.
        Substitutes anaphoric pronouns ('it', 'its', 'they', 'their', 'this system', 'the system')
        with the primary entity / topic established in prior conversation turns.
        """
        if not conversation_history:
            return raw_query

        q_lower = raw_query.lower().strip()
        ANAPHORA_PATTERNS = [
            (r'\b(?:where\s+is|where\s+are)\s+it\b', 'where is {entity}'),
            (r'\b(?:how\s+does|how\s+do)\s+it\b', 'how does {entity}'),
            (r'\b(?:what\s+is|what\s+are)\s+it\b', 'what is {entity}'),
            (r'\b(?:why\s+is|why\s+does)\s+it\b', 'why does {entity}'),
            (r'\b(?:what\s+are)\s+(?:its|their)\b', 'what are {entity}\'s'),
            (r'\b(?:explain|describe)\s+its\b', 'explain {entity}\'s'),
            (r'\b(?:explain|describe)\s+it\b', 'explain {entity}'),
            (r'\b(?:can|does|do|will|is|are)\s+it\b', r'\g<0>'.replace('it', '{entity}')),
            (r'\b(?:can|does|do|will|is|are)\s+they\b', r'\g<0>'.replace('they', '{entity}')),
            (r'\babout\s+it\b', 'about {entity}'),
            (r'\bfor\s+it\b', 'for {entity}'),
            (r'\bwith\s+it\b', 'with {entity}'),
            (r'\b(?:in|on|to)\s+it\b', r'\g<0>'.replace('it', '{entity}')),
            (r'\bthis\s+system\b', '{entity}'),
            (r'\bthe\s+system\b', '{entity}'),
        ]

        needs_resolution = False
        for pat, _ in ANAPHORA_PATTERNS:
            if re.search(pat, q_lower):
                needs_resolution = True
                break

        if not needs_resolution and not re.search(r'\b(it|its|their|they)\b', q_lower):
            return raw_query

        # Extract target entity from recent turns (from newest to oldest)
        recent_entity = None
        for turn in reversed(conversation_history):
            role = turn.get("role")
            content = turn.get("content", "").strip()
            if not content:
                continue

            if role == "user":
                # Look for target entity in user query
                m = re.search(r'(?:what\s+is|what\s+are|define|explain|where\s+is|how\s+does)\s+(?:a\s+|an\s+|the\s+)?([a-zA-Z0-9_\-\s]{2,40}?)(?:\?|$|\.|\,)', content, re.I)
                if m:
                    cand = m.group(1).strip()
                    cand = re.sub(r'\s+(?:used|applied|deployed|for|in|mean|means)$', '', cand, flags=re.I).strip()
                    if cand.lower() not in cls.STOPWORDS:
                        recent_entity = cand
                        break
            elif role == "assistant":
                # Check for strong leading entity in assistant response (e.g. "YARN is used in...", "Cybersecurity is...")
                m = re.match(r'^(?:[A-Za-z0-9_\-\s]+?:\s+)?([A-Za-z0-9_\-]+)\s+(?:is|was|refers|functions|operates)\b', content)
                if m:
                    cand = m.group(1).strip()
                    if cand.lower() not in cls.STOPWORDS and len(cand) > 2:
                        recent_entity = cand
                        break

        if not recent_entity:
            return raw_query

        # Substitute anaphora with identified entity
        contextualized = raw_query
        for pat, template in ANAPHORA_PATTERNS:
            replacement = template.format(entity=recent_entity)
            contextualized = re.sub(pat, replacement, contextualized, flags=re.I)

        # Handle remaining standalone " it " / " its " if query is very short (e.g. "Where is it?")
        if len(raw_query.split()) <= 6:
            contextualized = re.sub(r'\bits\b', f"{recent_entity}'s", contextualized, flags=re.I)
            contextualized = re.sub(r'\b(it|they)\b', recent_entity, contextualized, flags=re.I)

        return contextualized.strip()

    @staticmethod
    def _edit_distance(s1: str, s2: str) -> int:
        """Compute Levenshtein edit distance with length difference pruning."""
        if abs(len(s1) - len(s2)) > 2:
            return 99
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
        return dp[m][n]

    def _correct_typos(self, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Correct spelling mistakes and typos by matching against corpus vocabulary.
        E.g., 'what is bid data' -> 'what is big data', 'nosql databse' -> 'nosql database'.
        """
        if not self.vocabulary:
            return text, {}

        protected = self.STOPWORDS | {
            "define", "explain", "meaning", "overview", "detail", "details",
            "compare", "versus", "difference", "between", "how", "what",
            "where", "when", "why", "who", "which", "marks", "mark", "format"
        }

        tokens = text.split()
        corrected_tokens = []
        typos_fixed = {}

        for token in tokens:
            m = re.match(r'^([^\w]*)([\w\-]+)([^\w]*)$', token)
            if not m:
                corrected_tokens.append(token)
                continue

            prefix, word, suffix = m.groups()
            word_low = word.lower()

            if len(word_low) < 3 or word_low in protected or self.vocabulary.get(word_low, 0) >= 5:
                corrected_tokens.append(token)
                continue

            # Search vocabulary for nearest high-frequency term
            max_allowed = 1 if len(word_low) <= 4 else 2
            best_cand = None
            min_dist = 99
            max_freq = 0

            for term, freq in self.vocabulary.items():
                if freq < 1:
                    continue
                d = self._edit_distance(word_low, term)
                if d <= max_allowed:
                    if d < min_dist or (d == min_dist and freq > max_freq):
                        min_dist = d
                        max_freq = freq
                        best_cand = term

            current_freq = self.vocabulary.get(word_low, 0)
            if best_cand and (current_freq == 0 or max_freq >= max(3, current_freq * 3)):
                # Preserve capitalization
                replacement = best_cand.capitalize() if word.istitle() else best_cand
                typos_fixed[word] = replacement
                corrected_tokens.append(f"{prefix}{replacement}{suffix}")
            else:
                corrected_tokens.append(token)

        corrected_text = " ".join(corrected_tokens)
        return corrected_text, typos_fixed

    def process(
        self,
        raw_query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> ProcessedQuery:
        """
        Full query processing pipeline.

        Steps:
        1. Contextualize query with conversation history (coreference resolution)
        2. Clean text
        3. Detect noise
        4. Expand abbreviations
        5. Vocabulary-aware typo and spelling correction
        6. Extract keywords
        7. Detect query type
        8. Generate variants

        Returns:
            ProcessedQuery with all processed fields populated.
        """
        contextualized = self.contextualize_query(raw_query, conversation_history)
        cleaned = self._clean_text(contextualized)
        is_noisy, noise_score = self._detect_noise(cleaned)

        if self.expand_abbreviations:
            cleaned = self._expand_abbreviations(cleaned)

        # Apply corpus-aware typo correction
        corrected_query, typos_fixed = self._correct_typos(cleaned)

        # Use corrected query as primary representation if typos were fixed
        query_for_retrieval = corrected_query if typos_fixed else cleaned
        if typos_fixed:
            is_noisy = True
            noise_score = max(noise_score, 0.4)

        keywords = self._extract_keywords(query_for_retrieval)
        query_type = self._detect_query_type(query_for_retrieval)
        variants = self._generate_variants(query_for_retrieval, keywords, query_type)

        # Include original cleaned form in variants if corrected
        if typos_fixed and cleaned.lower() not in [v.lower() for v in variants]:
            variants.append(cleaned)

        return ProcessedQuery(
            original=raw_query,
            cleaned=query_for_retrieval,
            variants=variants,
            keywords=keywords,
            is_noisy=is_noisy,
            noise_score=noise_score,
            query_type=query_type,
            metadata={
                "n_variants": len(variants),
                "n_keywords": len(keywords),
                "contextualized_query": contextualized if contextualized != raw_query else None,
                "typos_fixed": typos_fixed,
                "corrected_query": corrected_query if typos_fixed else None,
            }
        )

    def batch_process(self, queries: List[str]) -> List[ProcessedQuery]:
        return [self.process(q) for q in queries]

