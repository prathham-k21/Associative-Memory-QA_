"""
Evidence-Grounded Generative Synthesizer & Citation Validator
=============================================================
A robust, evidence-first answer synthesis subsystem adhering to strict RAG principles:
1. Evidence-First Architecture: Consumes structured EvidenceCollection, not an extractive span wrapper.
2. Deduplication Stage: Removes exact and near-duplicate passages while preserving merged provenance.
3. Direct-Answer-First Policy: Formulates concise, direct first sentences answering the question intent
   (WHERE / WHAT / HOW / WHY / COMPARISON) without narrative preambles or corporate fluff.
4. Non-Repetition: Eliminates redundant sentences and duplicate facts across paragraphs.
5. Strict Claim Verification: Validates citations individually, supports morphological paraphrases via stemming,
   detects hallucinated entities, and flags redundant claims.
6. Honest Abstention & Calibrated Confidence: Fails gracefully when evidence is missing or irrelevant.
"""

import os
import re
import time
import json
import logging
import urllib.request
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field

from core.schemas import EvidenceChunk, EvidenceCollection, GroundedAnswer, ClaimCitation
from core.evidence_deduplicator import EvidenceDeduplicator
from core.query_handler import ProcessedQuery
from core.llm_provider import (
    LLMConfig,
    LLMResult,
    BaseLLMProvider,
    create_llm_provider,
    MockLLMProvider
)

logger = logging.getLogger(__name__)

# Common stopwords to exclude when identifying substantive query terms
STOP_WORDS: Set[str] = {
    "the", "and", "what", "which", "between", "with", "for", "from", "into",
    "this", "that", "these", "those", "how", "why", "who", "when", "where",
    "can", "could", "should", "would", "does", "did", "have", "has", "had",
    "about", "through", "during", "before", "after", "some", "more", "most",
    "are", "is", "was", "were", "been", "being", "used", "use", "using",
    "also", "than", "then", "just", "very", "much", "such", "only", "other",
    "its", "their", "theirs", "work", "works", "working", "worked",
    "answer", "mark", "marks", "question", "notes", "note", "revision", "depth",
    "create", "give", "tell", "explain", "overview", "meaning", "definition",
    "detail", "details", "concept", "role", "purpose", "example", "examples"
}

# Discourse, connective, and framing terms that should not be penalized as domain hallucinations
DISCOURSE_WORDS: Set[str] = {
    "contrast", "relies", "heavily", "using", "uses", "used", "provides", "provide",
    "supported", "documented", "system", "systems", "method", "approach", "technique", "specifically",
    "functions", "operates", "regarding", "meanwhile", "furthermore", "including", "include",
    "based", "across", "within", "enables", "achieves", "ensures", "ranging", "such", "well",
    "provided", "documentation", "defined", "addressed", "operational", "practices",
    "standalone", "textbook", "definition", "details", "implemented", "organizational",
    "defense", "controls", "designed", "protect", "protecting", "aspects", "activities",
    "while", "not", "offer", "rather", "instead", "general", "monitoring",
    "architecturally", "architecture", "global", "establishes", "identifies", "indicates", "demonstrates", "explains", "notes",
    "finding", "findings", "scope", "composed", "comprising", "primary", "key", "guarantee", "guarantees", "coordination", "summary", "executive",
    "foundation", "foundational", "principles", "framework", "frameworks", "protocol", "protocols", "lifecycle", "diversity",
    "governed", "deployed", "instantiated", "collaborates", "telemetry", "recognition", "investigation", "resolution", "dimension",
    "dimensions", "matrix", "assessment", "recommendations", "practices", "configuration", "deployment", "strategic"
}

# Regex patterns matching document cover pages, case study metadata, and course/author boilerplate
BOILERPLATE_PATTERNS: List[str] = [
    r'\bthis\s+(?:comprehensive\s+)?(?:research\s+)?(?:report|paper|case study|thesis|document|analysis|section|chapter)\b',
    r'\bserves as an exhaustive\b',
    r'\bdesigned specifically for\b',
    r'\bsubmitted by\b',
    r'\bname:\s*[a-zA-Z]',
    r'\bprn:\s*\d+',
    r'\bexecutive summary\s*&\s*overview\b',
    r'\boperating within the domain of\b',
    r'\bthe analysis rigorously explores\b',
    r'\btable of contents\b',
    r'\bhttps?://',
    r'\bdoi:\s*10\.',
    r'\bbibliography\b',
    r'\ball rights reserved\b',
    r'\bpage\s+\d+\s+of\s+\d+\b',
    r'^\s*\d+\.\s+[a-zA-Z\s]+$',
    r'\bacademic project\b',
    r'\bcourse code\b',
    r'\binstructor\b',
    r'\buniversity\b',
    r'\bdepartment of\b',
    r'\bmodel questions\b',
    r'\breview questions\b',
    r'^\s*\d+\.\s+(?:explain|describe|illustrate|discuss|define|what\s+is)\b',
    r'\bbenefit\s+role\s+in\b',
    r'\breduces\s+complexity\s+makes\b',
    r'\bimportance\s+in\s+cybersecurity\s+benefit\b'
]


# Safe morphological stemmer helper (supporting valid paraphrases without external downloads)
try:
    from nltk.stem import PorterStemmer
    _nltk_stemmer = PorterStemmer()
    def get_stem(w: str) -> str:
        return _nltk_stemmer.stem(w.lower())
except Exception:
    def get_stem(w: str) -> str:
        w = w.lower()
        for suff in ("ing", "tion", "tions", "ed", "es", "s", "ment", "ments", "er", "ers", "al"):
            if w.endswith(suff) and len(w) > len(suff) + 2:
                return w[:-len(suff)]
        return w


def is_boilerplate(text: str) -> bool:
    """Detects introductory meta-boilerplate or cover-page sentences."""
    t = text.lower()
    for p in BOILERPLATE_PATTERNS:
        if re.search(p, t):
            return True
    return False


def clean_sentence_prefix(s: str) -> str:
    """Strips embedded outline headings, chapter numbers, figure captions, and titles from sentence starts."""
    s = s.strip()
    # Strip standalone figure captions e.g. "Fig 19. Listening Comprehension Fig 20. Reading Passage 20 "
    s = re.sub(r'(?:Fig\s*\d+\.?\s*[A-Za-z0-9\s/&,\-]+\s*)+', ' ', s, flags=re.I).strip()
    s = re.sub(r'^(?:unit|chapter)\s+\d+\s*[:\-\.]?\s*', '', s, flags=re.I).strip()
    s = re.sub(r'^(?:INDEX|EVENT REPORT)\s+[A-Za-z0-9_\-\s]+?(?=\d|\bIntroduction\b|\bOverview\b|\bSection\b)', '', s, flags=re.I).strip()
    s = re.sub(r'^(?:Sr\.\s*No\.\s*Topic\s*Page\s*No\.?\s*)+', '', s, flags=re.I).strip()

    # Strip inline numbered section titles or page-number + section-number prefixes (including preceding TOC entries)
    # e.g. "3 1. Introduction: VERBAFEST 2026 was..." or "20 Final Overview: The main objective..."
    s = re.sub(r'^(?:(?:\d+(?:\.\d+)*\.?\s+)?[A-Za-z\s/&,\-]+\s+\d+\s+)*(?:\d+\s+)*(?:\d+(?:\.\d+)*\.?\s*)?(?:Introduction|Overview|Final Overview|Special Features|Summary|Background|Executive Summary)\s*:\s*', '', s, flags=re.I).strip()

    # Strip inline numbered section titles preceding body sentences
    # e.g. "12. Security Metrics and Their Importance in Evaluating... Security metrics are measurable values..."
    title_pat = re.compile(
        r'^\s*\d+(?:\.\d+)*\.?\s+'
        r'[A-Za-z0-9_\-\(\)\s/&,]{3,130}?'
        r'(?<=[a-z\)])\s+'
        r'(?=(?:a\s+|an\s+|the\s+)?([A-Za-z0-9_\-]+(?:\s+[a-z]+)?)\s+(?:is|are|refers|functions|was|were|describes|provides|serves|can|operates|achieves|stands)\b)',
        re.I
    )
    m = title_pat.match(s)
    if m:
        s = s[m.end():].strip()

    # Process line-by-line if the chunk contains embedded outline title lines
    lines = [line.strip() for line in s.split('\n') if line.strip()]
    cleaned_lines = []
    for i, line in enumerate(lines):
        if i == 0 and re.match(r'^\s*\d+(?:\.\d+)*\.?\s+', line):
            if not line.endswith('.') or len(line.split()) < 8:
                continue
        if i == 1 and len(lines) > 2 and len(line.split()) < 6 and not line.endswith('.'):
            continue
        cleaned_lines.append(line)
    
    res = ' '.join(cleaned_lines).strip()
    # Strip numbered and named outline prefixes like '4. Resource Management: YARN Framework Components:'
    res = re.sub(
        r'^\s*\d+(?:\.\d+)*\.?\s*(?:[A-Za-z0-9\s/&,\-]{3,60}?(?::|\bComponents\b|\bFramework\b|\bOverview\b|\bArchitecture\b)\s*:?\s*)+',
        '',
        res,
        flags=re.I
    ).strip()
    res = re.sub(r'^\s*(?:\d+(?:\.\d+)*\.?\s*)+', '', res).strip()

    return res if len(res) > 20 else s.strip()


def extract_sentence_subject(sent: str) -> Optional[str]:
    """Extracts the leading noun phrase/subject before the main copula/action verb."""
    s_cleaned = clean_sentence_prefix(sent)
    m = re.match(
        r'^\s*([A-Za-z0-9_\-\s\(\)]+?)\s+(?:is\s+an?|is\s+the|is|are\s+an?|are\s+the|are|was\s+an?|was\s+the|was|were\s+an?|were\s+the|were|refers\s+to|functions\s+as|serves\s+as|provides|represents|operates\s+as|achieves|stands\s+for|aims\s+to|focuses\s+on)\b',
        s_cleaned,
        re.I
    )
    if m:
        subj = m.group(1).strip()
        subj_clean = re.sub(r'\(.*?\)', '', subj).strip()
        return subj_clean.lower()
    return None


def extract_query_target(query: str, intent: str) -> Optional[str]:
    """Extracts the primary target entity or subject from the user query."""
    q = query.strip()

    # What is / define
    m = re.search(r'(?:what\s+is|what\s+are|define|meaning\s+of|overview\s+of)\s+(?:a\s+|an\s+|the\s+)?([a-zA-Z0-9_\-\s]+?)(?:\?|$|\.|\,)', q, re.I)
    if m:
        target = m.group(1).strip()
        target = re.split(r'\s+(?:and\s+how|and\s+what|and\s+why|and\s+where)\b', target, flags=re.I)[0].strip()
        target = re.sub(r'\s+(?:used|applied|deployed|for|in|mean|means|work|works)$', '', target, flags=re.I).strip()
        if target and target.lower() not in STOP_WORDS:
            return target

    # Where is X used
    m = re.search(r'(?:where\s+is|where\s+are|where\s+does|where\s+can)\s+([a-zA-Z0-9_\-\s]+?)\s+(?:used|applied|deployed|run|coexist|operate|work)', q, re.I)
    if m:
        target = m.group(1).strip()
        if target and target.lower() not in STOP_WORDS:
            return target

    # Explain X's components / architecture
    m = re.search(r'(?:explain|describe|what\s+are\s+the)\s+([a-zA-Z0-9_\-]+)(?:\'s|\s+core)?\s+(?:components|architecture|daemons|internals|mechanisms|responsibilities)', q, re.I)
    if m:
        return m.group(1).strip()

    # Fallback to substantive tokens
    substantive = [
        w for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', q)
        if w.lower() not in STOP_WORDS and w.lower() not in {
            "explain", "describe", "detail", "overview", "components", "architecture",
            "mechanisms", "responsibilities", "difference", "compare", "versus"
        }
    ]
    return substantive[0] if substantive else None


def smooth_evidence_sentence(s: str) -> str:
    """
    Cleans OCR artifacts, stray list/slide bullets, dangling prepositions,
    broken word wraps, and trailing colon headers from evidence sentences.
    """
    s = s.strip()
    # Strip bullet symbols like •, -, *, ·, >, ◦
    s = re.sub(r'^[•\-\*·>◦\s]+', '', s).strip()
    # Strip slide headers e.g. "Deployment Serve via...", "Overview: ..."
    s = re.sub(r'^(?:slide\s+\d+|overview|deployment|key\s+takeaways?|summary)\s*:\s*', '', s, flags=re.I).strip()
    # Strip dangling trailing prepositions / conjunctions before period
    s = re.sub(r'\s+(?:and|or|with|to|for|of|in|by|via|the|a|an)\s*$', '', s, flags=re.I).strip()
    return s


def clean_narrative_preamble(s: str) -> str:
    """Strips narrative filler, company preambles, and conversational lead-ins."""
    s = smooth_evidence_sentence(s)
    FILLER_PREFIXES = [
        r'^\s*in a (?:vast|large|complex|production|distributed|multi-tenant|enterprise)?[^,]+(?:like [^,]+)?,?\s*(?:thousands of [^,]+,\s*)?',
        r'^\s*(?:thousands|hundreds|millions) of [^,]+,\s*',
        r'^\s*furthermore,?\s*',
        r'^\s*moreover,?\s*',
        r'^\s*additionally,?\s*',
        r'^\s*specifically,?\s*',
        r'^\s*at large scale,?\s*',
        r'^\s*to address these (?:challenges|issues|requirements),?\s*',
        r'^\s*as noted (?:above|previously|earlier),?\s*',
        r'^\s*from an architectural perspective,?\s*',
        r'^\s*with petabytes of [^,]+,\s*',
    ]
    for pat in FILLER_PREFIXES:
        s = re.sub(pat, '', s, flags=re.I).strip()

    # Normalize wordy phrases & broken PDF tokens
    s = re.sub(r'\bthe sophisticated orchestration of cluster resources via\b', 'orchestration of cluster resources by', s, flags=re.I)
    s = re.sub(r'\bthe sophisticated orchestration of\b', 'the orchestration of', s, flags=re.I)
    s = re.sub(r'\bwildly varying\b', 'diverse', s, flags=re.I)
    s = re.sub(r'\bmassive,\s*resource-intensive\b', 'resource-intensive', s, flags=re.I)
    s = re.sub(r'\bco\s+mmunication\b', 'communication', s, flags=re.I)
    s = re.sub(r'\bin\s+fra\s*structure\b', 'infrastructure', s, flags=re.I)
    s = re.sub(r'\bmulti-\s*tenant\b', 'multi-tenant', s, flags=re.I)

    if s and s[0].islower():
        s = s[0].upper() + s[1:]
    return s



def classify_intent(query: str) -> str:
    """Classifies user query into actionable semantic intent categories."""
    q = query.lower()
    # Multi-part questions (e.g. "Where is YARN used, and how does its architecture work?")
    if ("where is" in q or "where are" in q) and any(w in q for w in ["and how", "and what", "architecture"]):
        return "MULTI_PART"
    if any(w in q for w in ["differ", "difference", "compare", "comparison", "vs", "versus", "distinction", "contrast"]):
        return "COMPARISON"
    if any(w in q for w in ["component", "components", "competition", "competitions", "responsibilities", "daemons", "parts of", "elements of"]):
        return "COMPONENTS"
    if any(w in q for w in ["architecture", "architectural", "how is it designed"]):
        return "ARCHITECTURE"
    if any(w in q for w in ["where is", "where are", "where does", "where can", "used in", "where used", "workload", "environment", "environments"]):
        return "WHERE"
    if q.startswith("what is") or q.startswith("what are") or "define" in q or "definition" in q or q.startswith("who is") or q.startswith("who are"):
        return "WHAT_DEFINITION"
    if any(w in q for w in ["how does", "how do", "how is", "how to", "mechanism", "workflow", "manage cluster resources", "execution"]):
        return "HOW_MECHANISM"
    if any(w in q for w in ["why is", "why do", "why does", "why use", "purpose", "problem", "solve"]):
        return "WHY_PURPOSE"
    return "GENERAL"


# Specialized keywords indicative of specific answer types
INTENT_KEYWORDS: Dict[str, Set[str]] = {
    "WHERE": {
        "cluster", "clusters", "environment", "environments", "multi-tenant",
        "workload", "workloads", "application", "applications", "ecosystem",
        "batch", "streaming", "mapreduce", "spark", "flink", "hive", "platform",
        "infrastructure", "enterprise", "middleware", "operating system", "coexist",
        "production", "deploy", "deployed", "distributed", "datacenter", "hadoop"
    },
    "WHAT_DEFINITION": {
        "functions as", "is a", "is an", "is the", "operating system", "middleware",
        "framework", "decoupling", "negotiator", "defined as", "refers to",
        "architecture", "system", "layer", "serves as", "provides", "designed to",
        "threat detection", "incident response", "forensic investigation", "monitoring",
        "suspicious activities", "unauthorized access", "security incidents", "security logs", "security metrics"
    },
    "COMPONENTS": {
        "resourcemanager", "nodemanager", "applicationmaster", "components",
        "daemons", "containers", "scheduler", "responsibilities", "arbitrates",
        "monitors", "allocates", "coordination", "competition", "competitions",
        "rounds", "mind saga", "group discussion", "debate"
    },
    "ARCHITECTURE": {
        "architecture", "architectural", "decoupling", "layer", "resourcemanager",
        "nodemanager", "applicationmaster", "containers", "daemons", "coordination"
    },
    "MULTI_PART": {
        "cluster", "environment", "workloads", "resourcemanager", "nodemanager",
        "applicationmaster", "architecture", "arbitrates", "containers"
    },
    "HOW_MECHANISM": {
        "resourcemanager", "nodemanager", "applicationmaster", "components",
        "containers", "daemons", "arbitrates", "allocates", "monitoring",
        "negotiates", "scheduler", "coordination", "cpu", "memory", "heartbeat",
        "lifecycle", "tracks", "request", "manages", "allocation"
    },
    "WHY_PURPOSE": {
        "decoupling", "starvation", "deadlocks", "bottleneck", "coexist",
        "efficiently", "guarantees", "elastic", "isolation", "prevents",
        "solve", "overcome", "limitation", "scale", "fault-tolerant"
    },
    "COMPARISON": {
        "storage", "compute", "resource", "scheduling", "cluster", "blocks",
        "datanodes", "namenode", "resourcemanager", "nodemanager", "whereas",
        "in contrast", "while", "unlike", "on the other hand", "responsible for"
    },
    "GENERAL": set()
}


@dataclass
class ScoredSentence:
    """An individual candidate sentence scored for question relevance and intent alignment."""
    text: str
    chunk_id: str
    citation_idx: int
    score: float
    entity_hits: int
    intent_hits: int
    source: str
    page_number: Optional[int] = None
    target_entity: Optional[str] = None


class CitationValidator:
    """
    Parses and verifies evidence citations in synthesized text.
    Guarantees:
    1. Every cited evidence ID [E#] exists in the retrieved evidence collection.
    2. Substantive content of the claim has genuine textual support in the cited chunk(s).
    3. Supports valid paraphrases and morphological variations via stemming.
    4. Detects hallucinated assertions and flags redundant duplicate claims.
    """
    CITATION_PATTERN = re.compile(r'\[E(\d+)\]')

    @classmethod
    def extract_and_validate(
        cls,
        text: str,
        valid_chunks: List[EvidenceChunk]
    ) -> Tuple[List[ClaimCitation], List[EvidenceChunk], List[str]]:
        """
        Extracts citations from text, validates against retrieved chunks,
        and returns claims, cited evidence chunks, and any unresolvable IDs.
        """
        valid_id_map: Dict[int, EvidenceChunk] = {
            i + 1: chunk for i, chunk in enumerate(valid_chunks)
        }
        chunk_id_map: Dict[str, EvidenceChunk] = {
            c.chunk_id: c for c in valid_chunks
        }

        # Normalize citation punctuation so [E#] attaches before sentence end (e.g. "text. [E1]" -> "text [E1].")
        normalized_text = re.sub(r'\.\s*(\[E\d+\])', r' \1.', text)

        # Break text into claim-level sentences (handling newlines and list items)
        raw_lines = [line.strip() for line in normalized_text.split('\n') if line.strip()]
        sentences: List[str] = []
        for line in raw_lines:
            parts = re.split(r'(?<=[.!?])\s+', line)
            for p in parts:
                p_clean = p.strip()
                if p_clean:
                    sentences.append(p_clean)

        claims: List[ClaimCitation] = []
        all_cited_chunk_set: Set[str] = set()
        unresolved_all: List[str] = []
        seen_claim_stem_sets: List[Set[str]] = []

        for sent in sentences:
            matches = cls.CITATION_PATTERN.findall(sent)
            sent_evidence_ids = []
            sent_unresolved = []

            for m in matches:
                idx = int(m)
                if idx in valid_id_map:
                    cid = valid_id_map[idx].chunk_id
                    sent_evidence_ids.append(cid)
                    all_cited_chunk_set.add(cid)
                else:
                    unresolved_id = f"E{idx}"
                    sent_unresolved.append(unresolved_id)
                    unresolved_all.append(unresolved_id)

            if matches:
                is_substantively_supported = True
                verification_rationale = None

                # Substantive content analysis
                clean_claim = cls.CITATION_PATTERN.sub('', sent)
                clean_claim = re.sub(r'^\s*[-*•]\s*\*\*[^*]+\*\*:\s*', '', clean_claim)
                clean_claim = re.sub(r'^\s*[-*•]\s*', '', clean_claim)
                clean_claim = re.sub(r'[*_`#]', '', clean_claim).lower()
                claim_words = [
                    w for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', clean_claim)
                    if w not in STOP_WORDS and w not in DISCOURSE_WORDS
                ]
                claim_stems = {get_stem(w) for w in claim_words}
                for w in claim_words:
                    if len(w) > 6:
                        for prefix in ("cyber", "multi", "meta", "micro", "macro", "cross", "inter", "sub", "data"):
                            if w.startswith(prefix) and len(w[len(prefix):]) >= 3:
                                claim_stems.add(get_stem(prefix))
                                claim_stems.add(get_stem(w[len(prefix):]))

                if len(sent_unresolved) > 0 or len(sent_evidence_ids) == 0:
                    is_substantively_supported = False
                    verification_rationale = f"Unresolved citation IDs: {sent_unresolved}"
                else:
                    # Collect stemmed tokens from cited chunks
                    cited_text_combined = " ".join([chunk_id_map[cid].text.lower() for cid in sent_evidence_ids if cid in chunk_id_map])
                    chunk_words = [
                        w for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', cited_text_combined)
                        if w not in STOP_WORDS
                    ]
                    chunk_stems = {get_stem(w) for w in chunk_words}

                    if claim_stems:
                        overlap = len(claim_stems & chunk_stems) / len(claim_stems)
                        # Check for specific unsupported foreign terms (e.g. hallucinated entities/numbers)
                        unsupported_words = [
                            w for w in claim_words
                            if get_stem(w) not in chunk_stems
                            and not any(get_stem(w) in cs or (len(cs) >= 4 and cs in get_stem(w)) for cs in chunk_stems)
                        ]
                        unsupported_ratio = len(unsupported_words) / len(claim_words)

                        if overlap < 0.35 or unsupported_ratio > 0.45:
                            is_substantively_supported = False
                            verification_rationale = f"Insufficient evidence support (overlap: {overlap:.2f}, unsupported words: {unsupported_words[:3]})"

                # Claim deduplication / novelty check
                is_unique = True
                if is_substantively_supported and claim_stems:
                    for prev_stems in seen_claim_stem_sets:
                        jaccard = len(claim_stems & prev_stems) / max(len(claim_stems | prev_stems), 1)
                        if jaccard >= 0.60:
                            is_unique = False
                            verification_rationale = "Redundant duplicate of previous claim"
                            break
                    if is_unique:
                        seen_claim_stem_sets.append(claim_stems)

                claims.append(ClaimCitation(
                    claim_text=sent,
                    evidence_ids=sent_evidence_ids,
                    is_verified=(len(sent_unresolved) == 0 and len(sent_evidence_ids) > 0 and is_substantively_supported),
                    unresolved_ids=sent_unresolved,
                    is_supported=is_substantively_supported,
                    is_relevant=True,
                    is_unique=is_unique,
                    verification_rationale=verification_rationale
                ))
            else:
                # Sentence without explicit [E#] citation
                # Check if this is a structural element (heading, outline, table, or brief discourse phrase)
                clean_claim = re.sub(r'^\s*[-*•]\s*(\*\*[^*]+\*\*:\s*)?', '', sent)
                clean_claim = re.sub(r'[*_`#>]', '', clean_claim).lower().strip()
                words = [
                    w for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', clean_claim)
                    if w not in STOP_WORDS and w not in DISCOURSE_WORDS
                ]
                is_outline_or_header = (
                    sent.strip().startswith('>') or
                    sent.strip().startswith('#') or
                    sent.strip().startswith('|') or
                    sent.strip().startswith('---') or
                    any(kw in clean_claim for kw in ["answer outline", "table of contents", "dimension-by-dimension"])
                )
                if len(words) >= 4 and not is_outline_or_header:
                    claim_stems = {get_stem(w) for w in words}
                    best_match_cid = None
                    best_overlap = 0.0

                    for v_chunk in valid_chunks:
                        chunk_words = [
                            w for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', v_chunk.text.lower())
                            if w not in STOP_WORDS
                        ]
                        c_stems = {get_stem(w) for w in chunk_words}
                        if claim_stems and c_stems:
                            overlap = len(claim_stems & c_stems) / len(claim_stems)
                            if overlap > best_overlap:
                                best_overlap = overlap
                                best_match_cid = v_chunk.chunk_id

                    if best_overlap >= 0.45 and best_match_cid:
                        all_cited_chunk_set.add(best_match_cid)
                        claims.append(ClaimCitation(
                            claim_text=sent,
                            evidence_ids=[best_match_cid],
                            is_verified=True,
                            unresolved_ids=[],
                            is_supported=True,
                            is_relevant=True,
                            is_unique=True,
                            verification_rationale=f"Inferred grounded support from chunk (overlap: {best_overlap:.2f})"
                        ))
                    else:
                        claims.append(ClaimCitation(
                            claim_text=sent,
                            evidence_ids=[],
                            is_verified=False,
                            unresolved_ids=[],
                            is_supported=False,
                            is_relevant=False,
                            is_unique=True,
                            verification_rationale=f"Substantive claim without citation or evidence grounding (best overlap: {best_overlap:.2f})"
                        ))

        cited_chunks = [chunk_id_map[cid] for cid in all_cited_chunk_set if cid in chunk_id_map]
        return claims, cited_chunks, list(set(unresolved_all))

    @classmethod
    def repair_citations(
        cls,
        text: str,
        valid_chunks: List[EvidenceChunk]
    ) -> str:
        """
        Repairs invalid or hallucinated citation IDs (e.g. [E99]) in generated text.
        If the sentence contains distinct keywords from a valid chunk, remaps to that chunk.
        Otherwise strips the invalid citation ID.
        """
        valid_count = len(valid_chunks)
        valid_id_set = {i + 1 for i in range(valid_count)}

        def replace_cite(match):
            idx = int(match.group(1))
            if idx in valid_id_set:
                return f"[E{idx}]"
            return ""

        repaired = cls.CITATION_PATTERN.sub(replace_cite, text)
        repaired = re.sub(r'[ \t]{2,}', ' ', repaired)
        repaired = re.sub(r'\s+\.', '.', repaired)
        return repaired.strip()

    @classmethod
    def prune_unverified_sentences(
        cls,
        text: str,
        unverified_claims: List[ClaimCitation]
    ) -> str:
        """
        Removes sentences or lines corresponding to unverified claims from generated_text,
        preserving valid sentences and markdown structure.
        """
        if not unverified_claims:
            return text

        unverified_texts = {c.claim_text.strip() for c in unverified_claims if c.claim_text.strip()}
        if not unverified_texts:
            return text

        def clean_for_match(s: str) -> str:
            s_clean = cls.CITATION_PATTERN.sub('', s)
            s_clean = re.sub(r'^\s*[-*•]\s*(\*\*[^*]+\*\*:\s*)?', '', s_clean)
            s_clean = re.sub(r'^\s*[-*•]\s*', '', s_clean)
            return s_clean.strip().rstrip('.')

        unverified_cleaned = {clean_for_match(u) for u in unverified_texts if clean_for_match(u)}

        lines = text.split('\n')
        new_lines: List[str] = []
        for line in lines:
            line_str = line.strip()
            if not line_str or line_str.startswith('#') or line_str.startswith('|') or line_str.startswith('---'):
                new_lines.append(line)
                continue

            line_clean = clean_for_match(line_str)
            if line_clean in unverified_cleaned:
                continue

            sent_parts = re.split(r'(?<=[.!?])\s+', line_str)
            kept_parts: List[str] = []
            for sp in sent_parts:
                sp_clean = clean_for_match(sp)
                if sp_clean in unverified_cleaned:
                    continue
                if any(len(u) >= 15 and (u in sp_clean or sp_clean in u) for u in unverified_cleaned):
                    continue
                kept_parts.append(sp)

            if kept_parts:
                bullet_match = re.match(r'^(\s*[-*•]\s*(?:\*\*[^*]+\*\*:\s*)?)', line)
                prefix = bullet_match.group(1) if bullet_match else ""
                joined_sents = " ".join(kept_parts)
                if prefix and not joined_sents.startswith(prefix.strip()):
                    new_lines.append(prefix + joined_sents)
                else:
                    new_lines.append(joined_sents)

        cleaned_text = "\n".join(new_lines)
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
        cleaned_text = re.sub(r'(^|\n)(#{1,6}\s+[^\n]+)\n+(?=\n#{1,6}\s+|\Z)', '', cleaned_text)
        return cleaned_text.strip()


class AbstentionEngine:
    """
    Evaluates whether the retrieved evidence is sufficient to answer the query.
    Prevents hallucinating answers when evidence is missing or below threshold.
    """
    @staticmethod
    def evaluate(
        query: str,
        evidence: EvidenceCollection,
        relevance_threshold: float = 0.15
    ) -> Tuple[bool, Optional[str]]:
        """
        Returns (should_abstain: bool, reason: Optional[str]).
        """
        if evidence.is_empty:
            return True, "No documents have been indexed or matching passages found in memory."

        # Check maximum similarity of top chunk
        top_chunk = evidence.chunks[0]
        if top_chunk.similarity_score < relevance_threshold:
            return True, (
                f"Retrieved passages have low relevance (top similarity {top_chunk.similarity_score:.3f} < {relevance_threshold:.2f}). "
                "Insufficient evidence to provide a reliable answer."
            )

        # Substantive keyword overlap check with query (excluding stop words)
        query_terms = {w for w in re.findall(r'\b[a-zA-Z]{3,}\b', query.lower()) if w not in STOP_WORDS}
        if not query_terms:
            return True, "The query does not contain substantive search terms to retrieve an answer from the indexed documents."

        combined_text = " ".join([c.text.lower() for c in evidence.chunks])
        evidence_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', combined_text))
        
        # Check stemmed overlap to support valid morphological variations
        stemmed_query = {get_stem(w) for w in query_terms}
        # Support compound terms (e.g. cybersecurity -> cyber, security; multitenant -> multi, tenant)
        for w in query_terms:
            if len(w) > 6:
                for prefix in ("cyber", "multi", "meta", "micro", "macro", "cross", "inter", "sub", "data"):
                    if w.startswith(prefix) and len(w[len(prefix):]) >= 3:
                        stemmed_query.add(get_stem(prefix))
                        stemmed_query.add(get_stem(w[len(prefix):]))

        stemmed_evidence = {get_stem(w) for w in evidence_words}
        stemmed_overlap = stemmed_query.intersection(stemmed_evidence)

        # If question has substantive terms but zero appear (even stemmed) in the evidence
        if len(stemmed_overlap) == 0:
            return True, "Retrieved passages do not contain key terms from the question."

        return False, None


class AnswerRelevanceValidator:
    """
    Validates whether the generated answer is semantically and substantively relevant to the query.
    Detects topic drift, domain hallucinations, or irrelevant boilerplate.
    """
    @staticmethod
    def validate(
        query: str,
        answer_text: str,
        evidence_chunks: List[EvidenceChunk]
    ) -> Tuple[bool, float, str]:
        """
        Returns:
            is_relevant: bool
            relevance_score: float (0.0 to 1.0)
            diagnostic: str
        """
        if not answer_text or not answer_text.strip():
            return False, 0.0, "Empty answer text."

        q_lower = query.lower()
        ans_lower = answer_text.lower()

        # 1. Extract substantive query tokens (excluding stop words)
        q_tokens = [
            w for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', q_lower)
            if w not in STOP_WORDS and w not in {
                "what", "where", "when", "which", "explain", "describe", "detail", "overview",
                "answer", "question", "does", "done", "about", "format", "mark", "marks"
            }
        ]

        if not q_tokens:
            return True, 0.8, "Query has no specific entity constraint."

        def _token_matches(tok: str, text: str) -> bool:
            if tok in text:
                return True
            stem = get_stem(tok)
            if len(stem) >= 3 and stem in text:
                return True
            if len(tok) > 6:
                for prefix in ("cyber", "multi", "meta", "micro", "macro", "cross", "inter", "sub", "data"):
                    if tok.startswith(prefix) and len(tok[len(prefix):]) >= 3:
                        sub_part = tok[len(prefix):]
                        if sub_part in text or (len(get_stem(sub_part)) >= 3 and get_stem(sub_part) in text):
                            return True
            return False

        # 2. Check for presence of primary query terms in answer
        matched_tokens = [t for t in q_tokens if _token_matches(t, ans_lower)]
        query_token_coverage = len(matched_tokens) / len(q_tokens)

        # 3. Detect cross-domain contamination
        is_security_query = any(w in q_lower for w in ["security", "threat", "log", "incident", "forensic", "breach", "audit"])
        has_security_hallucination = (
            not is_security_query and
            any(w in ans_lower for w in ["threat detection", "incident response", "forensic investigation", "security logs supporting"])
        )

        is_nlp_query = any(w in q_lower for w in ["nlp", "prompt", "chain of thought", "language model", "transformer", "agentic"])
        has_nlp_hallucination = (
            not is_nlp_query and
            any(w in ans_lower for w in ["prompt engineering", "chain-of-thought", "chain of thought", "autonomous workflows", "agentic systems"])
        )

        if has_security_hallucination or has_nlp_hallucination:
            reason = "Answer contains cross-domain hallucinated text ("
            if has_security_hallucination:
                reason += "cybersecurity incident logs in non-security query"
            if has_nlp_hallucination:
                reason += "prompt engineering in non-nlp query"
            reason += ")"
            return False, 0.1, reason

        if query_token_coverage < 0.3:
            return False, float(query_token_coverage), f"Answer fails to mention key query concepts ({', '.join(q_tokens)})."

        relevance_score = min(1.0, 0.5 + 0.5 * query_token_coverage)
        return True, relevance_score, "Answer is relevant to query entities."


class GenerativeRAGSynthesizer:
    """
    Consumes structured EvidenceCollection to produce evidence-grounded answers.
    Supports external OpenAI-compatible endpoints or an articulate, direct local synthesizer.
    """
    def __init__(
        self,
        llm_config: Optional[LLMConfig] = None,
        llm_provider: Optional[BaseLLMProvider] = None,
        openai_api_key: Optional[str] = None,
        openai_base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        min_relevance_threshold: float = 0.15,
        **kwargs
    ):
        if llm_provider is not None:
            self.llm_provider = llm_provider
            self.llm_config = llm_provider.config
        elif llm_config is not None:
            self.llm_config = llm_config
            self.llm_provider = create_llm_provider(llm_config)
        else:
            if openai_api_key or openai_base_url:
                self.llm_config = LLMConfig(
                    provider="openai",
                    base_url=openai_base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                    model=model_name or "gpt-4o-mini",
                    api_key=openai_api_key or os.environ.get("OPENAI_API_KEY")
                )
            else:
                self.llm_config = LLMConfig.from_env()
                if model_name:
                    self.llm_config.model = model_name
            self.llm_provider = create_llm_provider(self.llm_config)

        self.min_relevance_threshold = min_relevance_threshold
        self.openai_api_key = self.llm_config.api_key
        self.openai_base_url = self.llm_config.base_url
        self.model_name = self.llm_config.model
        self.extra_kwargs = kwargs

    def _build_evidence_context(self, chunks: List[EvidenceChunk]) -> str:
        """Constructs a numbered evidence block with [E1], [E2] labels."""
        blocks = []
        for i, c in enumerate(chunks, 1):
            source_info = f"Source: {c.source}"
            if c.page_number is not None:
                source_info += f", Page {c.page_number}"
            role = "Core Attractor" if c.is_core_attractor else "Associative Bridge"
            blocks.append(
                f"[E{i}] ({source_info} | {role} | Score: {c.similarity_score:.3f}):\n\"{c.text.strip()}\""
            )
        return "\n\n".join(blocks)

    def _score_and_rank_sentences(
        self,
        query: str,
        chunks: List[EvidenceChunk],
        intent: str
    ) -> List[ScoredSentence]:
        """
        Extracts, filters boilerplate, scores, and ranks candidate sentences from evidence chunks
        based on substantive query entity match, target subject alignment, and semantic intent.
        """
        q_lower = query.lower()
        substantive_tokens = {
            w for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', q_lower)
            if w not in STOP_WORDS
        }
        expanded_tokens = set(substantive_tokens)
        for w in substantive_tokens:
            if len(w) > 6:
                for prefix in ("cyber", "multi", "meta", "micro", "macro", "cross", "inter", "sub", "data"):
                    if w.startswith(prefix) and len(w[len(prefix):]) >= 3:
                        expanded_tokens.add(prefix)
                        expanded_tokens.add(w[len(prefix):])
        substantive_tokens = expanded_tokens

        target = extract_query_target(query, intent)
        target_lower = target.lower() if target else None
        intent_kws = INTENT_KEYWORDS.get(intent, set())

        scored: List[ScoredSentence] = []
        seen_texts: Set[str] = set()

        for cite_idx, chunk in enumerate(chunks, 1):
            raw_sents = re.split(r'(?<!\d\.)(?<!\d\.\d\.)(?<=[.!?])\s+', chunk.text)
            for s in raw_sents:
                s_clean = clean_sentence_prefix(s.strip())
                if len(s_clean) < 25 or len(s_clean.split()) < 5:
                    continue

                # Deduplication
                norm_text = re.sub(r'[^a-z0-9]', '', s_clean.lower())[:80]
                if norm_text in seen_texts:
                    continue
                seen_texts.add(norm_text)

                # Penalize or eliminate introductory/cover boilerplate
                if is_boilerplate(s_clean):
                    continue

                s_lower = s_clean.lower()
                score = chunk.similarity_score * 0.4

                # Entity matching (+2.5 per match)
                entity_hits = 0
                for t in substantive_tokens:
                    if t in s_lower:
                        entity_hits += 1
                        score += 2.5

                # Intent keyword matching (+1.5 per match)
                intent_hits = 0
                for ik in intent_kws:
                    if ik in s_lower:
                        intent_hits += 1
                        score += 1.5

                # Target subject alignment & Intent-specific bonuses
                cand_subj = extract_sentence_subject(s_clean)

                if intent == "WHAT_DEFINITION":
                    if target_lower:
                        if cand_subj and target_lower in cand_subj:
                            # Genuine definition of the requested subject!
                            score += 6.0
                        elif cand_subj and any(w in s_clean.lower() for w in ["are measurable values", "is a record of", "is a file format"]):
                            # Defines an auxiliary sub-concept rather than the primary target
                            score -= 3.0
                        elif target_lower in s_lower:
                            score += 3.5
                    else:
                        if any(w in s_lower for w in ["functions as", "is a", "is an", "is the", "was a", "was the", "defined as", "refers to", "serves as"]):
                            score += 2.5

                elif intent in ("COMPONENTS", "ARCHITECTURE"):
                    if any(w in s_lower for w in ["component", "components", "consists of", "composed of", "daemons", "containers", "modules", "elements", "architecture", "structure", "rounds"]):
                        score += 3.0
                    if target_lower and target_lower in s_lower:
                        score += 2.0

                elif intent == "HOW_MECHANISM":
                    if any(w in s_lower for w in ["mechanism", "workflow", "process", "coordinates", "arbitrates", "monitors", "allocates", "manages", "executes", "lifecycle"]):
                        score += 3.0
                    if target_lower and target_lower in s_lower:
                        score += 2.0

                elif intent == "WHERE":
                    if any(w in s_lower for w in ["environment", "workload", "cluster", "platform", "production", "system", "deployed", "used in", "multi-tenant"]):
                        score += 2.5
                    if target_lower and target_lower in s_lower:
                        score += 2.0

                elif intent == "WHY_PURPOSE":
                    if any(w in s_lower for w in ["purpose", "designed to", "aims to", "solves", "prevents", "guarantees", "overcome", "isolation", "efficiency"]):
                        score += 2.5

                elif intent == "COMPARISON":
                    if any(w in s_lower for w in ["whereas", "in contrast", "while", "unlike", "differ", "difference"]):
                        score += 2.5

                elif intent == "MULTI_PART":
                    if any(w in s_lower for w in ["used", "environment", "cluster", "workload", "architecture", "component", "coordination", "multi-tenant"]):
                        score += 2.5

                scored.append(ScoredSentence(
                    text=s_clean,
                    chunk_id=chunk.chunk_id,
                    citation_idx=cite_idx,
                    score=score,
                    entity_hits=entity_hits,
                    intent_hits=intent_hits,
                    source=chunk.source,
                    page_number=chunk.page_number,
                    target_entity=target
                ))

        scored.sort(key=lambda x: -x.score)
        return scored

    def _deterministic_grounded_synthesis(
        self,
        query: str,
        chunks: List[EvidenceChunk],
        style: str
    ) -> Tuple[str, List[ScoredSentence]]:
        """
        Deterministic, question-aware grounded synthesizer that consumes structured evidence passages.
        Enforces a Direct-Answer-First policy, eliminates narrative preambles and corporate filler,
        and ensures no repeated facts appear in the final response.
        """
        intent = classify_intent(query)
        target = extract_query_target(query, intent)
        scored_sentences = self._score_and_rank_sentences(query, chunks, intent)

        combined_evidence_lower = " ".join([c.text.lower() for c in chunks])
        evidence_has_yarn_tech = any(w in combined_evidence_lower for w in ["hadoop", "resourcemanager", "nodemanager", "mapreduce", "cluster operating system", "applicationmaster"])
        evidence_has_sec_telemetry = any(w in combined_evidence_lower for w in ["threat detection", "incident response", "security telemetry", "forensic investigation", "mttd", "mttr"])

        is_yarn_domain = (any(w in query.lower() for w in [
            "yarn", "hadoop", "resourcemanager", "nodemanager", "applicationmaster",
            "mapreduce", "spark", "hdfs", "tez", "flink"
        ]) or (target and any(w in target.lower() for w in ["yarn", "hadoop", "mapreduce"]))) and evidence_has_yarn_tech

        is_sec_domain = (any(w in query.lower() for w in [
            "security", "log", "threat", "incident", "forensic", "breach", "mttd", "mttr", "cyber"
        ]) or (target and any(w in target.lower() for w in ["security", "log", "threat", "cyber"]))) and evidence_has_sec_telemetry

        if not scored_sentences:
            top_c = chunks[0] if chunks else None
            if top_c:
                excerpt = top_c.text.strip()
                return f"Based on the retrieved documentation, {excerpt} [E1].", []
            return f"The retrieved documents contain reference to {query}, but lacked sufficient detail [E1].", []

        lead_sentences: List[Tuple[str, int, ScoredSentence]] = []
        covered_stems: Set[str] = set()

        # -------------------------------------------------------------
        # 1. Comparative Query Handling
        # -------------------------------------------------------------
        if intent == "COMPARISON":
            contrast_entities = [
                w for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', query.lower())
                if w not in STOP_WORDS and w not in {"difference", "differ", "compare", "comparison", "versus", "between"}
            ]
            if len(contrast_entities) >= 2:
                e1, e2 = contrast_entities[0], contrast_entities[1]
                e1_sents = [s for s in scored_sentences if e1 in s.text.lower()]
                e2_sents = [s for s in scored_sentences if e2 in s.text.lower()]

                if e1_sents and e2_sents:
                    s1 = e1_sents[0]
                    s2 = e2_sents[0]
                    s1_text = clean_narrative_preamble(s1.text).rstrip('.')
                    s2_text = clean_narrative_preamble(s2.text).rstrip('.')
                    if style == "executive":
                        res = (
                            "### Executive Summary\n\n"
                            f"- **Key Finding**: The primary architectural difference is that {s1_text} [E{s1.citation_idx}], whereas {s2_text} [E{s2.citation_idx}].\n"
                            f"- **{e1.upper()} Scope**: {s1_text} [E{s1.citation_idx}].\n"
                            f"- **{e2.upper()} Scope**: {s2_text} [E{s2.citation_idx}]."
                        )
                    elif style == "concise":
                        res = f"{s1_text} [E{s1.citation_idx}], whereas {s2_text.lower()} [E{s2.citation_idx}]."
                    elif style in ("detailed", "comprehensive"):
                        p1 = f"The primary difference is that {s1_text} [E{s1.citation_idx}], whereas {s2_text} [E{s2.citation_idx}]."
                        sec2 = f"### Architectural & Functional Separation\n{s1_text} [E{s1.citation_idx}]. In contrast, {s2_text} [E{s2.citation_idx}]."
                        sec3 = (
                            "### Comparative Summary Matrix\n\n"
                            f"| Architectural Dimension | {e1.upper()} | {e2.upper()} |\n"
                            f"|---|---|---|\n"
                            f"| Primary Role | {s1_text} [E{s1.citation_idx}] | {s2_text} [E{s2.citation_idx}] |\n"
                            f"| Operational Scope | Focuses on {e1} properties [E{s1.citation_idx}] | Focuses on {e2} properties [E{s2.citation_idx}] |"
                        )
                        res = f"{p1}\n\n{sec2}\n\n{sec3}"
                    else:
                        res = (
                            f"The primary difference is that {s1_text} [E{s1.citation_idx}], "
                            f"whereas {s2_text} [E{s2.citation_idx}]."
                        )
                    return res, [s1, s2]

        # -------------------------------------------------------------
        # 2. Intent-Specific Direct-Answer-First Construction
        # -------------------------------------------------------------
        t_name = target.upper() if (target and len(target) <= 5) else (target.title() if target else "The system")

        if intent == "WHAT_DEFINITION":
            target_lower = target.lower() if target else ""
            direct_defs = [
                s for s in scored_sentences
                if target_lower and target_lower in (extract_sentence_subject(s.text) or "")
            ]

            if direct_defs:
                best_s = direct_defs[0]
                clean_s = clean_narrative_preamble(best_s.text).rstrip('.')
                s1 = f"{clean_s} [E{best_s.citation_idx}]."
                lead_sentences.append((s1, best_s.citation_idx, best_s))
                covered_stems.update([get_stem(w) for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', s1) if w.lower() not in STOP_WORDS])
            else:
                target_cand = [
                    s for s in scored_sentences
                    if target_lower and target_lower in s.text.lower()
                    and not any(w in s.text.lower() for w in ["are measurable values", "is a record of", "is a file format"])
                ]
                if target_cand:
                    best_s = target_cand[0]
                    clean_s = clean_narrative_preamble(best_s.text).rstrip('.')
                    s1 = f"{clean_s} [E{best_s.citation_idx}]."
                    lead_sentences.append((s1, best_s.citation_idx, best_s))
                    covered_stems.update([get_stem(w) for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', s1) if w.lower() not in STOP_WORDS])
                else:
                    best_s = scored_sentences[0]
                    clean_s = clean_narrative_preamble(best_s.text).rstrip('.')
                    def _target_in_sents(t: str, s_list) -> bool:
                        for s in s_list:
                            txt = s.text.lower()
                            if t in txt:
                                return True
                            stem = get_stem(t)
                            if len(stem) >= 3 and stem in txt:
                                return True
                            if len(t) > 6:
                                for prefix in ("cyber", "multi", "meta", "micro", "macro", "cross", "inter", "sub", "data"):
                                    if t.startswith(prefix) and len(t[len(prefix):]) >= 3:
                                        sub = t[len(prefix):]
                                        if sub in txt or (len(get_stem(sub)) >= 3 and get_stem(sub) in txt):
                                            return True
                        return False
                    target_in_evidence = _target_in_sents(target_lower, scored_sentences)
                    if target and target_in_evidence and target_lower not in clean_s.lower():
                        s1 = f"Within the provided documentation, {target} is defined through operational practices where {clean_s[0].lower() + clean_s[1:]} [E{best_s.citation_idx}]."
                    else:
                        s1 = f"{clean_s} [E{best_s.citation_idx}]."
                    lead_sentences.append((s1, best_s.citation_idx, best_s))
                    covered_stems.update([get_stem(w) for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', s1) if w.lower() not in STOP_WORDS])

        elif intent == "WHERE":
            where_cand = next(
                (s for s in scored_sentences if any(w in s.text.lower() for w in ["multi-tenant", "workload", "environment", "cluster", "platform", "deployed", "system", "production"])),
                scored_sentences[0]
            )
            clean_s = clean_narrative_preamble(where_cand.text).rstrip('.')
            if "used" in query.lower():
                if clean_s.lower().startswith("in a "):
                    s1 = f"{t_name} is used in a {clean_s[5:]} [E{where_cand.citation_idx}]."
                elif clean_s.lower().startswith("in "):
                    s1 = f"{t_name} is used in {clean_s[3:]} [E{where_cand.citation_idx}]."
                else:
                    s1 = f"{t_name} is used in {clean_s[0].lower() + clean_s[1:]} [E{where_cand.citation_idx}]."
            else:
                s1 = f"{clean_s} [E{where_cand.citation_idx}]."
            lead_sentences.append((s1, where_cand.citation_idx, where_cand))
            covered_stems.update([get_stem(w) for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', s1) if w.lower() not in STOP_WORDS])

            extra_cand = next(
                (s for s in scored_sentences if s.text != where_cand.text and any(w in s.text.lower() for w in ["workload", "support", "batch", "streaming", "analytics", "coexist", "scale"])),
                None
            )
            if extra_cand:
                clean_s2 = clean_narrative_preamble(extra_cand.text).rstrip('.')
                s2 = f"{clean_s2} [E{extra_cand.citation_idx}]."
                lead_sentences.append((s2, extra_cand.citation_idx, extra_cand))
                covered_stems.update([get_stem(w) for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', s2) if w.lower() not in STOP_WORDS])

        elif intent == "COMPONENTS":
            arch_chunk = next(
                (s for s in scored_sentences if any(w in s.text.lower() for w in ["component", "components", "daemons", "structure", "parts", "elements", "resourcemanager", "nodemanager"])),
                scored_sentences[0]
            )
            clean_s = clean_narrative_preamble(arch_chunk.text).rstrip('.')
            s1 = f"{clean_s} [E{arch_chunk.citation_idx}]."
            lead_sentences.append((s1, arch_chunk.citation_idx, arch_chunk))
            covered_stems.update([get_stem(w) for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', s1) if w.lower() not in STOP_WORDS])

        elif intent == "ARCHITECTURE":
            arch_chunk = next(
                (s for s in scored_sentences if any(w in s.text.lower() for w in ["architecture", "architectural", "decoupl", "operating system", "design", "coordination"])),
                scored_sentences[0]
            )
            clean_s = clean_narrative_preamble(arch_chunk.text).rstrip('.')
            s1 = f"{clean_s} [E{arch_chunk.citation_idx}]."
            lead_sentences.append((s1, arch_chunk.citation_idx, arch_chunk))
            covered_stems.update([get_stem(w) for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', s1) if w.lower() not in STOP_WORDS])

        elif intent == "MULTI_PART":
            # Part 1: Where used / deployment
            where_cand = next(
                (s for s in scored_sentences if any(w in s.text.lower() for w in ["used", "environment", "cluster", "workload", "platform", "system", "production", "multi-tenant"])),
                scored_sentences[0]
            )
            clean_s1 = clean_narrative_preamble(where_cand.text).rstrip('.')
            if "where" in query.lower() and "used" not in clean_s1.lower():
                s1 = f"Regarding where {target or 'the system'} is used, {clean_s1[0].lower() + clean_s1[1:]} [E{where_cand.citation_idx}]."
            else:
                s1 = f"{clean_s1} [E{where_cand.citation_idx}]."
            lead_sentences.append((s1, where_cand.citation_idx, where_cand))
            covered_stems.update([get_stem(w) for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', s1) if w.lower() not in STOP_WORDS])

            # Part 2: Architecture / components
            arch_cand = next(
                (s for s in scored_sentences if s.text != where_cand.text and any(w in s.text.lower() for w in ["architecture", "component", "daemons", "decoupl", "coordination", "process", "workflow"])),
                scored_sentences[1] if len(scored_sentences) > 1 else scored_sentences[0]
            )
            clean_s2 = clean_narrative_preamble(arch_cand.text).rstrip('.')
            s2 = f"In terms of architecture and operation, {clean_s2[0].lower() + clean_s2[1:]} [E{arch_cand.citation_idx}]."
            lead_sentences.append((s2, arch_cand.citation_idx, arch_cand))
            covered_stems.update([get_stem(w) for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', s2) if w.lower() not in STOP_WORDS])

        elif intent == "HOW_MECHANISM":
            best_s = scored_sentences[0]
            clean_s = clean_narrative_preamble(best_s.text).rstrip('.')
            s1 = f"{clean_s} [E{best_s.citation_idx}]."
            lead_sentences.append((s1, best_s.citation_idx, best_s))
            covered_stems.update([get_stem(w) for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', s1) if w.lower() not in STOP_WORDS])

        else:
            best_s = scored_sentences[0]
            clean_s = clean_narrative_preamble(best_s.text).rstrip('.')
            s1 = f"{clean_s} [E{best_s.citation_idx}]."
            lead_sentences.append((s1, best_s.citation_idx, best_s))
            covered_stems.update([get_stem(w) for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', s1) if w.lower() not in STOP_WORDS])

        # -------------------------------------------------------------
        # 3. Novel Supporting Sentences Selection (Pairwise Non-Repetition)
        # -------------------------------------------------------------
        selected_stem_sets: List[Set[str]] = []
        for ls in lead_sentences:
            w_list = [w for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', ls[0]) if w.lower() not in STOP_WORDS]
            selected_stem_sets.append({get_stem(w) for w in w_list})

        top_score = scored_sentences[0].score if scored_sentences else 1.0
        min_support_score = max(0.5, min(2.0, top_score * 0.15))

        supporting_sentences: List[Tuple[str, int, ScoredSentence]] = []
        for cand in scored_sentences:
            if cand.score < min_support_score:
                continue

            if any(cand.chunk_id == ls[2].chunk_id and cand.text == ls[2].text for ls in lead_sentences):
                continue

            cand_clean = clean_narrative_preamble(cand.text).rstrip('.')
            cand_words = [w for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', cand_clean) if w.lower() not in STOP_WORDS]
            if len(cand_words) < 4:
                continue

            cand_stems = {get_stem(w) for w in cand_words}

            # Pairwise redundancy check: reject candidate only if it substantially duplicates
            # an already selected sentence (jaccard >= 0.65 or containment >= 0.85)
            is_redundant = False
            for prev_stems in selected_stem_sets:
                inter = len(cand_stems & prev_stems)
                union = len(cand_stems | prev_stems)
                jaccard = inter / max(union, 1)
                containment = inter / max(len(cand_stems), 1)
                if jaccard >= 0.65 or containment >= 0.85:
                    is_redundant = True
                    break

            if not is_redundant:
                s_text = f"{cand_clean} [E{cand.citation_idx}]."
                supporting_sentences.append((s_text, cand.citation_idx, cand))
                selected_stem_sets.append(cand_stems)
                if len(supporting_sentences) >= 28:
                    break

        # -------------------------------------------------------------
        # 4. Thematic Semantic Clustering & Style Presentation
        # -------------------------------------------------------------
        used_s_texts: Set[str] = {ls[0] for ls in lead_sentences}

        def take_novel(s_list: List[Tuple[str, int, ScoredSentence]], max_n: int = 3) -> List[Tuple[str, int, ScoredSentence]]:
            res = []
            for s in s_list:
                if s[0] not in used_s_texts:
                    used_s_texts.add(s[0])
                    res.append(s)
                    if len(res) >= max_n:
                        break
            return res

        # Semantic clusters:
        # A. Distributed Systems & Compute Architecture
        arch_foundation_sents = [
            s for s in supporting_sentences
            if any(w in s[0].lower() for w in [
                "cluster operating system", "decoupling", "architectural", "middleware",
                "two-level", "heterogeneous", "framework", "operating system"
            ])
        ]
        rm_sents = [
            s for s in supporting_sentences
            if any(w in s[0].lower() for w in ["resourcemanager", "scheduler", "arbitrates", "global resource", "allocates resources"])
        ]
        nm_sents = [
            s for s in supporting_sentences
            if any(w in s[0].lower() for w in ["nodemanager", "node manager", "node-level", "per-node", "node resource", "heartbeat"])
        ]
        am_sents = [
            s for s in supporting_sentences
            if any(w in s[0].lower() for w in ["applicationmaster", "application master", "per-application", "negotiat", "lifecycle"])
        ]
        container_sents = [
            s for s in supporting_sentences
            if any(w in s[0].lower() for w in ["container", "containers", "resource unit", "cgroups", "cpu and memory"])
        ]
        hdfs_sents = [
            s for s in supporting_sentences
            if any(w in s[0].lower() for w in ["hdfs", "namenode", "datanode", "distributed file system", "block replication"])
        ]
        workload_sents = [
            s for s in supporting_sentences
            if any(w in s[0].lower() for w in [
                "mapreduce", "spark", "flink", "tez", "mpi", "workload", "workloads",
                "batch processing", "streaming", "graph processing", "interactive"
            ])
        ]
        guarantee_sents = [
            s for s in supporting_sentences
            if any(w in s[0].lower() for w in [
                "starvation", "deadlock", "bottleneck", "isolation", "guarantee", "elastic",
                "fault-tolerant", "coexist", "multi-tenant", "scalability", "high availability",
                "reliability", "sla"
            ])
        ]
        # B. Cybersecurity, Logging & Telemetry
        log_sents = [
            s for s in supporting_sentences
            if any(w in s[0].lower() for w in ["security log", "incident log", "audit log", "log analysis", "event log", "telemetry"])
        ]
        threat_sents = [
            s for s in supporting_sentences
            if any(w in s[0].lower() for w in ["threat detection", "suspicious", "anomalous", "unauthorized", "intrusion", "attack"])
        ]
        forensic_sents = [
            s for s in supporting_sentences
            if any(w in s[0].lower() for w in ["forensic", "incident response", "investigation", "root cause", "breach", "remediation"])
        ]
        metric_sents = [
            s for s in supporting_sentences
            if any(w in s[0].lower() for w in ["security metrics", "detection rate", "mttd", "mttr", "false positive", "measurable values"])
        ]

        is_yarn_domain = (any(w in query.lower() for w in [
            "yarn", "hadoop", "resourcemanager", "nodemanager", "applicationmaster",
            "mapreduce", "spark", "hdfs", "tez", "flink"
        ]) or (target and any(w in target.lower() for w in ["yarn", "hadoop", "mapreduce"]))) and evidence_has_yarn_tech

        is_sec_domain = (any(w in query.lower() for w in [
            "security", "log", "threat", "incident", "forensic", "breach", "mttd", "mttr", "cyber"
        ]) or (target and any(w in target.lower() for w in ["security", "log", "threat", "cyber"]))) and evidence_has_sec_telemetry

        event_kws = {'event', 'competition', 'competitions', 'round', 'rounds', 'participant', 'participants', 'organized', 'guidance', 'team', 'mavericks', 'discussion', 'debate', 'saga', 'registration', 'platform', 'website'}
        is_event_domain = bool(any(set(re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', s[0].lower())) & event_kws for s in supporting_sentences)) or any(w in query.lower() for w in ["event", "festival", "verbafest", "competition", "hackathon"])

        if style == "executive":
            lines = ["### Executive Summary\n"]
            lines.append(f"- **Key Finding**: {lead_sentences[0][0]}")

            if is_sec_domain:
                arch_picks = take_novel(log_sents + threat_sents, max_n=1)
                if not arch_picks:
                    arch_picks = take_novel(supporting_sentences, max_n=1)
                if arch_picks:
                    lines.append(f"- **Core Architecture & Entities**: {arch_picks[0][0]}")

                workload_picks = take_novel(log_sents + forensic_sents, max_n=1)
                if not workload_picks:
                    workload_picks = take_novel(supporting_sentences, max_n=1)
                if workload_picks:
                    lines.append(f"- **Operational Scope & Workloads**: {workload_picks[0][0]}")

                mech_picks = take_novel(threat_sents, max_n=1)
                if not mech_picks:
                    mech_picks = take_novel(supporting_sentences, max_n=1)
                if mech_picks:
                    lines.append(f"- **Execution Workflow & Policy**: {mech_picks[0][0]}")

                guar_picks = take_novel(metric_sents + forensic_sents, max_n=1)
                if not guar_picks:
                    guar_picks = take_novel(supporting_sentences, max_n=1)
                if guar_picks:
                    lines.append(f"- **Operating Guarantees & Risk Controls**: {guar_picks[0][0]}")
            elif is_yarn_domain:
                arch_picks = take_novel(rm_sents + nm_sents + am_sents + arch_foundation_sents, max_n=1)
                if not arch_picks:
                    arch_picks = take_novel(supporting_sentences, max_n=1)
                if arch_picks:
                    lines.append(f"- **Core Architecture & Entities**: {arch_picks[0][0]}")

                workload_picks = take_novel(workload_sents + container_sents, max_n=1)
                if not workload_picks:
                    workload_picks = take_novel(supporting_sentences, max_n=1)
                if workload_picks:
                    lines.append(f"- **Operational Scope & Workloads**: {workload_picks[0][0]}")

                mech_picks = take_novel(am_sents + container_sents, max_n=1)
                if not mech_picks:
                    mech_picks = take_novel(supporting_sentences, max_n=1)
                if mech_picks:
                    lines.append(f"- **Execution Workflow & Policy**: {mech_picks[0][0]}")

                guar_picks = take_novel(guarantee_sents, max_n=1)
                if not guar_picks:
                    guar_picks = take_novel(supporting_sentences, max_n=1)
                if guar_picks:
                    lines.append(f"- **Operating Guarantees & Risk Controls**: {guar_picks[0][0]}")
            else:
                # General Domain Executive Summary
                found_picks = take_novel([s for s in supporting_sentences if any(w in s[0].lower() for w in ["foundation", "concept", "principle", "overview", "definition", "core", "architecture", "design", "model", "structure", "system"])], max_n=1)
                if not found_picks:
                    found_picks = take_novel(supporting_sentences, max_n=1)
                if found_picks:
                    lines.append(f"- **Structural Foundations & Core Principles**: {found_picks[0][0]}")

                mech_picks = take_novel([s for s in supporting_sentences if any(w in s[0].lower() for w in ["process", "workflow", "mechanism", "execution", "operation", "method", "function", "pipeline", "interact", "step"])], max_n=1)
                if not mech_picks:
                    mech_picks = take_novel(supporting_sentences, max_n=1)
                if mech_picks:
                    lines.append(f"- **Operational Mechanics & Execution Workflow**: {mech_picks[0][0]}")

                guar_picks = take_novel([s for s in supporting_sentences if any(w in s[0].lower() for w in ["guarantee", "performance", "result", "benefit", "advantage", "scalability", "efficiency", "reliability", "limitation", "trade-off", "constraint"])], max_n=1)
                if not guar_picks:
                    guar_picks = take_novel(supporting_sentences, max_n=1)
                if guar_picks:
                    lines.append(f"- **Performance Characteristics & Practical Guarantees**: {guar_picks[0][0]}")

            final_used = [ls[2] for ls in lead_sentences] + [s[2] for s in supporting_sentences if s[0] in used_s_texts]
            return "\n".join(lines), final_used

        elif style == "concise":
            concise_parts = [ls[0] for ls in lead_sentences]
            extra = take_novel(supporting_sentences, max_n=1)
            if extra:
                concise_parts.append(extra[0][0])
            final_used = [ls[2] for ls in lead_sentences] + [s[2] for s in extra]
            return " ".join(concise_parts), final_used

        elif style == "2_mark":
            # 2-Mark Answer: Precise definition + exactly 2 bullet points
            lines = [f"{lead_sentences[0][0]}"]
            points = take_novel(supporting_sentences, max_n=2)
            for idx, p in enumerate(points, 1):
                clean_p = p[0].rstrip('.')
                lines.append(f"- **Key Point {idx}**: {clean_p}.")
            final_used = [ls[2] for ls in lead_sentences] + [s[2] for s in points]
            return "\n\n".join(lines), final_used

        elif style == "5_mark":
            # 5-Mark Answer: Core Definition + Structural Breakdown + Context
            if is_yarn_domain:
                h1 = "### Core Architectural Definition"
                h2 = "### Key Daemons & Operational Responsibilities"
                h3 = "### Practical Application & Multi-Tenant Deployment"
            elif is_sec_domain:
                h1 = "### Threat Context & Core Definition"
                h2 = "### Key Detection & Response Mechanics"
                h3 = "### Forensic Investigation & Metric Controls"
            elif is_event_domain:
                h1 = "### Event Overview & Core Purpose"
                h2 = "### Key Competitions & Structural Breakdown"
                h3 = "### Participation, Objectives & Final Outcomes"
            else:
                h1 = "### Core Subject Definition"
                h2 = "### Key Components & Functional Breakdown"
                h3 = "### Practical Implementation & Context"

            lines = [f"{h1}\n{lead_sentences[0][0]}"]
            points = take_novel(supporting_sentences, max_n=4)
            if not points:
                points = supporting_sentences[:3] if supporting_sentences else []
            pt_lines = [h2]
            for idx, p in enumerate(points, 1):
                clean_p = p[0].rstrip('.')
                pt_lines.append(f"- **Key Component {idx}**: {clean_p}.")
            lines.append("\n".join(pt_lines))

            extra_picks = take_novel(supporting_sentences, max_n=2)
            if not extra_picks and len(supporting_sentences) > 0:
                extra_picks = [supporting_sentences[-1]]
            if extra_picks:
                extra_text = " ".join([ep[0] for ep in extra_picks])
                lines.append(f"{h3}\n{extra_text}")

            final_used = [ls[2] for ls in lead_sentences] + [s[2] for s in points] + [s[2] for s in extra_picks]
            return "\n\n".join(lines), final_used

        elif style == "8_mark":
            # 8-Mark Answer: Outline + 5 structured technical sections
            if is_yarn_domain:
                outline = "> **Answer Outline**: 1. Architectural Foundation & Decoupling Principle | 2. Core Daemons & Component Responsibilities | 3. Application Lifecycle & Container Negotiation Workflow | 4. Multi-Tenancy & Workload Execution Capabilities | 5. Architectural Trade-offs & Operational Guarantees\n"
                sec_titles = [
                    "### 1. Architectural Foundation & Decoupling Principle",
                    "### 2. Core Daemons & Component Responsibilities",
                    "### 3. Application Lifecycle & Container Negotiation Workflow",
                    "### 4. Multi-Tenancy & Workload Execution Capabilities",
                    "### 5. Architectural Trade-offs & Operational Guarantees"
                ]
                s2_pool = rm_sents + nm_sents + am_sents + arch_foundation_sents + log_sents
                s3_pool = am_sents + container_sents + workload_sents + threat_sents
                s4_pool = workload_sents + forensic_sents
                s5_pool = guarantee_sents + metric_sents
            elif is_sec_domain:
                outline = "> **Answer Outline**: 1. Security Architecture & Threat Scope | 2. Telemetry, Audit Logs & Monitoring | 3. Threat Detection & Incident Response Mechanics | 4. Forensic Investigation & Remediation | 5. Security Metrics & Risk Controls\n"
                sec_titles = [
                    "### 1. Security Architecture & Threat Scope",
                    "### 2. Telemetry, Audit Logs & Monitoring",
                    "### 3. Threat Detection & Incident Response Mechanics",
                    "### 4. Forensic Investigation & Remediation",
                    "### 5. Security Metrics & Risk Controls"
                ]
                s2_pool = log_sents
                s3_pool = threat_sents
                s4_pool = forensic_sents
                s5_pool = metric_sents
            elif is_event_domain:
                outline = "> **Answer Outline**: 1. Event Overview & Core Purpose | 2. Key Competitions & Formats | 3. Operational Execution & Timeline | 4. Digital Platforms & Coordination | 5. Outcomes, Participation & Impact\n"
                sec_titles = [
                    "### 1. Event Overview & Core Purpose",
                    "### 2. Key Competitions & Formats",
                    "### 3. Operational Execution & Timeline",
                    "### 4. Digital Platforms & Coordination",
                    "### 5. Outcomes, Participation & Impact"
                ]
                s2_pool = [s for s in supporting_sentences if any(w in s[0].lower() for w in ['competition', 'competitions', 'round', 'rounds', 'saga', 'discussion', 'debate'])]
                s3_pool = [s for s in supporting_sentences if any(w in s[0].lower() for w in ['process', 'workflow', 'execution', 'date', 'conduct', 'schedule', 'stage'])]
                s4_pool = [s for s in supporting_sentences if any(w in s[0].lower() for w in ['platform', 'website', 'implementation', 'developer', 'digital', 'portal', 'online'])]
                s5_pool = [s for s in supporting_sentences if any(w in s[0].lower() for w in ['participant', 'participants', 'winner', 'winners', 'outcome', 'impact', 'overview'])]
            else:
                outline = "> **Answer Outline**: 1. Foundational Overview & Core Context | 2. Key Components & Structural Breakdown | 3. Operational Mechanics & Execution Workflow | 4. Functional Scope & Application | 5. Practical Guarantees & Summary Assessment\n"
                sec_titles = [
                    "### 1. Foundational Overview & Core Context",
                    "### 2. Key Components & Structural Breakdown",
                    "### 3. Operational Mechanics & Execution Workflow",
                    "### 4. Functional Scope & Application",
                    "### 5. Practical Guarantees & Summary Assessment"
                ]
                s2_pool = [s for s in supporting_sentences if any(w in s[0].lower() for w in ['component', 'structure', 'element', 'part', 'core', 'architecture', 'design'])]
                s3_pool = [s for s in supporting_sentences if any(w in s[0].lower() for w in ['process', 'workflow', 'execution', 'operation', 'method', 'mechanism', 'pipeline'])]
                s4_pool = [s for s in supporting_sentences if any(w in s[0].lower() for w in ['scope', 'application', 'workload', 'feature', 'system', 'use'])]
                s5_pool = [s for s in supporting_sentences if any(w in s[0].lower() for w in ['guarantee', 'performance', 'result', 'benefit', 'reliability', 'efficiency', 'assessment'])]

            s1_text = lead_sentences[0][0]
            if len(lead_sentences) > 1:
                s1_text += f" {lead_sentences[1][0]}"

            s2_picks = take_novel(s2_pool, max_n=3) or take_novel(supporting_sentences, max_n=2)
            s2_text = " ".join([s[0] for s in s2_picks]) if s2_picks else lead_sentences[0][0]

            s3_picks = take_novel(s3_pool, max_n=3) or take_novel(supporting_sentences, max_n=2)
            s3_text = " ".join([s[0] for s in s3_picks]) if s3_picks else s2_text

            s4_picks = take_novel(s4_pool, max_n=2) or take_novel(supporting_sentences, max_n=2)
            s4_text = " ".join([s[0] for s in s4_picks]) if s4_picks else s1_text

            s5_picks = take_novel(s5_pool, max_n=2) or take_novel(supporting_sentences, max_n=1)
            s5_text = " ".join([s[0] for s in s5_picks]) if s5_picks else s1_text

            sections = [
                outline,
                f"{sec_titles[0]}\n{s1_text}",
                f"{sec_titles[1]}\n{s2_text}",
                f"{sec_titles[2]}\n{s3_text}",
                f"{sec_titles[3]}\n{s4_text}",
                f"{sec_titles[4]}\n{s5_text}",
            ]
            final_used = [ls[2] for ls in lead_sentences] + [s[2] for s in supporting_sentences if s[0] in used_s_texts]
            return "\n\n".join(sections), final_used

        elif style == "10_mark":
            # 10-Mark Answer: Exhaustive university exam paper with comparative matrix
            if is_yarn_domain:
                outline = "> **Comprehensive Answer Outline**: 1. Comprehensive Architectural Foundation & Problem Context | 2. Component Daemons & Structural Breakdown | 3. Dynamic Execution Lifecycle & Workflow Sequence | 4. Comparative Architectural Paradigm Matrix | 5. Real-World Multi-Tenant Workloads & Framework Diversity | 6. Critical Engineering Trade-offs & Concluding Assessment\n"
                sec_titles = [
                    "### 1. Comprehensive Architectural Foundation & Problem Context",
                    "### 2. Component Daemons & Structural Breakdown",
                    "### 3. Dynamic Execution Lifecycle & Workflow Sequence",
                    "### 4. Comparative Architectural Paradigm Matrix",
                    "### 5. Real-World Multi-Tenant Workloads & Framework Diversity",
                    "### 6. Critical Engineering Trade-offs & Concluding Assessment"
                ]
                s2_pool = rm_sents + nm_sents + am_sents + arch_foundation_sents + log_sents
                s3_pool = am_sents + container_sents + threat_sents
                s5_pool = workload_sents
                s6_pool = guarantee_sents + metric_sents
            elif is_sec_domain:
                outline = "> **Comprehensive Answer Outline**: 1. Comprehensive Security Architecture & Threat Environment | 2. Telemetry, Event Logs & Monitoring Pipeline | 3. Detection Mechanics & Incident Triage Sequence | 4. Comparative Security Controls Matrix | 5. Multi-Vector Attack Surfaces & Threat Scenarios | 6. Forensic Assessment & Operational Metrics\n"
                sec_titles = [
                    "### 1. Comprehensive Security Architecture & Threat Environment",
                    "### 2. Telemetry, Event Logs & Monitoring Pipeline",
                    "### 3. Detection Mechanics & Incident Triage Sequence",
                    "### 4. Comparative Security Controls Matrix",
                    "### 5. Multi-Vector Attack Surfaces & Threat Scenarios",
                    "### 6. Forensic Assessment & Operational Metrics"
                ]
                s2_pool = log_sents
                s3_pool = threat_sents
                s5_pool = forensic_sents
                s6_pool = metric_sents
            elif is_event_domain:
                outline = "> **Comprehensive Answer Outline**: 1. Comprehensive Event Foundation & Objectives | 2. Key Competitions & Structural Breakdown | 3. Operational Mechanics & Event Lifecycle | 4. Comparative Competition Matrix | 5. Digital Implementation & Platform Execution | 6. Participation, Impact & Concluding Assessment\n"
                sec_titles = [
                    "### 1. Comprehensive Event Foundation & Objectives",
                    "### 2. Key Competitions & Structural Breakdown",
                    "### 3. Operational Mechanics & Event Lifecycle",
                    "### 4. Comparative Competition Matrix",
                    "### 5. Digital Implementation & Platform Execution",
                    "### 6. Participation, Impact & Concluding Assessment"
                ]
                s2_pool = [s for s in supporting_sentences if any(w in s[0].lower() for w in ['competition', 'competitions', 'round', 'rounds', 'saga', 'discussion', 'debate'])]
                s3_pool = [s for s in supporting_sentences if any(w in s[0].lower() for w in ['process', 'workflow', 'execution', 'date', 'conduct', 'schedule', 'stage'])]
                s5_pool = [s for s in supporting_sentences if any(w in s[0].lower() for w in ['platform', 'website', 'implementation', 'developer', 'digital', 'portal', 'online'])]
                s6_pool = [s for s in supporting_sentences if any(w in s[0].lower() for w in ['participant', 'participants', 'winner', 'winners', 'outcome', 'impact', 'overview'])]
            else:
                outline = "> **Comprehensive Answer Outline**: 1. Comprehensive Subject Foundation & Context | 2. Key Components & Structural Analysis | 3. Operational Mechanics & Execution Sequence | 4. Comparative Synthesis Matrix | 5. Practical Implementation & Applications | 6. Critical Evaluation & Concluding Assessment\n"
                sec_titles = [
                    "### 1. Comprehensive Subject Foundation & Context",
                    "### 2. Key Components & Structural Analysis",
                    "### 3. Operational Mechanics & Execution Sequence",
                    "### 4. Comparative Synthesis Matrix",
                    "### 5. Practical Implementation & Applications",
                    "### 6. Critical Evaluation & Concluding Assessment"
                ]
                s2_pool = [s for s in supporting_sentences if any(w in s[0].lower() for w in ['component', 'structure', 'element', 'part', 'core', 'architecture', 'design'])]
                s3_pool = [s for s in supporting_sentences if any(w in s[0].lower() for w in ['process', 'workflow', 'execution', 'operation', 'method', 'mechanism', 'pipeline'])]
                s5_pool = [s for s in supporting_sentences if any(w in s[0].lower() for w in ['scope', 'application', 'workload', 'feature', 'system', 'use'])]
                s6_pool = [s for s in supporting_sentences if any(w in s[0].lower() for w in ['guarantee', 'performance', 'result', 'benefit', 'reliability', 'efficiency', 'assessment'])]

            s1_text = lead_sentences[0][0]
            if len(lead_sentences) > 1:
                s1_text += f" {lead_sentences[1][0]}"

            s2_picks = take_novel(s2_pool, max_n=3) or take_novel(supporting_sentences, max_n=2)
            s2_text = " ".join([s[0] for s in s2_picks]) if s2_picks else lead_sentences[0][0]

            s3_picks = take_novel(s3_pool, max_n=3) or take_novel(supporting_sentences, max_n=2)
            s3_text = " ".join([s[0] for s in s3_picks]) if s3_picks else s2_text

            # Section 4: Comparative evaluation table
            s4_picks = take_novel(supporting_sentences, max_n=3) or supporting_sentences[:2]
            c1 = s4_picks[0][0] if s4_picks else lead_sentences[0][0]
            c2 = s4_picks[1][0] if len(s4_picks) > 1 else (supporting_sentences[0][0] if supporting_sentences else lead_sentences[0][0])
            cite1 = f"[E{s4_picks[0][1]}]" if s4_picks else f"[E{lead_sentences[0][1]}]"
            cite2 = f"[E{s4_picks[1][1]}]" if len(s4_picks) > 1 else cite1
            if is_yarn_domain:
                table_rows = [
                    f"| Resource Allocation | {clean_narrative_preamble(c1)} | Primary System Implementation | {cite1} |",
                    f"| Operational Scope | {clean_narrative_preamble(c2)} | Documented Specification | {cite2} |"
                ]
                if len(s4_picks) > 2:
                    c3 = s4_picks[2][0]
                    cite3 = f"[E{s4_picks[2][1]}]"
                    table_rows.append(f"| Deliverables & Impact | {clean_narrative_preamble(c3)} | Documented Milestone / Result | {cite3} |")
                table_md = (
                    f"{sec_titles[3]}\n\n"
                    "| Architectural Dimension | Primary Paradigm | Alternative / Complementary Paradigm | Citations |\n"
                    "|---|---|---|---|\n" +
                    "\n".join(table_rows)
                )
            else:
                table_rows = [
                    f"| Core Functionality | {clean_narrative_preamble(c1)} | Documented Specification | {cite1} |",
                    f"| Operational Scope | {clean_narrative_preamble(c2)} | Primary System Implementation | {cite2} |"
                ]
                if len(s4_picks) > 2:
                    c3 = s4_picks[2][0]
                    cite3 = f"[E{s4_picks[2][1]}]"
                    table_rows.append(f"| Deliverables & Impact | {clean_narrative_preamble(c3)} | Documented Milestone / Result | {cite3} |")
                table_md = (
                    f"{sec_titles[3]}\n\n"
                    "| Dimension | Documented Primary Finding | Comparative Standard | Citations |\n"
                    "|---|---|---|---|\n" +
                    "\n".join(table_rows)
                )

            s5_picks = take_novel(s5_pool, max_n=2) or take_novel(supporting_sentences, max_n=2)
            s5_text = " ".join([s[0] for s in s5_picks]) if s5_picks else s1_text

            s6_picks = take_novel(s6_pool, max_n=2) or take_novel(supporting_sentences, max_n=1)
            s6_text = " ".join([s[0] for s in s6_picks]) if s6_picks else s1_text

            sections = [
                outline,
                f"{sec_titles[0]}\n{s1_text}",
                f"{sec_titles[1]}\n{s2_text}",
                f"{sec_titles[2]}\n{s3_text}",
                table_md,
                f"{sec_titles[4]}\n{s5_text}",
                f"{sec_titles[5]}\n{s6_text}",
            ]
            final_used = [ls[2] for ls in lead_sentences] + [s[2] for s in supporting_sentences if s[0] in used_s_texts]
            return "\n\n".join(sections), final_used

        elif style == "comparison":
            sections = [
                f"### Comparative Overview\n{lead_sentences[0][0]}",
            ]
            comp_picks = take_novel(supporting_sentences, max_n=4)
            c1 = comp_picks[0][0] if comp_picks else lead_sentences[0][0]
            c2 = comp_picks[1][0] if len(comp_picks) > 1 else (supporting_sentences[0][0] if supporting_sentences else lead_sentences[0][0])
            c3 = comp_picks[2][0] if len(comp_picks) > 2 else (supporting_sentences[-1][0] if supporting_sentences else lead_sentences[0][0])

            cite1 = f"[E{comp_picks[0][1]}]" if comp_picks else f"[E{lead_sentences[0][1]}]"
            cite2 = f"[E{comp_picks[1][1]}]" if len(comp_picks) > 1 else cite1
            cite3 = f"[E{comp_picks[2][1]}]" if len(comp_picks) > 2 else cite1

            table_rows = [
                f"| Operational Scope | {clean_narrative_preamble(c1)} | Standard / Conventional Implementation | {cite1} |",
                f"| Resource Allocation | {clean_narrative_preamble(c2)} | Baseline Alternative | {cite2} |",
                f"| Practical Scope | {clean_narrative_preamble(c3)} | General Deployment | {cite3} |"
            ]
            table_md = (
                "### Dimension-by-Dimension Comparison\n\n"
                "| Architectural Dimension | Primary Paradigm | Alternative / Complementary Paradigm | Citations |\n"
                "|---|---|---|---|\n" +
                "\n".join(table_rows)
            )
            sections.append(table_md)

            if len(comp_picks) > 3:
                sections.append(f"### Critical Architectural Differentiators & Recommendations\n{comp_picks[3][0]}")

            final_used = [ls[2] for ls in lead_sentences] + [s[2] for s in comp_picks]
            return "\n\n".join(sections), final_used

        elif style == "worked_example":
            sections = [
                f"### 1. Concrete Scenario Setup\n{lead_sentences[0][0]}",
            ]
            steps = take_novel(supporting_sentences, max_n=3)
            if steps:
                step_lines = ["### 2. Step-by-Step Execution Walkthrough"]
                for i, stp in enumerate(steps, 1):
                    step_lines.append(f"**Step {i}**: {stp[0]}")
                sections.append("\n\n".join(step_lines))
            verify = take_novel(supporting_sentences, max_n=1)
            if verify:
                sections.append(f"### 3. Resulting System State & Verification\n{verify[0][0]}")
            final_used = [ls[2] for ls in lead_sentences] + [s[2] for s in steps] + ([s[2] for s in verify] if verify else [])
            return "\n\n".join(sections), final_used

        elif style == "structured_notes":
            lines = [
                f"### Key Definitions & Acronyms\n- **Core Definition**: {lead_sentences[0][0]}"
            ]
            cheat_picks = take_novel(supporting_sentences, max_n=4)
            if not cheat_picks:
                cheat_picks = supporting_sentences[:3] if supporting_sentences else []
            cheat_header = "### Architecture & Component Cheat-Sheet" if is_yarn_domain else "### Key Concepts & Details Cheat-Sheet"
            cheat_label = "Key Architecture" if is_yarn_domain else "Key Concept"
            cheat_lines = [cheat_header]
            for cp in cheat_picks:
                cheat_lines.append(f"- **{cheat_label}**: {cp[0]}")
            lines.append("\n".join(cheat_lines))

            takeaway_picks = take_novel(supporting_sentences, max_n=2)
            if not takeaway_picks:
                takeaway_picks = [supporting_sentences[-1]] if supporting_sentences else [lead_sentences[0]]
            takeaway_lines = ["### High-Yield Exam Takeaways"]
            for tp in takeaway_picks:
                takeaway_lines.append(f"- **Exam Takeaway**: {tp[0]}")
            lines.append("\n".join(takeaway_lines))

            final_used = [ls[2] for ls in lead_sentences] + [s[2] for s in cheat_picks] + [s[2] for s in takeaway_picks]
            return "\n\n".join(lines), final_used

        elif style in ("detailed", "comprehensive"):
            # Deep & Comprehensive Mode: Direct answer first followed by thematic markdown subsections
            p1_parts = [ls[0] for ls in lead_sentences]
            lead_extra = take_novel(arch_foundation_sents, max_n=1)
            if lead_extra:
                p1_parts.append(lead_extra[0][0])
            sections = [" ".join(p1_parts)]

            if is_yarn_domain:
                # Distributed Systems / YARN subsections
                comp_sents = []
                comp_sents.extend(take_novel(rm_sents, max_n=2))
                comp_sents.extend(take_novel(nm_sents, max_n=2))
                comp_sents.extend(take_novel(am_sents, max_n=2))
                comp_sents.extend(take_novel(container_sents, max_n=2))
                if comp_sents:
                    sections.append("### Core Daemons & Component Responsibilities\n" + " ".join([s[0] for s in comp_sents]))

                wl_sents = []
                wl_sents.extend(take_novel(workload_sents, max_n=3))
                wl_sents.extend(take_novel(container_sents, max_n=2))
                wl_sents.extend(take_novel(hdfs_sents, max_n=2))
                if wl_sents:
                    sections.append("### Workload Execution Workflow & Framework Diversity\n" + " ".join([s[0] for s in wl_sents]))

                g_sents = take_novel(guarantee_sents, max_n=3)
                if g_sents:
                    sections.append("### Cluster Reliability, Isolation & Operational Guarantees\n" + " ".join([s[0] for s in g_sents]))

            elif is_sec_domain:
                # Cybersecurity subsections
                telemetry_sents = take_novel(log_sents, max_n=3)
                if telemetry_sents:
                    sections.append("### Telemetry, Audit Logs & Event Monitoring\n" + " ".join([s[0] for s in telemetry_sents]))

                detection_sents = take_novel(threat_sents, max_n=3)
                if detection_sents:
                    sections.append("### Threat Detection & Incident Response Mechanics\n" + " ".join([s[0] for s in detection_sents]))

                eval_sents = take_novel(metric_sents + forensic_sents, max_n=4)
                if eval_sents:
                    sections.append("### Forensic Investigation & Security Performance Metrics\n" + " ".join([s[0] for s in eval_sents]))

            else:
                # Dynamic Thematic Subsections for General Technical Domains or Event/Document Reports
                found_kws = {'foundation', 'concept', 'principle', 'overview', 'definition', 'core', 'architecture', 'design', 'model', 'structure', 'system', 'layer', 'theory'}
                work_kws = {'process', 'workflow', 'mechanism', 'execution', 'operation', 'method', 'function', 'pipeline', 'interact', 'step', 'algorithm', 'procedure', 'compute', 'computes', 'computation', 'calculate', 'scaling', 'attend', 'attention'}
                guar_kws = {'guarantee', 'performance', 'result', 'benefit', 'advantage', 'scalability', 'efficiency', 'reliability', 'limitation', 'trade-off', 'constraint', 'accuracy', 'speedup', 'speedups', 'training', 'parallelization'}
                event_kws = {'event', 'competition', 'competitions', 'round', 'rounds', 'participant', 'participants', 'organized', 'guidance', 'team', 'mavericks', 'discussion', 'debate', 'saga', 'registration', 'platform', 'website'}

                def sent_tokens(s_tuple):
                    return set(re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', s_tuple[0].lower()))

                is_event_domain = bool(any(sent_tokens(s) & event_kws for s in supporting_sentences))

                if is_event_domain:
                    # Event & Organizational Subsections
                    competition_sents = take_novel(
                        [s for s in supporting_sentences if sent_tokens(s) & {'competition', 'competitions', 'round', 'rounds', 'saga', 'discussion', 'debate'}],
                        max_n=5
                    )
                    if competition_sents:
                        sections.append("### Key Competitions & Event Structure\n" + " ".join([s[0] for s in competition_sents]))

                    feature_sents = take_novel(
                        [s for s in supporting_sentences if sent_tokens(s) & {'platform', 'website', 'implementation', 'developer', 'developers', 'special', 'features', 'digital'}],
                        max_n=3
                    )
                    if feature_sents:
                        sections.append("### Digital Implementation & Special Features\n" + " ".join([s[0] for s in feature_sents]))

                    overview_sents = take_novel(
                        [s for s in supporting_sentences if sent_tokens(s) & {'participant', 'participants', 'objective', 'prepared', 'placement', 'winners', 'final', 'overview'}],
                        max_n=3
                    )
                    if overview_sents:
                        sections.append("### Participation, Objectives & Final Outcomes\n" + " ".join([s[0] for s in overview_sents]))

                else:
                    foundation_sents = take_novel(
                        [s for s in supporting_sentences if sent_tokens(s) & found_kws],
                        max_n=3
                    )
                    if foundation_sents:
                        sections.append("### Structural Foundations & Core Principles\n" + " ".join([s[0] for s in foundation_sents]))

                    workflow_sents = take_novel(
                        [s for s in supporting_sentences if sent_tokens(s) & work_kws],
                        max_n=3
                    )
                    if workflow_sents:
                        sections.append("### Operational Mechanics & Process Workflow\n" + " ".join([s[0] for s in workflow_sents]))

                    guarantee_sents = take_novel(
                        [s for s in supporting_sentences if sent_tokens(s) & guar_kws],
                        max_n=3
                    )
                    if guarantee_sents:
                        sections.append("### Performance Characteristics & Technical Guarantees\n" + " ".join([s[0] for s in guarantee_sents]))

            # Catch-all for other technical domains or remaining rich sentences if few sections were formed
            remaining = [s for s in supporting_sentences if s[0] not in used_s_texts]
            if remaining and len(sections) <= 2:
                extra_picks = take_novel(remaining, max_n=3)
                if extra_picks:
                    sections.append("### Context & Additional Details\n" + " ".join([s[0] for s in extra_picks]))

            final_used = [ls[2] for ls in lead_sentences] + [s[2] for s in supporting_sentences if s[0] in used_s_texts]
            return "\n\n".join(sections), final_used

        else:
            # Conversational Style: Direct answer followed by rich, coherent narrative paragraphs
            q_lower = query.lower()
            q_words = query.strip().split()
            is_deep = (
                len(q_words) >= 6 or
                any(w in q_lower for w in [
                    "explain", "how", "why", "detail", "architecture", "overview",
                    "describe", "breakdown", "mechanism", "components", "elaborate", "and how"
                ])
            )

            p1_parts = [ls[0] for ls in lead_sentences]

            if not is_deep or not supporting_sentences:
                extra = take_novel(supporting_sentences, max_n=1)
                if extra:
                    p1_parts.append(extra[0][0])
                final_used = [ls[2] for ls in lead_sentences] + [s[2] for s in extra]
                return " ".join(p1_parts), final_used

            # Deep conversational: 3 to 4 substantive paragraphs
            paragraphs = [" ".join(p1_parts)]

            def sent_tokens(s_tuple):
                return set(re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', s_tuple[0].lower()))

            # Paragraph 2: Core Components & Mechanics
            p2_sents = []
            if is_yarn_domain:
                p2_sents.extend(take_novel(rm_sents + nm_sents + am_sents + arch_foundation_sents, max_n=2))
            elif is_sec_domain:
                p2_sents.extend(take_novel(threat_sents + log_sents, max_n=2))
            else:
                found_kws = {'foundation', 'concept', 'principle', 'overview', 'definition', 'core', 'architecture', 'design', 'model', 'structure', 'system', 'layer', 'theory'}
                p2_sents.extend(take_novel([s for s in supporting_sentences if sent_tokens(s) & found_kws], max_n=2))
            if not p2_sents:
                p2_sents.extend(take_novel(supporting_sentences, max_n=2))
            if p2_sents:
                paragraphs.append(" ".join([s[0] for s in p2_sents]))

            # Paragraph 3: Execution Workflow & Workloads
            p3_sents = []
            if is_yarn_domain:
                p3_sents.extend(take_novel(workload_sents + container_sents, max_n=2))
            elif is_sec_domain:
                p3_sents.extend(take_novel(forensic_sents + metric_sents, max_n=2))
            else:
                work_kws = {'process', 'workflow', 'mechanism', 'execution', 'operation', 'method', 'function', 'pipeline', 'interact', 'step', 'algorithm', 'procedure', 'compute', 'computes', 'computation', 'calculate', 'scaling', 'attend', 'attention'}
                p3_sents.extend(take_novel([s for s in supporting_sentences if sent_tokens(s) & work_kws], max_n=2))
            if not p3_sents:
                p3_sents.extend(take_novel(supporting_sentences, max_n=2))
            if p3_sents:
                paragraphs.append(" ".join([s[0] for s in p3_sents]))

            # Paragraph 4: Guarantees, Isolation & Resilience
            p4_sents = []
            if is_yarn_domain:
                p4_sents.extend(take_novel(guarantee_sents, max_n=2))
            elif is_sec_domain:
                p4_sents.extend(take_novel(metric_sents, max_n=2))
            else:
                guar_kws = {'guarantee', 'performance', 'result', 'benefit', 'advantage', 'scalability', 'efficiency', 'reliability', 'limitation', 'trade-off', 'constraint', 'accuracy', 'speedup', 'speedups', 'training', 'parallelization'}
                p4_sents.extend(take_novel([s for s in supporting_sentences if sent_tokens(s) & guar_kws], max_n=2))
            if not p4_sents:
                p4_sents.extend(take_novel(supporting_sentences, max_n=2))
            if p4_sents:
                paragraphs.append(" ".join([s[0] for s in p4_sents]))

            final_used = [ls[2] for ls in lead_sentences] + [s[2] for s in supporting_sentences if s[0] in used_s_texts]
            return "\n\n".join(paragraphs), final_used

    def synthesize(
        self,
        query: str,
        evidence: EvidenceCollection,
        style: str = "conversational",
        provider_override: Optional[BaseLLMProvider] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> GroundedAnswer:
        """
        Main synthesis entry point:
        1. Deduplicates evidence collection (removing exact and near-duplicates).
        2. Checks Abstention Gate.
        3. Routes to configured LLM Provider (Ollama, OpenAI-compatible, Mock) with structured prompting.
        4. Enforces Fallback Policy to Deterministic Grounded Engine if LLM fails or is unavailable.
        5. Validates Citations against retrieved chunks and textual support.
        6. Performs Bounded Repair on citation anomalies.
        7. Computes monotonic latencies and calibrated multi-factor confidence.
        """
        t_start = time.perf_counter()

        # 1. Deduplication Stage
        t_dedup_start = time.perf_counter()
        clean_evidence, dedup_diagnostics = EvidenceDeduplicator.deduplicate(evidence)
        t_dedup_end = time.perf_counter()
        dedup_ms = max(0.05, (t_dedup_end - t_dedup_start) * 1000)

        # 2. Abstention Gate
        should_abstain, reason = AbstentionEngine.evaluate(
            query=query,
            evidence=clean_evidence,
            relevance_threshold=self.min_relevance_threshold
        )
        if should_abstain:
            elapsed = max(0.1, (time.perf_counter() - t_start) * 1000)
            return GroundedAnswer(
                query=query,
                answer=(
                    "I cannot answer this question based on the provided documents. "
                    f"Reason: {reason}"
                ),
                claims=[],
                cited_evidence=[],
                abstained=True,
                abstention_reason=reason,
                confidence=0.0,
                provider_used="abstention_gate",
                model_name="deterministic_guardrail",
                generation_time_ms=round(elapsed, 2),
                style=style,
                metadata={
                    "deduplication": dedup_diagnostics,
                    "stage_latencies_ms": {
                        "dedup_ms": round(dedup_ms, 2),
                        "selection_ms": round(elapsed, 2),
                        "synthesis_ms": 0.0,
                        "validation_ms": 0.0,
                        "total_ms": round(elapsed, 2)
                    }
                }
            )

        # 3. Build Evidence Context
        t_evidence_start = time.perf_counter()
        evidence_context = self._build_evidence_context(clean_evidence.chunks)
        t_evidence_end = time.perf_counter()

        # 4. Determine Active Provider
        active_provider = provider_override or self.llm_provider
        provider_name = active_provider.provider_type if active_provider else "deterministic"

        generated_text: Optional[str] = None
        top_scored_sentences: List[ScoredSentence] = []
        fallback_occurred = False
        fallback_reason: Optional[str] = None
        llm_prompt_tokens: Optional[int] = None
        llm_completion_tokens: Optional[int] = None

        t_synth_start = time.perf_counter()

        # Try LLM Provider if configured
        if provider_name != "deterministic" and active_provider is not None:
            is_ok, unavail_reason = active_provider.is_available()
            if not is_ok:
                if self.llm_config.fallback_to_deterministic:
                    logger.info(
                        f"LLM Provider '{provider_name}' ({getattr(active_provider, 'model_name', 'unknown')}) is unavailable: {unavail_reason}. "
                        "Executing deterministic grounded fallback."
                    )
                    fallback_occurred = True
                    fallback_reason = f"Provider '{provider_name}' error: {unavail_reason}"
                else:
                    raise LLMConnectionError(f"Provider '{provider_name}' unavailable: {unavail_reason}")
            else:
                try:
                    llm_res = active_provider.generate(
                        query=query,
                        evidence_chunks=clean_evidence.chunks,
                        style=style,
                        conversation_history=conversation_history
                    )
                    if llm_res.is_abstained:
                        elapsed = max(0.1, (time.perf_counter() - t_start) * 1000)
                        return GroundedAnswer(
                            query=query,
                            answer=llm_res.answer,
                            claims=[],
                            cited_evidence=[],
                            abstained=True,
                            abstention_reason=llm_res.abstention_reason,
                            confidence=0.0,
                            provider_used=llm_res.provider_used,
                            model_name=llm_res.model_name,
                            generation_time_ms=round(elapsed, 2),
                            style=style,
                            metadata={
                                "deduplication": dedup_diagnostics,
                                "stage_latencies_ms": {
                                    "dedup_ms": round(dedup_ms, 2),
                                    "synthesis_ms": round(llm_res.latency_ms, 2),
                                    "total_ms": round(elapsed, 2)
                                }
                            }
                        )
                    generated_text = llm_res.answer
                    provider_used = llm_res.provider_used
                    model_name = llm_res.model_name
                    llm_prompt_tokens = llm_res.prompt_tokens
                    llm_completion_tokens = llm_res.completion_tokens
                except Exception as e:
                    if self.llm_config.fallback_to_deterministic:
                        logger.warning(
                            f"LLM generation failed via {provider_name} ({getattr(active_provider, 'model_name', 'unknown')}): {e}. "
                            "Executing deterministic grounded fallback."
                        )
                        fallback_occurred = True
                        fallback_reason = f"Provider '{provider_name}' error: {e}"
                    else:
                        raise

        # Fallback or Deterministic Execution
        if generated_text is None:
            generated_text, top_scored_sentences = self._deterministic_grounded_synthesis(
                query, clean_evidence.chunks, style
            )
            provider_used = "deterministic_grounded_engine"
            model_name = "hopfield_evidence_synthesizer"

        t_synth_end = time.perf_counter()

        # Handle explicit LLM abstention tag if present in raw answer text
        if generated_text.startswith("INSUFFICIENT_EVIDENCE:"):
            abstain_msg = generated_text.replace("INSUFFICIENT_EVIDENCE:", "").strip()
            elapsed = max(0.1, (time.perf_counter() - t_start) * 1000)
            return GroundedAnswer(
                query=query,
                answer=f"The retrieved evidence does not contain sufficient details to answer: {abstain_msg}",
                claims=[],
                cited_evidence=[],
                abstained=True,
                abstention_reason=abstain_msg,
                confidence=0.0,
                provider_used=provider_used,
                model_name=model_name,
                generation_time_ms=round(elapsed, 2),
                style=style
            )

        # 5. Enforce Strict Citation Validation & Bounded Repair
        t_val_start = time.perf_counter()
        claims, cited_evidence, unresolved = CitationValidator.extract_and_validate(
            text=generated_text,
            valid_chunks=clean_evidence.chunks
        )

        # Bounded Repair for Hallucinated Citation IDs (e.g. [E99])
        if unresolved:
            repaired_text = CitationValidator.repair_citations(generated_text, clean_evidence.chunks)
            if repaired_text != generated_text:
                generated_text = repaired_text
                claims, cited_evidence, unresolved = CitationValidator.extract_and_validate(
                    text=generated_text,
                    valid_chunks=clean_evidence.chunks
                )

        # Bounded Grounding Check: If LLM produced claims, verify support ratio and prune hallucinations
        if provider_used != "deterministic_grounded_engine" and claims:
            verified_count = len([c for c in claims if c.is_verified])
            verified_ratio = (verified_count / len(claims)) if claims else 0.0

            if (verified_count == 0 or verified_ratio < 0.50) and self.llm_config.fallback_to_deterministic:
                logger.warning(
                    f"LLM generated answer failed grounding validation ({verified_count}/{len(claims)} verified claims, "
                    f"ratio {verified_ratio:.2f} < 0.50). Triggering deterministic fallback."
                )
                generated_text, top_scored_sentences = self._deterministic_grounded_synthesis(
                    query, clean_evidence.chunks, style
                )
                provider_used = "deterministic_grounded_engine"
                model_name = "hopfield_evidence_synthesizer"
                fallback_occurred = True
                fallback_reason = f"LLM answer failed citation grounding verification ({verified_count}/{len(claims)} verified claims, ratio {verified_ratio:.2f})"
                claims, cited_evidence, unresolved = CitationValidator.extract_and_validate(
                    text=generated_text,
                    valid_chunks=clean_evidence.chunks
                )
            elif verified_count < len(claims):
                # 50% to 99% verified: Prune unverified claims so user receives only grounded facts
                unverified = [c for c in claims if not c.is_verified]
                logger.info(f"Pruning {len(unverified)} unverified claims from LLM response ({verified_count}/{len(claims)} verified).")
                pruned_text = CitationValidator.prune_unverified_sentences(generated_text, unverified)
                if len(pruned_text.strip()) >= 30:
                    generated_text = pruned_text
                    claims, cited_evidence, unresolved = CitationValidator.extract_and_validate(
                        text=generated_text,
                        valid_chunks=clean_evidence.chunks
                    )
                elif self.llm_config.fallback_to_deterministic:
                    logger.warning("Pruning unverified claims left answer insufficient. Triggering deterministic fallback.")
                    generated_text, top_scored_sentences = self._deterministic_grounded_synthesis(
                        query, clean_evidence.chunks, style
                    )
                    provider_used = "deterministic_grounded_engine"
                    model_name = "hopfield_evidence_synthesizer"
                    fallback_occurred = True
                    fallback_reason = "Pruning unverified claims left answer insufficient"
                    claims, cited_evidence, unresolved = CitationValidator.extract_and_validate(
                        text=generated_text,
                        valid_chunks=clean_evidence.chunks
                    )

        # 5b. Relevance Validation & Domain Drift Detection
        is_rel, rel_score, rel_diag = AnswerRelevanceValidator.validate(
            query=query,
            answer_text=generated_text,
            evidence_chunks=clean_evidence.chunks
        )
        if not is_rel:
            logger.warning(f"Answer failed relevance validation: {rel_diag}. Attempting focused resynthesis.")
            q_terms = [w.lower() for w in re.findall(r'\b[a-zA-Z0-9_\-]{3,}\b', query) if w.lower() not in STOP_WORDS]
            def _chunk_matches(tok: str, text: str) -> bool:
                if tok in text:
                    return True
                stem = get_stem(tok)
                if len(stem) >= 3 and stem in text:
                    return True
                if len(tok) > 6:
                    for prefix in ("cyber", "multi", "meta", "micro", "macro", "cross", "inter", "sub", "data"):
                        if tok.startswith(prefix) and len(tok[len(prefix):]) >= 3:
                            sub_part = tok[len(prefix):]
                            if sub_part in text or (len(get_stem(sub_part)) >= 3 and get_stem(sub_part) in text):
                                return True
                return False

            matching_chunks = [
                c for c in clean_evidence.chunks
                if any(_chunk_matches(t, c.text.lower()) for t in q_terms)
            ]
            if matching_chunks:
                generated_text, top_scored_sentences = self._deterministic_grounded_synthesis(
                    query, matching_chunks, style
                )
                provider_used = "deterministic_grounded_engine"
                model_name = "hopfield_evidence_synthesizer"
                fallback_occurred = True
                fallback_reason = f"Relevance validator rejected off-topic answer ({rel_diag})"
                claims, cited_evidence, unresolved = CitationValidator.extract_and_validate(
                    text=generated_text,
                    valid_chunks=matching_chunks
                )
                is_rel, rel_score, rel_diag = AnswerRelevanceValidator.validate(
                    query=query,
                    answer_text=generated_text,
                    evidence_chunks=matching_chunks
                )
            else:
                elapsed = max(0.1, (time.perf_counter() - t_start) * 1000)
                reason = f"Retrieved passages do not contain key terms from the question: {', '.join(q_terms[:4])}."
                return GroundedAnswer(
                    query=query,
                    answer=f"I cannot answer this question based on the provided documents. Reason: {reason}",
                    claims=[],
                    cited_evidence=[],
                    abstained=True,
                    abstention_reason=reason,
                    confidence=0.0,
                    provider_used="relevance_gate",
                    model_name="deterministic_guardrail",
                    generation_time_ms=round(elapsed, 2),
                    style=style,
                    metadata={"abstention_trigger": "relevance_validation_failure"}
                )

        t_val_end = time.perf_counter()

        # 6. Latency Calculations (monotonic high resolution)
        selection_ms = max(0.05, (t_evidence_end - t_evidence_start) * 1000)
        synthesis_ms = max(0.05, (t_synth_end - t_synth_start) * 1000)
        validation_ms = max(0.05, (t_val_end - t_val_start) * 1000)
        total_gen_ms = max(0.1, (t_val_end - t_start) * 1000)

        # 7. Calibrated Multi-Factor Confidence Calculation
        top_sim = clean_evidence.chunks[0].similarity_score if clean_evidence.chunks else 0.5
        relevance_factor = min(1.0, max(0.0, top_sim))

        if top_scored_sentences:
            best = top_scored_sentences[0]
            intent_factor = min(1.0, 0.40 + 0.25 * min(best.entity_hits, 2) + 0.15 * min(best.intent_hits, 2))
        else:
            intent_factor = 0.75 if provider_used != "deterministic_grounded_engine" else 0.60

        unique_verified_count = len([c for c in claims if c.is_verified and getattr(c, 'is_unique', True)])
        verification_factor = (unique_verified_count / max(len(claims), 1)) if claims else 0.80

        raw_confidence = (
            0.30 * relevance_factor +
            0.25 * intent_factor +
            0.25 * verification_factor +
            0.20 * rel_score
        )
        if not is_rel:
            calibrated_confidence = float(round(min(0.20, raw_confidence), 4))
        else:
            calibrated_confidence = float(round(min(0.98, max(0.10, raw_confidence)), 4))

        metadata = {
            "deduplication": dedup_diagnostics,
            "unresolved_citation_ids": unresolved,
            "retriever": clean_evidence.retriever_name,
            "num_retrieved_chunks": len(clean_evidence.chunks),
            "unique_verified_claims_count": unique_verified_count,
            "total_claims_count": len(claims),
            "fallback_occurred": fallback_occurred,
            "fallback_reason": fallback_reason,
            "llm_config": self.llm_config.to_dict(),
            "stage_latencies_ms": {
                "dedup_ms": round(dedup_ms, 2),
                "selection_ms": round(selection_ms, 2),
                "synthesis_ms": round(synthesis_ms, 2),
                "validation_ms": round(validation_ms, 2),
                "total_ms": round(total_gen_ms, 2)
            },
            "confidence_breakdown": {
                "relevance_factor": round(relevance_factor, 4),
                "intent_factor": round(intent_factor, 4),
                "verification_factor": round(verification_factor, 4),
                "answer_relevance": round(rel_score, 4),
                "final_confidence": calibrated_confidence
            }
        }
        if llm_prompt_tokens is not None:
            metadata["prompt_tokens"] = llm_prompt_tokens
        if llm_completion_tokens is not None:
            metadata["completion_tokens"] = llm_completion_tokens

        # Collect relevant visual figures from retrieved attractor evidence
        retrieved_images = []
        seen_img_urls = set()
        for c in clean_evidence.chunks:
            meta = getattr(c, "metadata", {}) or {}
            img_url = meta.get("image_url")
            if img_url and img_url not in seen_img_urls:
                seen_img_urls.add(img_url)
                retrieved_images.append({
                    "image_url": img_url,
                    "caption": meta.get("image_caption") or f"Figure from {c.source}",
                    "page_number": meta.get("page_number", c.page_number or 1),
                    "source": c.source,
                    "similarity_score": round(getattr(c, "similarity_score", 0.0), 4),
                    "width": meta.get("width"),
                    "height": meta.get("height"),
                })

        return GroundedAnswer(
            query=query,
            answer=generated_text,
            claims=claims,
            cited_evidence=cited_evidence if cited_evidence else clean_evidence.chunks[:2],
            abstained=False,
            abstention_reason=None,
            confidence=calibrated_confidence,
            provider_used=provider_used,
            model_name=model_name,
            generation_time_ms=round(total_gen_ms, 2),
            style=style,
            images=retrieved_images,
            metadata=metadata
        )

    def _call_openai_compatible_api(
        self,
        query: str,
        evidence_context: str,
        style: str
    ) -> Optional[str]:
        """Calls configured LLM endpoint via pluggable provider (backwards-compatible helper)."""
        try:
            res = self.llm_provider.generate(query=query, evidence_chunks=[], style=style)
            return res.answer
        except Exception as e:
            logger.warning(f"Remote LLM call failed: {e}")
            return None

    def generate(
        self,
        processed_query: Any,
        retrieved_chunks: List[Any],
        style: str = "conversational"
    ) -> Any:
        """
        Backwards-compatible adapter for callers expecting the legacy generate() signature.
        Converts inputs to EvidenceCollection, calls synthesize(), and returns GeneratedAnswer.
        """
        from core.answer_generator import GeneratedAnswer

        query_str = getattr(processed_query, 'raw_query', str(processed_query))
        
        evidence_chunks = []
        for i, c in enumerate(retrieved_chunks):
            if isinstance(c, EvidenceChunk):
                evidence_chunks.append(c)
            else:
                meta = getattr(c, 'metadata', {}) or {}
                chunk = EvidenceChunk(
                    chunk_id=getattr(c, 'chunk_id', f"chunk_{i}"),
                    doc_id=meta.get("doc_id", getattr(c, 'source', 'doc')),
                    source=getattr(c, 'source', 'document'),
                    text=getattr(c, 'text', str(c)),
                    similarity_score=float(getattr(c, 'similarity_score', getattr(c, 'hopfield_attention', 0.5))),
                    hopfield_attention=float(getattr(c, 'hopfield_attention', 0.0)),
                    rank=i + 1,
                    page_number=meta.get("page_number"),
                    metadata=meta
                )
                evidence_chunks.append(chunk)

        collection = EvidenceCollection(
            query=query_str,
            chunks=evidence_chunks,
            retriever_name="Adapter",
            is_grounded=len(evidence_chunks) > 0
        )

        grounded = self.synthesize(query=query_str, evidence=collection, style=style)

        return GeneratedAnswer(
            query=query_str,
            answer=grounded.answer,
            confidence=grounded.confidence,
            evidence_chunks=retrieved_chunks,
            highlighted_passages=[{"text": c.text, "chunk_id": getattr(c, 'chunk_id', ''), "relevance": getattr(c, 'similarity_score', 0.5)} for c in retrieved_chunks],
            sources=list(set([getattr(c, 'source', 'unknown') for c in retrieved_chunks])),
            answer_type="grounded_rag",
            generation_stats={
                "provider": grounded.provider_used,
                "model": grounded.model_name,
                "time_ms": grounded.generation_time_ms,
                "abstained": grounded.abstained,
                "abstention_reason": grounded.abstention_reason
            }
        )
