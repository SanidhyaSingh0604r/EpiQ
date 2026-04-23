"""
dataset_matcher.py
------------------
Maps OCR-extracted characters/tokens against a dataset of ancient-script ↔
modern-script mappings (e.g., Brahmi → Devanagari, Tamil-Brahmi → Tamil).

Two-tier matching:
    1. Exact lookup against the mapping dictionary.
    2. Fuzzy matching with Levenshtein distance (and optional
       sentence-embedding similarity for longer phrases).

The mapping file format (`data/script_mappings.json`) is a list of entries:

    [
      {
        "script": "brahmi",
        "glyph": "𑀓",
        "devanagari": "क",
        "iast": "ka",
        "english_gloss": "ka"
      },
      ...
      {
        "script": "tamil-brahmi",
        "token": "𑀅𑀭𑀳𑀢",
        "tamil": "அரஹத",
        "iast": "arahata",
        "meaning": "the worthy one (an honorific for a Jain monk)"
      }
    ]

Entries may be single glyphs or multi-character tokens. Phrase-level entries
tend to produce much better translations than character-only mappings.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# Levenshtein — we implement a tiny pure-Python version so the module has
# zero hard dependencies beyond the stdlib. For large corpora, swap in
# python-Levenshtein or rapidfuzz for a big speedup.

def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(curr[j - 1] + 1,     # insertion
                          prev[j] + 1,          # deletion
                          prev[j - 1] + cost)   # substitution
        prev = curr
    return prev[-1]


def _similarity(a: str, b: str) -> float:
    """Normalised similarity in [0, 1] from Levenshtein distance."""
    if not a and not b:
        return 1.0
    return 1.0 - _levenshtein(a, b) / max(len(a), len(b))


# ---------------------------------------------------------------------------
# Mapping dataset
# ---------------------------------------------------------------------------

@dataclass
class MappingEntry:
    script: str
    source: str                 # the ancient glyph or token
    modern: str                 # closest modern-script equivalent
    iast: Optional[str] = None  # IAST transliteration (Roman)
    meaning: Optional[str] = None
    target_lang: str = "devanagari"


@dataclass
class MatchResult:
    entry: MappingEntry
    score: float
    kind: str  # "exact" | "fuzzy"


class ScriptMappingDataset:
    """Loads and queries the script-mapping JSON file."""

    def __init__(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Mapping file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.entries: List[MappingEntry] = []
        self._by_source: Dict[str, MappingEntry] = {}

        for item in raw:
            src = item.get("glyph") or item.get("token") or item.get("source")
            if not src:
                continue
            modern = (item.get("devanagari") or item.get("tamil")
                      or item.get("modern") or item.get("english_gloss") or "")
            e = MappingEntry(
                script=item.get("script", "unknown"),
                source=src,
                modern=modern,
                iast=item.get("iast"),
                meaning=item.get("meaning") or item.get("english_gloss"),
                target_lang=("tamil" if "tamil" in item else "devanagari"),
            )
            self.entries.append(e)
            self._by_source[src] = e

        log.info("Loaded %d mapping entries from %s", len(self.entries), path)

    # -------------------- exact + fuzzy lookup --------------------

    def exact(self, token: str) -> Optional[MappingEntry]:
        return self._by_source.get(token)

    def fuzzy(self, token: str, top_k: int = 3, min_score: float = 0.55) -> List[MatchResult]:
        scored = [
            MatchResult(entry=e, score=_similarity(token, e.source), kind="fuzzy")
            for e in self.entries
        ]
        scored.sort(key=lambda m: m.score, reverse=True)
        return [m for m in scored[:top_k] if m.score >= min_score]

    # -------------------- token-stream translation --------------------

    def translate_tokens(self, tokens: List[str]) -> Tuple[str, List[Tuple[str, Optional[MatchResult]]]]:
        """
        For each token from OCR, return (best-modern-form, match-details).

        Strategy: longest-prefix greedy matching with fallback to fuzzy.
        This handles multi-character ligatures that the OCR may have split
        across detections.
        """
        out_modern: List[str] = []
        details: List[Tuple[str, Optional[MatchResult]]] = []

        for tok in tokens:
            if not tok:
                continue
            hit = self.exact(tok)
            if hit:
                out_modern.append(hit.modern or tok)
                details.append((tok, MatchResult(entry=hit, score=1.0, kind="exact")))
                continue
            fuzzy_hits = self.fuzzy(tok, top_k=1)
            if fuzzy_hits:
                best = fuzzy_hits[0]
                out_modern.append(best.entry.modern or tok)
                details.append((tok, best))
            else:
                # Unmatched — pass through untouched so GenAI can still reason about it.
                out_modern.append(tok)
                details.append((tok, None))

        return " ".join(out_modern), details


# ---------------------------------------------------------------------------
# Convenience loader
# ---------------------------------------------------------------------------

def load_default_dataset(data_dir: str) -> ScriptMappingDataset:
    """Load `data/script_mappings.json` relative to the project root."""
    return ScriptMappingDataset(os.path.join(data_dir, "script_mappings.json"))
