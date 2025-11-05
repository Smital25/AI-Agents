from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Set

import nltk
import yake
from nltk.corpus import wordnet as wn

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")


@dataclass
class Keywords:
    primary: List[str]
    expanded: List[str]


_kw_extractor = yake.KeywordExtractor(lan="en", top=12, n=1)


def clean_text(t: str) -> str:
    return re.sub(r"\s+", " ", t or "").strip()


def extract_keywords(title: str) -> List[str]:
    title = clean_text(title)
    if not title:
        return []
    kws = [k for k, _ in _kw_extractor.extract_keywords(title)]
    toks = re.findall(r"[A-Za-z0-9\\-]+", title)
    out: List[str] = list(dict.fromkeys([*(kws or []), *toks]))
    return [w.lower() for w in out if len(w) > 2]


def expand_terms(terms: List[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for t in terms:
        cands: Set[str] = {t}
        for syn in wn.synsets(t):
            lemmas = syn.lemmas() or []
            for lem in lemmas:
                cands.add(lem.name().replace("_", " "))
        for c in cands:
            c = c.lower()
            if c not in seen and len(c) > 2:
                seen.add(c)
                out.append(c)
    return out
