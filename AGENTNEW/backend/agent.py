from __future__ import annotations

from typing import Dict, List

from .ontology import map_topics
from .pdf_utils import best_pdf_url, extract_text_from_pdf_bytes, fetch_pdf_bytes
from .rank import deduplicate, hybrid_rank
from .search import multi_source_search
from .summarize import llm_summarize
from .text_utils import expand_terms, extract_keywords

MAX_DOCS = 30


def extract_findings_and_limitations(text: str):
    text_low = text.lower()
    findings = ""
    limits = ""

    if "results show" in text_low:
        findings = text_low.split("results show", 1)[1][:200]
    elif "our results" in text_low:
        findings = text_low.split("our results", 1)[1][:200]
    elif "we found" in text_low:
        findings = text_low.split("we found", 1)[1][:200]

    if "limitations" in text_low:
        limits = text_low.split("limitations", 1)[1][:200]
    elif "however" in text_low:
        limits = text_low.split("however", 1)[1][:200]

    return findings.strip(), limits.strip()


class LitReviewAgent:
    def __init__(self, enable_pdf: bool = True):
        self.enable_pdf = enable_pdf

    def plan_queries(self, title: str) -> List[str]:
        base = extract_keywords(title)
        expanded = expand_terms(base)[:20]
        queries = [title] + base + [" ".join(base[:3])] + expanded[:5]
        seen = set()
        out = []
        for q in queries:
            if q and q not in seen:
                seen.add(q)
                out.append(q)
        return out

    def retrieve(self, queries: List[str]) -> List[Dict]:
        items = multi_source_search(queries, per_source=12)
        items = deduplicate(items)
        return items[:200]

    def rank(self, title: str, items: List[Dict]) -> List[Dict]:
        ranked = hybrid_rank(title, items)
        ordered = [items[i] for i, _ in ranked]
        return ordered[:MAX_DOCS]

    def enrich_with_text(self, items: List[Dict]) -> List[Dict]:
        out = []
        for it in items:
            text = None
            if self.enable_pdf:
                pdf = best_pdf_url(it)
                if pdf:
                    data = fetch_pdf_bytes(pdf)
                    if data:
                        text = extract_text_from_pdf_bytes(data)

            it2 = dict(it)
            it2["fulltext"] = text

            if text:
                findings, limits = extract_findings_and_limitations(text)
            else:
                findings, limits = "", ""

            it2["findings"] = findings
            it2["limitations"] = limits

            out.append(it2)
        return out

    def cluster_topics(self, items: List[Dict]) -> List[str]:
        kws = []
        for it in items:
            kws += (it.get("title") or "").lower().split()
        return map_topics(kws)

    def write_review(self, title: str, items: List[Dict]) -> Dict:
        snippets, refs = [], []
        for it in items[:12]:
            snippet = it.get("abstract") or (it.get("fulltext") or "")[:1500]
            if not snippet:
                continue
            snippets.append(snippet)
            refs.append(it.get("title") or "Untitled")

        body = llm_summarize(snippets, refs) if snippets else "(No content to summarize)"

        references = []
        for idx, it in enumerate(items, 1):
            t = it.get("title") or "Untitled"
            a = ", ".join(it.get("authors") or [])
            y = it.get("year") or ""
            v = it.get("venue") or ""
            u = it.get("url") or it.get("pdf_url") or ""
            references.append(f"[{idx}] {t} — {a} ({y}). {v}. {u}")

        return {"title": title, "body": body, "references": references}

    def run(self, title: str) -> Dict:
        queries = self.plan_queries(title)
        items = self.retrieve(queries)
        ranked = self.rank(title, items)
        enriched = self.enrich_with_text(ranked)
        topics = self.cluster_topics(enriched)
        review = self.write_review(title, enriched)
        review["topics"] = topics
        review["items"] = enriched
        return review
