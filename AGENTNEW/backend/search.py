from __future__ import annotations

from typing import Dict, List

import requests
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "LitReview-Agent/0.1"}


def _record(title, authors, year, venue, abstract, url, pdf_url, doi, source):
    return {
        "title": title or "",
        "authors": authors or [],
        "year": year or 0,
        "venue": venue or "",
        "abstract": abstract,
        "url": url,
        "pdf_url": pdf_url,
        "doi": doi,
        "source": source,
    }


def search_openalex(query: str, n: int = 20) -> List[Dict]:
    url = "https://api.openalex.org/works"
    params = {"search": query, "per-page": n}
    r = requests.get(url, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json().get("results", [])
    out: List[Dict] = []
    for w in data:
        title = w.get("title")
        year = w.get("publication_year") or 0
        doi = w.get("doi") or ""
        inv = w.get("abstract_inverted_index")
        abstract = None
        if inv:
            tokens = sorted((pos, tok) for tok, poss in inv.items() for pos in poss)
            abstract = " ".join(tok for _, tok in tokens)
        oa = w.get("open_access", {}) or {}
        pdf_url = oa.get("oa_url")
        venue = (w.get("host_venue") or {}).get("display_name")
        url_final = (w.get("primary_location") or {}).get("landing_page_url")
        authors = [a.get("author", {}).get("display_name") for a in (w.get("authorships") or [])]
        out.append(_record(title, authors, year, venue, abstract, url_final, pdf_url, doi, "openalex"))
    return out


ARXIV = "http://export.arxiv.org/api/query"


def search_arxiv(query: str, n: int = 20) -> List[Dict]:
    params = {"search_query": f"all:{query}", "start": 0, "max_results": n}
    r = requests.get(ARXIV, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "xml")
    out: List[Dict] = []
    for entry in soup.find_all("entry"):
        title = entry.title.text.strip()
        abstract = entry.summary.text.strip()
        url_ = None
        pdf = None
        for link in entry.find_all("link"):
            if link.get("title") == "pdf":
                pdf = link.get("href")
            if link.get("rel") == "alternate":
                url_ = link.get("href")
        authors = [a.text for a in entry.find_all("name")]
        year = int(entry.published.text[:4])
        out.append(_record(title, authors, year, "arXiv", abstract, url_, pdf, None, "arxiv"))
    return out


S2 = "https://api.semanticscholar.org/graph/v1/paper/search"


def search_s2(query: str, n: int = 20) -> List[Dict]:
    params = {
        "query": query,
        "limit": n,
        "fields": "title,year,authors,abstract,venue,openAccessPdf,externalIds,url",
    }
    r = requests.get(S2, params=params, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json().get("data", [])
    out: List[Dict] = []
    for p in data:
        title = p.get("title")
        abstract = p.get("abstract")
        pdf = (p.get("openAccessPdf") or {}).get("url")
        url_ = p.get("url")
        venue = p.get("venue")
        year = p.get("year") or 0
        doi = (p.get("externalIds") or {}).get("DOI")
        authors = [a.get("name") for a in (p.get("authors") or [])]
        out.append(_record(title, authors, year, venue, abstract, url_, pdf, doi, "s2"))
    return out


def multi_source_search(queries: List[str], per_source: int = 12) -> List[Dict]:
    all_results: List[Dict] = []
    for q in queries:
        try: all_results += search_openalex(q, per_source)
        except: pass
        try: all_results += search_arxiv(q, per_source)
        except: pass
        try: all_results += search_s2(q, per_source)
        except: pass
    return all_results
