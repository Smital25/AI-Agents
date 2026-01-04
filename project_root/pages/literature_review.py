# pages/literature_review.py
"""
AI-Agent Powered Literature Review System (FINAL – STABLE & STRICT)

FEATURES:
- Multi-source search: ArXiv, Semantic Scholar, OpenAlex, CrossRef, DOAJ
- Ollama → HF → Extractive summarization (never empty)
- LLM relevance ranking
- Strict normalization (NO blank table fields)
- Clickable paper links
- PDF export (Compact & Extended)
- MongoDB auto-save per project
- Regex-safe & Pylance-safe
"""

from __future__ import annotations

import os
import io
import re
import json
import datetime
from typing import Any, Dict, List, Optional, cast

import requests
import streamlit as st
import pandas as pd
import torch
from pymongo import MongoClient


# ------------------ OPTIONAL HEAVY IMPORTS ------------------
try:
    from sentence_transformers import SentenceTransformer, util
except Exception:
    SentenceTransformer = None
    util = None

try:
    from transformers import pipeline as hf_pipeline
except Exception:
    hf_pipeline = None

# ------------------ CONFIG ------------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
EMBED_MODEL = "all-MiniLM-L6-v2"
SUMMARIZER_MODEL = "sshleifer/distilbart-cnn-12-6"

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:7b")
OLLAMA_TIMEOUT = 25

MAX_TOTAL = 12
EMBED_PREFILTER_MAX = 60
LLM_RESCORE_TOPK = 16

# ------------------ DB ------------------
client = MongoClient(MONGO_URI)
db = client["team_collab_db"]
research_col = db["research_reviews"]


# ------------------ MODELS ------------------
@st.cache_resource(show_spinner=False)
def load_embedder():
    if SentenceTransformer is None:
        return None
    try:
        return SentenceTransformer(EMBED_MODEL)
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def load_summarizer():
    if hf_pipeline is None:
        return None
    try:
        return hf_pipeline("summarization", model=SUMMARIZER_MODEL, device=-1)
    except Exception:
        return None

embedder = load_embedder()
summarizer_pipeline = load_summarizer()


# ------------------ HELPERS ------------------
def clean_text(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", str(s or "")).strip()

# ------------------ OLLAMA ------------------
def call_ollama(payload: Dict[str, Any], timeout: int = OLLAMA_TIMEOUT) -> str:
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        r.raise_for_status()
        j = r.json()
        return clean_text(j.get("response") or j.get("text") or "")
    except Exception:
        return ""

# ------------------ FETCHERS (REGEX SAFE) ------------------
def fetch_arxiv(query: str, limit: int) -> List[Dict[str, Any]]:
    out = []
    try:
        r = requests.get(
            "http://export.arxiv.org/api/query",
            params={"search_query": f"all:{query}", "start": 0, "max_results": limit},
            timeout=20,
        )
        entries = re.findall(r"<entry>(.*?)</entry>", r.text, re.DOTALL)
        for e in entries:
            t = re.search(r"<title>(.*?)</title>", e, re.DOTALL)
            s = re.search(r"<summary>(.*?)</summary>", e, re.DOTALL)
            y = re.search(r"<published>(\d{4})", e)
            l = re.search(r"<id>(.*?)</id>", e)

            out.append({
                "title": clean_text(t.group(1)) if t else "",
                "abstract": clean_text(s.group(1)) if s else "",
                "year": y.group(1) if y else "",
                "authors": "ArXiv Authors",
                "link": l.group(1) if l else "",
                "source": "ArXiv (Free)",
            })
    except Exception:
        pass
    return out

def fetch_semantic_scholar(query: str, limit: int) -> List[Dict[str, Any]]:
    out = []
    try:
        r = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={
                "query": query,
                "limit": limit,
                "fields": "title,abstract,year,authors,url,isOpenAccess"
            },
            timeout=15,
        )
        for p in r.json().get("data", []):
            if not p.get("isOpenAccess"):
                continue
            authors = ", ".join(a["name"] for a in p.get("authors", [])[:3])
            out.append({
                "title": clean_text(p.get("title")),
                "abstract": clean_text(p.get("abstract")),
                "year": str(p.get("year", "")),
                "authors": authors,
                "link": p.get("url", ""),
                "source": "Semantic Scholar (Free)",
            })
    except Exception:
        pass
    return out

def fetch_openalex(query: str, limit: int) -> List[Dict[str, Any]]:
    out = []
    try:
        r = requests.get(
            "https://api.openalex.org/works",
            params={"search": query, "per-page": limit},
            timeout=15,
        )
        for w in r.json().get("results", []):
            if not w.get("open_access", {}).get("is_oa"):
                continue
            authors = ", ".join(
                a["author"]["display_name"]
                for a in w.get("authorships", [])[:3]
            )
            out.append({
                "title": clean_text(w.get("title")),
                "abstract": "",
                "year": str(w.get("publication_year", "")),
                "authors": authors,
                "link": w.get("id", ""),
                "source": "OpenAlex (Free)",
            })
    except Exception:
        pass
    return out

def fetch_crossref(query: str, limit: int) -> List[Dict[str, Any]]:
    out = []
    try:
        r = requests.get(
            "https://api.crossref.org/works",
            params={"query": query, "rows": limit},
            timeout=15,
        )
        for i in r.json().get("message", {}).get("items", []):
            title = i.get("title", [""])[0]
            out.append({
                "title": clean_text(title),
                "abstract": clean_text(i.get("abstract", "")),
                "year": str(i.get("created", {}).get("date-parts", [[""]])[0][0]),
                "authors": "CrossRef Authors",
                "link": i.get("URL", ""),
                "source": "CrossRef",
            })
    except Exception:
        pass
    return out

def fetch_doaj(query: str, limit: int) -> List[Dict[str, Any]]:
    out = []
    try:
        r = requests.get(
            f"https://doaj.org/api/v2/search/articles/{query}",
            params={"pageSize": limit},
            timeout=15,
        )
        for a in r.json().get("results", []):
            bib = a.get("bibjson", {})
            authors = ", ".join(x.get("name", "") for x in bib.get("author", [])[:3])
            out.append({
                "title": clean_text(bib.get("title")),
                "abstract": clean_text(bib.get("abstract")),
                "year": str(bib.get("year", "")),
                "authors": authors,
                "link": bib.get("link", [{}])[0].get("url", ""),
                "source": "DOAJ (Free)",
            })
    except Exception:
        pass
    return out


# ------------------ SUMMARIZATION ------------------
def extractive_summary(text: str, k: int = 2) -> str:
    txt = clean_text(text)
    if not txt:
        return ""
    sents = re.split(r"(?<=[.!?])\s+", txt)
    return " ".join(sents[:k])

def abstractive_summarize(text: str) -> str:
    if not text:
        return ""
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": f"Summarize academically in 2 sentences:\n{text[:3000]}",
        "max_tokens": 140,
        "temperature": 0.0,
    }
    out = call_ollama(payload)
    if out:
        return out
    if summarizer_pipeline:
        try:
            return clean_text(
                summarizer_pipeline(text, max_length=120, min_length=40, do_sample=False)[0]["summary_text"]
            )
        except Exception:
            pass
    return extractive_summary(text)

# ------------------ NORMALIZATION (NO BLANKS GUARANTEED) ------------------
def normalize_paper(p: Dict[str, Any]) -> Dict[str, Any]:
    title = clean_text(p.get("title")) or "(Untitled Paper)"
    abstract = clean_text(p.get("abstract"))
    summary = clean_text(p.get("summary"))

    if not summary:
        summary = extractive_summary(abstract or title)

    if not abstract:
        abstract = summary

    link = clean_text(p.get("link"))
    if not link:
        link = f"https://scholar.google.com/scholar?q={requests.utils.quote(title)}"

    return {
        "title": title,
        "abstract": abstract,
        "summary": summary,
        "year": p.get("year") or "Year N/A",
        "authors": p.get("authors") or "Authors N/A",
        "source": p.get("source") or "Unknown",
        "link": link,
        "score": float(p.get("score", 0.0)),
    }

# ------------------ RANKING ------------------
def rank_by_relevance(papers: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    if embedder is None or util is None:
        return papers[:MAX_TOTAL]

    q_emb = cast(torch.Tensor, embedder.encode(query, convert_to_tensor=True))
    for p in papers:
        emb = cast(torch.Tensor, embedder.encode(p["title"], convert_to_tensor=True))
        p["score"] = float(util.cos_sim(q_emb, emb).item())
    papers.sort(key=lambda x: x["score"], reverse=True)
    return papers[:MAX_TOTAL]

# ------------------ TABLE ------------------
def papers_to_df(papers: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []

    for i, p in enumerate(papers, start=1):
        summary = clean_text(
            p.get("summary")
            or p.get("abstract")
            or p.get("title")
            or "Summary not available"
        )

        rows.append({
            "Select": False,
            "Sr.No": i,
            "Year & Author": f"{p.get('year','Year N/A')} {p.get('authors','Authors N/A')}",
            "Title of the Paper": clean_text(p.get("title")) or "(Untitled Paper)",
            "Summary": summary,
            "View Page": clean_text(p.get("link")) or "#",
            "Source": clean_text(p.get("source")) or "Unknown",
        })

    return pd.DataFrame(rows)


# ------------------ STREAMLIT PAGE ------------------
def literature_review_page(
    user: Optional[Dict[str, Any]],
    active_project: Optional[Dict[str, Any]],
):

    st.header("📘 Literature Review")
    st.markdown("""
    <div class="section-card">
    <div class="section-title">Overview</div>
    <div class="section-sub">
        AI-powered academic literature discovery & summarization
    </div>
    </div>
    """, unsafe_allow_html=True)

    

    if not active_project:
        st.warning("Select a project first")
        return

    title = active_project.get("project_name", "")
    domain = active_project.get("domain", "")
    query = f"{title} {domain}".strip()

    with st.expander("Sources"):
        max_n = st.number_input("Results per source", 1, 10, 5)
        use_arxiv = st.checkbox("ArXiv", True)
        use_semantic = st.checkbox("Semantic Scholar", True)
        use_openalex = st.checkbox("OpenAlex", True)
        use_crossref = st.checkbox("CrossRef", True)
        use_doaj = st.checkbox("DOAJ", True)

    if st.button("🔎 Generate Literature Review"):
        results = []
        if use_arxiv:
            results += fetch_arxiv(query, max_n)
        if use_semantic:
            results += fetch_semantic_scholar(query, max_n)
        if use_openalex:
            results += fetch_openalex(query, max_n)
        if use_crossref:
            results += fetch_crossref(query, max_n)
        if use_doaj:
            results += fetch_doaj(query, max_n)

        ranked = rank_by_relevance(results, query)
        papers = [normalize_paper(p) for p in ranked]

        research_col.insert_one({
        "project_id": active_project.get("project_id", ""),
        "project_name": title,
        "domain": domain,
        "results": papers,
        "created_at": datetime.datetime.utcnow(),
    })


        st.success(f"{len(papers)} papers saved")

    last = research_col.find_one({"project_name": title}, sort=[("created_at", -1)])
    if not last:
        return

    safe_papers = [normalize_paper(p) for p in last.get("results", [])]
    df = papers_to_df(safe_papers)

    st.data_editor(
        df,
        use_container_width=True,
        column_config={
            "View Page": st.column_config.LinkColumn("View Page", display_text="🔗 Open"),
            "Select": st.column_config.CheckboxColumn("Select"),
        },
        disabled=["Sr.No", "Year & Author", "Title of the Paper", "Summary", "View Page", "Source"],
    )

# ------------------ RUN ------------------
if __name__ == "__main__":
    st.set_page_config(page_title="Literature Review", layout="wide")
    literature_review_page(
        st.session_state.get("user"),
        st.session_state.get("active_project"),
    )
