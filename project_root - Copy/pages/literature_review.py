# pages/literature_review.py
"""
AI-Agent Powered Literature Review System (copy-paste ready)
 - Multi-API search (ArXiv, Semantic Scholar, OpenAlex, CrossRef)
 - Smart merging, dedupe, abstract generation
 - Summaries using: Ollama -> HF -> Extractive
 - LLM relevance ranking (optional)
 - PDF export (Compact / Extended)
 - Auto-saves result per-project into MongoDB research_reviews
"""

from __future__ import annotations

import os
import io
import re
import uuid
import json
import traceback
import datetime
from typing import Any, Dict, List, Optional

import requests
import streamlit as st
import pandas as pd

from pymongo import MongoClient

# Optional heavy libs guarded
try:
    from sentence_transformers import SentenceTransformer, util
except Exception:
    SentenceTransformer = None
    util = None

# Pre-declare PDF-related names as Any so Pylance doesn't treat them as Optional
A4: Any
getSampleStyleSheet: Any
ParagraphStyle: Any
TA_LEFT: Any
SimpleDocTemplate: Any
Paragraph: Any
Spacer: Any
Table: Any
TableStyle: Any
PageBreak: Any
colors: Any
pdfmetrics: Any
TTFont: Any

# PDF tools (reportlab). Guarded import to avoid runtime errors if lib missing.
try:
    from reportlab.lib.pagesizes import A4 as _A4
    from reportlab.lib.styles import getSampleStyleSheet as _getSampleStyleSheet, ParagraphStyle as _ParagraphStyle
    from reportlab.lib.enums import TA_LEFT as _TA_LEFT
    from reportlab.platypus import (
        SimpleDocTemplate as _SimpleDocTemplate,
        Paragraph as _Paragraph,
        Spacer as _Spacer,
        Table as _Table,
        TableStyle as _TableStyle,
        PageBreak as _PageBreak,
    )
    from reportlab.lib import colors as _colors
    from reportlab.pdfbase import pdfmetrics as _pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont as _TTFont

    A4 = _A4
    getSampleStyleSheet = _getSampleStyleSheet
    ParagraphStyle = _ParagraphStyle
    TA_LEFT = _TA_LEFT
    SimpleDocTemplate = _SimpleDocTemplate
    Paragraph = _Paragraph
    Spacer = _Spacer
    Table = _Table
    TableStyle = _TableStyle
    PageBreak = _PageBreak
    colors = _colors
    pdfmetrics = _pdfmetrics
    TTFont = _TTFont
except Exception:
    # Fallbacks used only if reportlab is missing; types are Any so Pylance is happy.
    A4 = None
    getSampleStyleSheet = None
    ParagraphStyle = None
    TA_LEFT = None
    SimpleDocTemplate = None
    Paragraph = None
    Spacer = None
    Table = None
    TableStyle = None
    PageBreak = None
    colors = None
    pdfmetrics = None
    TTFont = None

# Optional HF summarizer (transformers)
try:
    from transformers import pipeline as hf_pipeline
except Exception:
    hf_pipeline = None

# ------------------ CONFIG ------------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
SUMMARIZER_MODEL = os.getenv("SUMMARIZER_MODEL", "sshleifer/distilbart-cnn-12-6")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:7b")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "25"))

MAX_TOTAL = 12
LLM_RESCORE_TOPK = 16
EMBED_PREFILTER_MAX = 60

# ------------------ DB SETUP ------------------
client_db = MongoClient(MONGO_URI)
db = client_db["team_collab_db"]
research_col = db["research_reviews"]

# ------------------ CACHED MODELS ------------------
@st.cache_resource(show_spinner=False)
def load_embedder():
    """Load SentenceTransformer embedder if available, otherwise return None."""
    if SentenceTransformer is None:
        return None
    try:
        return SentenceTransformer(EMBED_MODEL)
    except Exception:
        try:
            return SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            return None

@st.cache_resource(show_spinner=False)
def load_hf_summarizer():
    if hf_pipeline is None:
        return None
    try:
        return hf_pipeline("summarization", model=SUMMARIZER_MODEL, device=-1)
    except Exception:
        try:
            return hf_pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=-1)
        except Exception:
            return None

embedder = load_embedder()
summarizer_pipeline = load_hf_summarizer()

# ------------------ HELPERS ------------------
def safe_get(d: Any, k: str, default=""):
    return d.get(k, default) if isinstance(d, dict) else default

def clean_text(s: Optional[str]) -> str:
    if not s:
        return ""
    return re.sub(r"\s+", " ", str(s)).strip()

# ------------------ OLLAMA CALL ------------------
def _ollama_post(payload: Dict[str, Any], timeout: int = OLLAMA_TIMEOUT) -> str:
    """
    Minimal robust wrapper for the Ollama local API.
    Accepts payload dict (model,prompt,max_tokens,...).
    Returns text string (never None).
    """
    try:
        headers = {"Content-Type": "application/json"}
        r = requests.post(OLLAMA_URL, json=payload, headers=headers, timeout=timeout)
        r.raise_for_status()
    except Exception:
        # Fail silently (return empty string), caller should fallback.
        return ""

    # Attempt to parse common json shapes
    text_out = ""
    try:
        j = r.json()
        if isinstance(j, dict):
            if "text" in j and isinstance(j["text"], str):
                text_out = j["text"]
            elif "output" in j and isinstance(j["output"], str):
                text_out = j["output"]
            elif "choices" in j and isinstance(j["choices"], list) and j["choices"]:
                c0 = j["choices"][0]
                if isinstance(c0, dict) and "text" in c0:
                    text_out = c0["text"]
                else:
                    text_out = str(c0)
            else:
                text_out = json.dumps(j)
        else:
            text_out = str(j)
    except Exception:
        # Non-JSON response
        text_out = r.text or ""

    return text_out or ""

def call_ollama(payload: Dict[str, Any], timeout: int = OLLAMA_TIMEOUT) -> str:
    """
    Public wrapper used throughout the module. Accepts
    a payload dict and returns text. This matches prior usage.
    """
    try:
        return _ollama_post(payload, timeout=timeout)
    except Exception:
        return ""

# ------------------ FETCHERS ------------------
BASE_ARXIV = "http://export.arxiv.org/api/query"
BASE_OPENALEX = "https://api.openalex.org/works"
SEMANTIC_SCHOLAR_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search"
CROSSREF_API = "https://api.crossref.org/works"

def fetch_arxiv(query: str, max_results: int = 6) -> List[Dict[str, Any]]:
    out = []
    try:
        r = requests.get(BASE_ARXIV, params={
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results
        }, timeout=20)
        xml = r.text
        entries = re.findall(r"<entry>(.*?)</entry>", xml, re.DOTALL)
        for e in entries[:max_results]:
            def g(tag):
                m = re.search(fr"<{tag}>(.*?)</{tag}>", e, re.DOTALL)
                return clean_text(m.group(1)) if m else ""
            out.append({
                "title": g("title") or "(No Title)",
                "abstract": g("summary"),
                "link": g("id"),
                "year": (g("published") or "")[:4],
                "source": "ArXiv"
            })
    except Exception:
        pass
    return out

def fetch_semantic_scholar(query: str, max_results: int = 6) -> List[Dict[str, Any]]:
    out = []
    try:
        r = requests.get(SEMANTIC_SCHOLAR_SEARCH, params={
            "query": query,
            "limit": max_results,
            "fields": "title,abstract,url,year"
        }, timeout=15)
        data = r.json().get("data", [])
        for p in data:
            out.append({
                "title": clean_text(p.get("title", "(No Title)")),
                "abstract": clean_text(p.get("abstract", "")),
                "link": p.get("url", ""),
                "year": str(p.get("year", "")),
                "source": "SemanticScholar"
            })
    except Exception:
        pass
    return out

def fetch_openalex(query: str, max_results: int = 6) -> List[Dict[str, Any]]:
    out = []
    try:
        r = requests.get(BASE_OPENALEX, params={"search": query, "per-page": max_results}, timeout=15)
        for w in r.json().get("results", []):
            abstract = ""
            if isinstance(w.get("abstract"), str):
                abstract = w["abstract"]
            else:
                inv = w.get("abstract_inverted_index")
                if isinstance(inv, dict):
                    items = []
                    for tok, pos_list in inv.items():
                        for p in pos_list:
                            items.append((p, tok))
                    abstract = " ".join([t for _, t in sorted(items)])
            out.append({
                "title": clean_text(w.get("title", "(No Title)")),
                "abstract": clean_text(abstract),
                "link": w.get("id", ""),
                "year": str(w.get("publication_year", "")),
                "source": "OpenAlex"
            })
    except Exception:
        pass
    return out

def fetch_crossref(query: str, max_results: int = 6) -> List[Dict[str, Any]]:
    out = []
    try:
        r = requests.get(CROSSREF_API, params={"query": query, "rows": max_results}, timeout=15)
        for i in r.json().get("message", {}).get("items", []):
            title = ""
            try:
                title = i.get("title", [""])[0]
            except Exception:
                pass
            out.append({
                "title": clean_text(title) or "(No Title)",
                "abstract": clean_text(i.get("abstract", "")),
                "link": i.get("URL", ""),
                "year": str(i.get("created", {}).get("date-parts", [[None]])[0][0] or ""),
                "source": "CrossRef"
            })
    except Exception:
        pass
    return out

# ------------------ PART 2: merge, summarization, ranking, PDF utils ------------------

def merge_and_dedup(lists: List[List[Dict[str, Any]]], max_total: int = MAX_TOTAL) -> List[Dict[str, Any]]:
    merged = [p for L in lists for p in (L or [])]
    seen = set()
    out: List[Dict[str, Any]] = []
    for p in merged:
        title_key = (p.get("title") or "").strip().lower()
        if not title_key:
            continue
        if title_key in seen:
            continue
        seen.add(title_key)
        # normalize
        p["title"] = p.get("title") or "(No Title)"
        p["abstract"] = p.get("abstract") or ""
        p["link"] = p.get("link") or ""
        p["year"] = p.get("year") or ""
        p["source"] = p.get("source") or ""
        out.append(p)
        if len(out) >= max_total:
            break
    return out

# ---------- SUMMARIZATION (Ollama -> HF -> Extractive) ----------
def abstractive_summarize_with_ollama(text: str, max_len: int = 140) -> str:
    if not text:
        return ""
    # Build a short payload for Ollama (we rely on call_ollama that expects a dict)
    prompt = (
        "You are a concise academic summarizer. Produce a short abstractive summary in 1-3 sentences.\n\n"
        "Text:\n" + text[:3000] + "\n\nSummary:"
    )
    payload = {"model": OLLAMA_MODEL, "prompt": prompt, "max_tokens": max_len, "temperature": 0.0}
    out = call_ollama(payload, timeout=OLLAMA_TIMEOUT)
    return clean_text(out)

def abstractive_summarize(text: str, max_len: int = 140) -> str:
    try:
        # Try Ollama first (if reachable)
        s = abstractive_summarize_with_ollama(text, max_len=max_len)
        if s:
            return s
    except Exception:
        pass

    # HF summarizer fallback
    if summarizer_pipeline is not None:
        try:
            out = summarizer_pipeline(text, max_length=max_len, min_length=max(30, int(max_len/4)), do_sample=False)
            if isinstance(out, list) and out and isinstance(out[0], dict):
                return clean_text(out[0].get("summary_text", ""))
            return clean_text(str(out))
        except Exception:
            pass

    # Extractive fallback
    try:
        return extractive_summary(text, top_k_sentences=3)
    except Exception:
        return ""

def extractive_summary(text: str, top_k_sentences: int = 3) -> str:
    txt = clean_text(text)
    if not txt:
        return ""
    sentences = re.split(r'(?<=[.!?])\s+', txt)
    if len(sentences) <= top_k_sentences:
        return " ".join(sentences)
    # if embedder or util missing — fallback to first sentences
    if embedder is None or util is None:
        return " ".join(sentences[:top_k_sentences])
    try:
        emb_doc = embedder.encode(txt, convert_to_tensor=True)
        sent_embs = embedder.encode(sentences, convert_to_tensor=True)
        # util.cos_sim may exist
        sim_scores = util.cos_sim(emb_doc, sent_embs)[0].cpu().tolist()
        ranked = sorted(
            enumerate(sim_scores),
            key=lambda x: x[1],
            reverse=True
        )[:top_k_sentences]
        ranked_sorted = sorted([i for i, _ in ranked])
        return " ".join([sentences[i] for i in ranked_sorted])
    except Exception:
        return " ".join(sentences[:top_k_sentences])

# ---------- RELEVANCE RANKING (embed + optional LLM rescore) ----------
def llm_rescore_relevance(paper: Dict[str, Any], query: str, model: str = OLLAMA_MODEL) -> Optional[float]:
    """
    Ask LLM to return a numeric relevance score between 0.0 and 1.0.
    This is optional and can fail silently.
    """
    title = paper.get("title", "")
    abstract = (paper.get("abstract") or "")[:2000]
    prompt = (
        f"Rate the RELEVANCE of the following paper to the query on a scale 0.0 to 1.0. "
        f"Return only the numeric score in decimal format.\n\nQuery: {query}\n\nTitle: {title}\n\nAbstract: {abstract}\n\nScore:"
    )
    payload = {"model": model, "prompt": prompt, "max_tokens": 12, "temperature": 0.0}
    try:
        out = call_ollama(payload, timeout=10)
        m = re.search(r"([01](?:\.[0-9]+)?)", out)
        if m:
            val = float(m.group(1))
            return max(0.0, min(1.0, val))
    except Exception:
        pass
    return None

def rank_by_relevance(papers: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    if not papers:
        return []
    candidates = papers[:EMBED_PREFILTER_MAX] if len(papers) > EMBED_PREFILTER_MAX else papers[:]
    # If embedder missing, skip embedding step
    if embedder is None or util is None:
        # Set a default score (0) and return top MAX_TOTAL
        for p in candidates:
            p["score"] = 0.0
        return candidates[:MAX_TOTAL]

    try:
        q_emb = embedder.encode(query, convert_to_tensor=True)
    except Exception:
        for p in candidates:
            p["score"] = 0.0
        return candidates[:MAX_TOTAL]

    scored = []
    for p in candidates:
        combined = (p.get("title", "") + ". " + (p.get("abstract") or ""))[:4000]
        try:
            emb = embedder.encode(combined, convert_to_tensor=True)
            sim = float(util.cos_sim(q_emb, emb).item())
        except Exception:
            sim = 0.0
        p["score"] = sim
        scored.append(p)
    scored = sorted(scored, key=lambda x: x.get("score", 0.0), reverse=True)

    # LLM rescore (optional)
    try:
        topk = min(len(scored), LLM_RESCORE_TOPK)
        for p in scored[:topk]:
            s = llm_rescore_relevance(p, query)
            if s is not None:
                p["score"] = 0.45 * p.get("score", 0.0) + 0.55 * s
        scored = sorted(scored, key=lambda x: x.get("score", 0.0), reverse=True)
    except Exception:
        pass

    threshold = 0.06
    filtered = [p for p in scored if p.get("score", 0.0) >= threshold]
    if len(filtered) < max(3, min(MAX_TOTAL, len(scored))):
        return scored[:MAX_TOTAL]
    return filtered[:MAX_TOTAL]

# ---------- PDF helpers ----------
def register_unicode_font() -> str:
    """Try popular system fonts, fallback to Helvetica."""
    # If pdfmetrics/TTFont are unavailable, just return Helvetica
    if pdfmetrics is None or TTFont is None:
        return "Helvetica"

    try:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            os.path.join(os.path.dirname(__file__), "DejaVuSans.ttf")
        ]
        for p in candidates:
            if p and os.path.exists(p):
                try:
                    pdfmetrics.registerFont(TTFont("DejaVuSans", p))
                    return "DejaVuSans"
                except Exception:
                    continue
    except Exception:
        pass
    return "Helvetica"

BASE_FONT = register_unicode_font()

def build_pdf_compact(papers: List[Dict[str, Any]], project_title: str) -> io.BytesIO:
    buf = io.BytesIO()

    # If reportlab is missing, return empty PDF buffer safely
    if (
        SimpleDocTemplate is None
        or getSampleStyleSheet is None
        or ParagraphStyle is None
        or Paragraph is None
        or Spacer is None
        or Table is None
        or TableStyle is None
        or A4 is None
    ):
        buf.write(b"")
        buf.seek(0)
        return buf

    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=20, leftMargin=20, topMargin=30, bottomMargin=20)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('title', parent=styles['Title'], alignment=TA_LEFT, fontName=BASE_FONT, fontSize=14)
    normal = ParagraphStyle('normal', parent=styles['Normal'], fontName=BASE_FONT, fontSize=9)
    elems = []
    elems.append(Paragraph(f"Literature Review — {project_title}", title_style))
    elems.append(Spacer(1, 8))

    data = [["#", "Title", "Year", "Source", "Abstract (short)", "Link", "Score"]]
    for i, p in enumerate(papers, start=1):
        short_abs = (p.get("summary") or p.get("abstract") or "")[:400].replace("\n", " ")
        data.append([str(i), p.get("title", ""), p.get("year", ""), p.get("source", ""), short_abs, p.get("link", ""), f"{p.get('score', 0):.3f}"])

    col_widths = [28, 180, 36, 60, 210, 120, 36]
    table = Table(data, colWidths=col_widths, repeatRows=1)
    if colors is not None:
        style = TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#F5F7FA")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("FONTNAME", (0, 0), (-1, -1), BASE_FONT),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("INNERGRID", (0, 0), (-1, -1), 0.25, colors.grey),
            ("BOX", (0, 0), (-1, -1), 0.25, colors.grey),
        ])
        table.setStyle(style)
    elems.append(table)
    elems.append(Spacer(1, 12))

    elems.append(Paragraph("5-Point Summary (Top papers):", styles["Heading3"]))
    bullets = generate_5_bullet_summary(papers, project_title)
    for line in bullets.splitlines():
        elems.append(Paragraph(line, normal))
        elems.append(Spacer(1, 4))

    doc.build(elems)
    buf.seek(0)
    return buf

def build_pdf_extended(papers: List[Dict[str, Any]], project_title: str) -> io.BytesIO:
    buf = io.BytesIO()

    if (
        SimpleDocTemplate is None
        or getSampleStyleSheet is None
        or ParagraphStyle is None
        or Paragraph is None
        or Spacer is None
        or PageBreak is None
        or A4 is None
    ):
        buf.write(b"")
        buf.seek(0)
        return buf

    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=30, leftMargin=30, topMargin=30, bottomMargin=30)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('title', parent=styles['Title'], alignment=TA_LEFT, fontName=BASE_FONT, fontSize=16)
    h3 = ParagraphStyle('h3', parent=styles['Heading3'], fontName=BASE_FONT)
    normal = ParagraphStyle('normal', parent=styles['Normal'], fontName=BASE_FONT, fontSize=10)
    elems = []
    elems.append(Paragraph(f"Literature Review — {project_title}", title_style))
    elems.append(Spacer(1, 8))

    for i, p in enumerate(papers, start=1):
        elems.append(Paragraph(f"{i}. {p.get('title', '')}", h3))
        meta = f"Source: {p.get('source','')}  •  Year: {p.get('year','')}  •  Score: {p.get('score',0):.3f}"
        elems.append(Paragraph(meta, normal))
        elems.append(Spacer(1, 4))
        elems.append(Paragraph(p.get("abstract", "(No abstract)"), normal))
        if p.get("link"):
            elems.append(Paragraph(f"Link: {p.get('link')}", normal))
        elems.append(Spacer(1, 12))
        if i % 3 == 0:
            elems.append(PageBreak())

    elems.append(Spacer(1, 8))
    elems.append(Paragraph("5-Point Summary (Top papers):", styles["Heading3"]))
    bullets = generate_5_bullet_summary(papers, project_title)
    for line in bullets.splitlines():
        elems.append(Paragraph(line, normal))
        elems.append(Spacer(1, 4))

    doc.build(elems)
    buf.seek(0)
    return buf

# ---------- 5-bullet summary & df util ----------
def generate_5_bullet_summary(papers: List[Dict[str, Any]], project_title: str) -> str:
    texts = []
    for p in papers[:8]:
        t = p.get("abstract") or p.get("summary") or p.get("title")
        if t:
            texts.append(clean_text(t))
    joined = "\n\n".join(texts)
    if not joined:
        return "No content to summarize."
    try:
        s = abstractive_summarize(joined, max_len=180)
        # Split into sentences for bullets (preserve ends)
        bullets = [f"- {b.strip()}" for b in re.split(r'(?<=\.)\s+', s) if b.strip()][:5]
        if not bullets:
            bullets = [f"- {s}"]
        return "\n".join(bullets)
    except Exception:
        return "\n".join([f"- {t.split('.')[0].strip()}" for t in texts[:5]])

def papers_to_df(papers: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for p in papers:
        rows.append({
            "Title": p.get("title", ""),
            "Year": p.get("year", ""),
            "Source": p.get("source", ""),
            "Abstract / Summary": (p.get("summary") or p.get("abstract") or "")[:800],
            "Link": p.get("link", ""),
            "Score": round(float(p.get("score", 0)), 3)
        })
    return pd.DataFrame(rows)
# ------------------ PART 3: MAIN STREAMLIT PAGE ------------------

def literature_review_page(user: Dict[str, Any], active_project: Dict[str, Any]):
    st.markdown("## 📘 Literature Review")

    # ---------------- Validate active project ----------------
    if not active_project:
        st.warning("Please select an active project first.")
        return

    # ALWAYS pull correct project details from active_project
    project_title = safe_get(active_project, "project_name", "Untitled Project")
    domain = safe_get(active_project, "domain", "")
    pid = safe_get(active_project, "project_id", "")

    st.info(f"**Project:** {project_title}  •  **Domain:** {domain}")

    # ---------------- Options ----------------
    with st.expander("Options & Sources"):
        cols = st.columns([1, 1, 1])
        max_per_source = int(cols[0].number_input("Max results per source", 1, 12, 5))
        include_semantic = cols[1].checkbox("SemanticScholar", True)
        include_openalex = cols[1].checkbox("OpenAlex", True)
        include_crossref = cols[2].checkbox("CrossRef", True)
        include_arxiv = cols[2].checkbox("ArXiv", True)
        pdf_mode = cols[0].selectbox("PDF Mode", ["Compact Table (A)", "Extended Report (B)"])

    # ---------------- Load last version ----------------
    last = research_col.find_one({"project_id": pid}, sort=[("created_at", -1)])
    papers: List[Dict[str, Any]] = []

    if last and st.button("📂 Load Last Saved Review"):
        papers = last.get("results", []) or []
        st.success(f"Loaded {len(papers)} saved papers.")

    # ---------------- Generate New ----------------
    if st.button("🔎 Generate Literature Review"):
        query = f"{project_title} {domain}".strip()
        st.info(f"Fetching papers for: **{query}**")

        results: List[Dict[str, Any]] = []

        # Fetching data
        try:
            if include_arxiv:
                results += fetch_arxiv(query, max_per_source)
            if include_semantic:
                results += fetch_semantic_scholar(query, max_per_source)
            if include_openalex:
                results += fetch_openalex(query, max_per_source)
            if include_crossref:
                results += fetch_crossref(query, max_per_source)
        except Exception as e:
            st.warning(f"Some sources failed: {e}")

        # Merge / dedupe
        merged = merge_and_dedup([results], max_total=MAX_TOTAL * 2)

        # Process abstracts + summaries
        processed = []
        for p in merged:
            p2 = dict(p)
            base_text = str(p2.get("abstract") or p2.get("title") or "")

            if not p2.get("abstract"):
                try:
                    p2["abstract"] = abstractive_summarize(base_text[:2000], max_len=80) or p2.get("title", "")
                except Exception:
                    p2["abstract"] = p2.get("title", "")

            try:
                p2["summary"] = abstractive_summarize(base_text[:2000], max_len=120) or extractive_summary(base_text, 2)
            except Exception:
                p2["summary"] = extractive_summary(base_text, 2)

            processed.append(p2)

        # Rank for relevance
        try:
            ranked = rank_by_relevance(processed, query)
        except Exception:
            ranked = processed

        papers = ranked[:MAX_TOTAL]

        # Save to DB
        try:
            research_col.insert_one({
                "project_id": pid,
                "project_name": project_title,
                "domain": domain,
                "results": papers,
                "created_at": datetime.datetime.utcnow()
            })
            st.success("Literature review saved to database.")
        except Exception:
            st.warning("⚠ Could not save to DB (non-fatal).")

    # ---------------- Display ----------------    #
    if not papers:
        st.info("No papers yet. Click *Generate Literature Review* or Load Last Saved Review.")
        return

    df = papers_to_df(papers)
    st.markdown("### 🔎 Results")
    st.dataframe(df, use_container_width=True)

    # ---------------- Selected Details ----------------
    st.markdown("### 📄 Selected Paper Details")
    idx = st.number_input("Select Paper #", 1, len(papers), 1)
    sel = papers[idx - 1]

    st.subheader(sel.get("title", ""))
    st.markdown(f"**Source:** `{sel.get('source')}` • **Year:** `{sel.get('year')}` • **Score:** `{sel.get('score', 0):.3f}`")
    st.write(sel.get("abstract", "(No abstract available)"))

    if sel.get("link"):
        st.markdown(f"[🔗 Open Paper]({sel.get('link')})")

    # ---------------- CSV & PDF ----------------
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    if pdf_mode.startswith("Compact"):
        pdf_buf = build_pdf_compact(papers, project_title)
    else:
        pdf_buf = build_pdf_extended(papers, project_title)

    st.download_button("📥 Download CSV", csv_bytes, f"{project_title}_literature.csv")
    st.download_button("📄 Download PDF", pdf_buf, f"{project_title}_literature.pdf")

    # ---------------- Bullet Summary ----------------
    st.markdown("### 🧠 5-Bullet Summary of Top Papers")
    try:
        bullets = generate_5_bullet_summary(papers, project_title)
    except Exception:
        bullets = ""
    st.text_area("Summary", bullets, height=220)


# ------------------ Dynamic Runner (uses session active_project) ------------------
if __name__ == "__main__":
    st.set_page_config(page_title="Literature Review", layout="centered")
    st.title("Literature Review")

    user = st.session_state.get("user")
    active_project = st.session_state.get("active_project")

    if not user:
        st.error("No user found in session. Please login from main app.")
    elif not active_project:
        st.error("No active project selected. Please open this page from Dashboard.")
    else:
        literature_review_page(user, active_project)
