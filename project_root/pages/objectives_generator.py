"""
🎯 TeamCollab — AI Objectives Generator (DeepSeek, Literature-Driven)
--------------------------------------------------------------------
Flow:
- Reads latest literature review from MongoDB 'research_reviews' (per project).
- Uses DeepSeek via Ollama to generate at least 10 objectives.
- Always falls back to safe academic objectives (no hard failure).
- UI shows EXACTLY 10 objectives.
- User selects ANY 5 and finalizes them.
- Final 5 are saved in MongoDB 'project_objectives' as 'objectives'.

This file exposes:
- objectives_page(user, active_project)  # Streamlit page
- generate_objectives(littext, domain, project)  # simple helper
"""

from __future__ import annotations

import os
import re
import json
from typing import Any, Dict, List

import requests
import streamlit as st
from pymongo import MongoClient

# ===== OPTIONAL SentenceTransformer =====
try:
    from sentence_transformers import SentenceTransformer, util
except Exception:  # noqa: BLE001
    SentenceTransformer = None
    util = None

# ===== CONFIG =====
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
# use DeepSeek by default (you can override with env OLLAMA_MODEL)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "deepseek-r1:32b")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "40"))

client = MongoClient(MONGO_URI)
db = client["team_collab_db"]

research_col = db["research_reviews"]
objectives_col = db["project_objectives"]  # final correct collection


# ===== embed model (optional) =====
@st.cache_resource(show_spinner=False)
def load_embedder():
    if SentenceTransformer:
        try:
            return SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:  # noqa: BLE001
            return None
    return None


embedder = load_embedder()


# ===== HELPERS =====
def safe_get(d: Any, k: str, default: str = "") -> str:
    return d.get(k, default) if isinstance(d, dict) else default


def call_ollama(prompt: str, max_tokens: int = 400) -> str:
    """
    Generic wrapper around Ollama /api/generate.
    Works with DeepSeek & other chatty models.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "stream": False,
    }
    try:
        r = requests.post(
            OLLAMA_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=OLLAMA_TIMEOUT,
        )
        r.raise_for_status()
    except Exception:  # noqa: BLE001
        return ""

    try:
        j = r.json()
        if isinstance(j, dict):
            # different backends sometimes use different keys
            if isinstance(j.get("response"), str):
                return j["response"]
            if isinstance(j.get("text"), str):
                return j["text"]
            if isinstance(j.get("output"), str):
                return j["output"]
        return str(j)
    except Exception:  # noqa: BLE001
        return r.text or ""


def parse_numbered_list(text: str) -> List[str]:
    """Parse a numbered (or dashed) list into items."""
    if not text:
        return []

    # remove DeepSeek <think> blocks if present
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)

    matches = re.findall(r"^\s*(\d+[\.\)]|-)\s+(.*\S)", text, flags=re.MULTILINE)
    items: List[str] = []

    for _, content in matches:
        content = content.strip()
        content = re.sub(r"[\-\*\u2022]+$", "", content).strip()
        if len(content.split()) >= 3:
            items.append(content)

    if not items:
        lines = text.splitlines()
        for line in lines:
            line = line.strip()
            m = re.match(r"^\d+[\.\)]\s+(.*)$", line)
            if m:
                content = m.group(1).strip()
                if len(content.split()) >= 3:
                    items.append(content)

    # cap at 12, we'll slice to 10 later
    return items[:12]


# ===== fallback =====
def fallback_objectives(text: str) -> List[str]:
    """
    Fallback objectives when the LLM output cannot be parsed.
    Produces clean, generic academic objectives (12 items).
    """
    if not text or len(text.split()) < 50:
        return [
            "To conduct a comprehensive review of existing literature relevant to the project domain.",
            "To identify key challenges, limitations, and open research gaps reported in prior studies.",
            "To design a conceptual framework that systematically addresses the identified research gaps.",
            "To propose a methodological approach suitable for investigating the defined research problem.",
            "To develop an appropriate prototype or model based on the proposed methodology.",
            "To define quantitative and qualitative metrics for evaluating the effectiveness of the proposed solution.",
            "To conduct experiments or simulations to assess the performance of the proposed approach.",
            "To compare the proposed approach with relevant baseline methods reported in the literature.",
            "To analyze the experimental results and interpret their implications in the context of the research problem.",
            "To evaluate the practical feasibility and scalability of the proposed solution.",
            "To formulate recommendations for future research and potential improvements to the proposed approach.",
            "To document the complete research process and outcomes in a structured academic format.",
        ]

    # light topical extraction so fallback is still domain-ish
    words = re.findall(r"[A-Za-z]{4,}", text.lower())
    stopwords = {
        "this",
        "that",
        "these",
        "those",
        "from",
        "with",
        "using",
        "about",
        "related",
        "methods",
        "method",
        "approach",
        "system",
        "model",
        "paper",
        "study",
        "based",
        "data",
        "results",
        "into",
        "within",
        "other",
        "their",
        "there",
        "which",
        "while",
        "where",
        "such",
        "also",
        "have",
        "been",
        "used",
        "than",
        "after",
        "before",
        "into",
    }
    freq: Dict[str, int] = {}
    for w in words:
        if w in stopwords:
            continue
        freq[w] = freq.get(w, 0) + 1

    topics = [k for k, v in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:8]] or ["key themes"]

    def t(i: int) -> str:
        return topics[i % len(topics)]

    return [
        f"To critically analyze existing literature on {t(0)} and synthesize the main findings and limitations.",
        f"To identify and formalize the key research gaps related to {t(1)} that have not been adequately addressed.",
        f"To design a research framework that integrates {t(2)} into the overall problem formulation.",
        f"To develop a methodological pipeline that operationalizes {t(3)} within the project context.",
        f"To implement a prototype or experimental setup that incorporates {t(4)} as a core component.",
        f"To define appropriate evaluation metrics for assessing the effectiveness of the proposed approach on {t(5)}.",
        f"To conduct empirical experiments focusing on {t(6)} and collect performance evidence.",
        f"To compare the proposed solution against baseline approaches commonly used for {t(7)}.",
        f"To analyze the empirical results to understand the impact of {t(0)} on system performance.",
        f"To examine the robustness of the proposed approach for scenarios involving {t(1)}.",
        f"To derive practical insights and recommendations for practitioners working with {t(2)}.",
        f"To document the contributions of the study regarding {t(3)} and outline future research directions.",
    ]


# ===== AI objective generation =====
def ai_generate_objectives(littext: str, domain: str, project: str) -> List[str]:
    """
    Main generator. Guarantees non-empty list.
    We will ultimately show exactly 10 to the user.
    """
    littext = (littext or "").strip()
    if not littext:
        # no literature → but still return safe objectives
        return fallback_objectives("")

    prompt = f"""
You are an expert academic research mentor.

Project: {project}
Domain: {domain}

You are given TOP-RANKED research papers from the project's literature review.
Each paper is labeled [PAPER i].

GUIDELINES:
- Base each objective on the provided papers.
- Objectives should reflect methods, models, evaluation strategies, or gaps
  mentioned in the papers.
- You MAY combine insights from multiple papers.
- If unsure, write conservative, literature-aligned objectives.
- Avoid generic textbook objectives.

INPUT PAPERS:
{littext}

TASK:
- Generate EXACTLY 12 academic project objectives.
- Number them clearly from 1 to 12.
- Each objective must be 1–2 sentences.
- Prefer referencing papers where relevant, but do NOT fail if reference is unclear.
- Output ONLY the numbered list. No explanations.
"""


    raw = call_ollama(prompt).strip()

    

    # If Ollama died or empty -> fallback
    if not raw:
        objs = fallback_objectives(littext)
        return objs

    if "INSUFFICIENT_LITERATURE" in raw.upper():
        # Instead of failing → still return good generic objectives
        return fallback_objectives(littext)

    parsed = parse_numbered_list(raw)

    if not parsed:
        # Try a simpler backup prompt once
        backup_prompt = f"""
Write 12 academic research objectives for the project "{project}" in domain "{domain}".
Use the following literature:

{littext[:5000]}

Format:
1. Objective ...
2. Objective ...
...
12. Objective ...
"""
        raw2 = call_ollama(backup_prompt).strip()
        if not raw2 or "INSUFFICIENT_LITERATURE" in raw2.upper():
            return fallback_objectives(littext)
        parsed = parse_numbered_list(raw2)

    if not parsed:
        parsed = fallback_objectives(littext)

    # ensure at least 10 items (top up with fallback if needed)
    if len(parsed) < 10:
        fb = fallback_objectives(littext)
        for o in fb:
            if len(parsed) >= 10:
                break
            if o not in parsed:
                parsed.append(o)

    # cap at 12, UI will slice to 10
    return parsed[:12]


# simple alias used by app.py if needed
def generate_objectives(littext: str, domain: str, project: str) -> List[str]:
    return ai_generate_objectives(littext, domain, project)


# ===== improve wording =====
def improve_objectives(objs: List[str], project: str, domain: str) -> List[str]:
    if not objs:
        return []

    prompt = f"""
Improve the academic wording for each objective, keeping the same meaning.

Project: {project}
Domain: {domain}

OBJECTIVES (JSON array):
{json.dumps(objs, indent=2)}

Return a numbered list (1., 2., ...) of rewritten objectives.
"""
    raw = call_ollama(prompt)
    parsed = parse_numbered_list(raw)
    return parsed if parsed else objs


# ===== clustering =====
def cluster_objectives(objectives: List[str]) -> Dict[str, List[str]]:
    if not objectives:
        return {}

    if not embedder or not util:
        cls: Dict[str, List[str]] = {}
        for o in objectives:
            key = " ".join(o.split()[:3]) + "."
            cls.setdefault(key, []).append(o)
        return cls

    try:
        vecs = embedder.encode(objectives, convert_to_tensor=True)
    except Exception:  # noqa: BLE001
        return {o[:20] + ".": [o] for o in objectives}

    clusters: Dict[str, List[str]] = {}
    for i, obj in enumerate(objectives):
        emb = vecs[i]
        best = None
        best_s = 0.0
        for label, _items in clusters.items():
            try:
                rep = embedder.encode([label], convert_to_tensor=True)[0]
                s = float(util.cos_sim(emb, rep))  # type: ignore[arg-type]
            except Exception:  # noqa: BLE001
                s = 0.0
            if s > best_s:
                best_s = s
                best = label

        if best and best_s > 0.55:
            clusters[best].append(obj)
        else:
            clusters[obj[:25] + "."] = [obj]
    return clusters


# ===== Streamlit Page =====
def objectives_page(user: Dict[str, Any], active_project: Dict[str, Any]):
    st.markdown("## 🎯 AI Objectives Generator (Literature-Driven)")

    ap_pid = safe_get(active_project, "project_id", "")
    ap_pname = safe_get(active_project, "project_name", "")
    ap_domain = safe_get(active_project, "domain", "")

    lit = None
    pid = ""
    pname = ""
    domain = ""

    # --- resolve which literature record to use ---
    if ap_pid:
        lit = research_col.find_one({"project_id": ap_pid}, sort=[("created_at", -1)])
        if lit:
            pid = ap_pid
            pname = lit.get("project_name", ap_pname or "Untitled Project")
            domain = lit.get("domain", ap_domain or "")
        else:
            alt = research_col.find_one({}, sort=[("created_at", -1)])
            if alt:
                lit = alt
                pid = alt.get("project_id", "")
                pname = alt.get("project_name", "Untitled Project")
                domain = alt.get("domain", "")
                st.warning(
                    f"No literature for active project '{ap_pname}'. Using latest review for '{pname}'."
                )
            else:
                st.warning("No literature found. Run Literature Review first.")
                return
    else:
        alt = research_col.find_one({}, sort=[("created_at", -1)])
        if alt:
            lit = alt
            pid = alt.get("project_id", "")
            pname = alt.get("project_name", "Untitled Project")
            domain = alt.get("domain", "")
            st.warning("No active project. Using latest literature review.")
        else:
            st.warning("No literature found. Run Literature Review first.")
            return

    st.info(f"Project: **{pname}**  •  Domain: **{domain}**")

    if "generated_objectives" not in st.session_state:
        st.session_state["generated_objectives"] = []

    # --- show and edit already saved final objectives (from prior run) ---
    saved = objectives_col.find_one({"project_id": pid})
    saved_objs = saved.get("objectives", []) if saved else []

    if saved_objs:
        st.markdown("### 📌 Saved Final Objectives (editable)")
        new_objs: List[str] = []
        for i, o in enumerate(saved_objs, 1):
            v = st.text_input(f"{i}.", o or "", key=f"obj_{i}")
            new_objs.append(v or "")

        c1, c2 = st.columns(2)
        if c1.button("💾 Save Changes"):
            objectives_col.update_one(
                {"project_id": pid},
                {"$set": {"objectives": new_objs}},
                upsert=True,
            )
            st.success("Saved updated objectives.")
            st.rerun()

        if c2.button("🗑 Delete All Objectives"):
            objectives_col.update_one(
                {"project_id": pid},
                {"$set": {"objectives": []}},
                upsert=True,
            )
            st.warning("Deleted all objectives.")
            st.rerun()

        st.markdown("### 🔍 Objective Clusters (by meaning)")
        cl = cluster_objectives(saved_objs)
        for label, items in cl.items():
            st.markdown(f"#### 📂 {label}")
            for it in items:
                st.markdown(f"- {it}")

        st.markdown("---")

    # --- flatten literature into text for DeepSeek ---
    raw_results = lit.get("results") if isinstance(lit, dict) else []
    papers_list = raw_results if isinstance(raw_results, list) else []

    papers = sorted(
        papers_list,
        key=lambda x: float(x.get("score", 0.0)) if isinstance(x, dict) else 0.0,
        reverse=True,
    )

    TOP_PAPERS_FOR_OBJECTIVES = 4
    selected_papers = papers[:TOP_PAPERS_FOR_OBJECTIVES]

    text = ""
    for i, p in enumerate(selected_papers, 1):
        title = safe_get(p, "title", "")
        summary = safe_get(p, "summary", "")
        abstract = safe_get(p, "abstract", "")
        source = safe_get(p, "source", "")
        year = safe_get(p, "year", "")

        text += (
            f"[PAPER {i}]\n"
            f"Title: {title}\n"
            f"Year: {year}\n"
            f"Source: {source}\n"
            f"Summary: {summary}\n"
            f"Abstract: {abstract}\n\n"
        )


    st.success(f"Literature review loaded from project **{pname}**.")

    # --- generation trigger ---
    if st.button("✨ Generate 10 Objectives from Literature (DeepSeek AI)"):
        with st.spinner("Generating objectives…"):
            objs = ai_generate_objectives(text, domain, pname)
        # ALWAYS have something
        if not objs:
            objs = fallback_objectives(text)
        # show exactly 10
        objs = (objs or [])[:10]
        st.session_state["generated_objectives"] = objs
        st.success("Objectives generated!")

    objs: List[str] = st.session_state.get("generated_objectives", [])
    if not objs:
        return

    st.markdown("### ✅ Select at least 5 objectives to keep (Final Objectives)")

    selected: List[str] = []
    for i, o in enumerate(objs, 1):
        if st.checkbox(o, key=f"chk_{i}"):
            selected.append(o)

    st.markdown(f"Selected: **{len(selected)} / 5 required**")

    if selected and st.button("✨ Improve Wording of Selected"):
        with st.spinner("Improving objectives…"):
            better = improve_objectives(selected, pname, domain)
        if better:
            # again keep only 10 for UI
            st.session_state["generated_objectives"] = better[:10]
            st.success("Improved objectives generated.")
            st.rerun()

    if st.button("💾 Save Selected as Final Objectives"):
        if len(selected) < 5:
            st.error("Select at least 5 objectives.")
        else:
            objectives_col.update_one(
                {"project_id": pid},
                {"$set": {"objectives": selected}},
                upsert=True,
            )
            st.success("Final objectives saved for this project.")
            st.session_state["generated_objectives"] = []
            st.rerun()


# ===== OPTIONAL AUTO-RUN WHEN USED AS A SINGLE PAGE APP =====
# ===== OPTIONAL AUTO-RUN WHEN USED AS A SINGLE PAGE APP =====
if __name__ == "__main__":
    st.set_page_config(page_title="Objectives Generator", layout="centered")
    st.title("Objectives Generator")

    user = st.session_state.get("user")
    active_project = st.session_state.get("active_project")

    if not user:
        st.error("No user in session. Please login from main app.")
    elif not active_project:
        st.error("No active project selected. Please open this page from Dashboard.")
    else:
        objectives_page(user, active_project)

