from __future__ import annotations

import io
import requests
import fitz
from pdfminer.high_level import extract_text as pdfminer_extract

HEADERS = {"User-Agent": "LitReview-Agent/0.1"}


def best_pdf_url(item: dict) -> str | None:
    if item.get("pdf_url"):
        return item["pdf_url"]
    if isinstance(item.get("url"), str) and item["url"].lower().endswith(".pdf"):
        return item["url"]
    return None


def fetch_pdf_bytes(url: str, timeout: int = 60) -> bytes | None:
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    if r.status_code == 200:
        return r.content
    return None


def extract_text_from_pdf_bytes(data: bytes) -> str:
    try:
        with fitz.open(stream=data, filetype="pdf") as doc:
            return "\n".join(page.get_text("text") for page in doc)
    except:
        try:
            with io.BytesIO(data) as f:
                return pdfminer_extract(f)
        except:
            return ""
