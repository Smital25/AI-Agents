from __future__ import annotations

from typing import Dict, List
import pandas as pd


def to_bibtex(items: List[Dict]) -> str:
    out = []
    for i, it in enumerate(items, 1):
        key = f"item{i}"
        authors = " and ".join(it.get("authors") or [])
        title = (it.get("title") or "").replace("{", " ").replace("}", " ")
        year = it.get("year") or ""
        venue = it.get("venue") or ""
        doi = it.get("doi") or ""
        url = it.get("url") or it.get("pdf_url") or ""
        out.append(
            f"@article{{{key},\n"
            f"  title={{ {title} }},\n"
            f"  author={{ {authors} }},\n"
            f"  journal={{ {venue} }},\n"
            f"  year={{ {year} }},\n"
            f"  doi={{ {doi} }},\n"
            f"  url={{ {url} }}\n"
            f"}}\n"
        )
    return "\n".join(out)


def to_dataframe(items: List[Dict]) -> pd.DataFrame:
    return pd.DataFrame(items)
