from __future__ import annotations

from typing import List
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.text_rank import TextRankSummarizer

LLM_PROVIDER = None
try:
    import os
    if os.getenv("OPENAI_API_KEY"):
        LLM_PROVIDER = "openai"
        from openai import OpenAI
        _client = OpenAI()
except:
    LLM_PROVIDER = None


def textrank_summary(text: str, sent_count: int = 6) -> str:
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    sents = summarizer(parser.document, sent_count)
    return " ".join(str(s) for s in sents)


PROMPT = (
    "You are a scientific writing assistant. Synthesize a literature review from given snippets. "
    "Write concise paragraphs with inline numeric citations like [1], [2]. "
    "Sections: Background, Methods, Key Findings, Trends, Gaps, Datasets. "
    "Stay faithful to evidence."
)


def llm_summarize(snippets: List[str], refs: List[str]) -> str:
    if LLM_PROVIDER == "openai":
        content = "\n\n".join(f"[{i+1}] {refs[i]}\n{snippets[i][:1200]}" for i in range(min(len(snippets), 12)))
        resp = _client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"{PROMPT}\n\n{content}"}],
            temperature=0.2,
        )
        return resp.choices[0].message.content

    return textrank_summary("\n".join(snippets), 10)
