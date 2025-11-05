import os
import pandas as pd
import streamlit as st

from backend.agent import LitReviewAgent
from backend.refs import to_bibtex, to_dataframe

st.set_page_config(page_title="LitReview AI Agent", layout="wide")

st.title("📚 LitReview AI Agent")
st.caption("Streamlit + Python · AIML-first retrieval, ranking, clustering, and synthesis")

with st.sidebar:
    st.header("Settings")
    pdf = st.toggle("Fetch & parse PDFs (slower)", value=True)
    max_docs = st.slider("Max docs in synthesis", 5, 30, 20, 1)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

query = st.text_input(
    "Enter your project title / topic",
    placeholder="e.g., Vision Transformers for Medical Imaging",
)
go = st.button("Generate Literature Review", type="primary")

if go and query.strip():
    agent: LitReviewAgent = LitReviewAgent(enable_pdf=pdf)
    with st.spinner("Running agent: planning → searching → ranking → parsing → writing..."):
        result = agent.run(query)
        items = result["items"][:max_docs]
        result["items"] = items

    st.subheader("Synthesis")
    st.markdown(f"### {result['title']}")
    st.write(result["body"])

    st.subheader("Topics (Ontology tags)")
    chips = ", ".join(result.get("topics") or []) or "(none)"
    st.write(chips)

    st.subheader("Corpus")

    df = to_dataframe(items)
    df["Author / Year"] = df.apply(lambda r: f"{', '.join(r['authors'])} / {r['year']}", axis=1)
    df["View Page"] = df["pdf_url"].fillna(df["url"])

    st.dataframe(
        df[["Author / Year", "title", "findings", "limitations", "View Page"]],
        use_container_width=True
    )

    st.subheader("References")
    st.markdown("\n".join(result["references"]))

    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            "Download Markdown",
            data=f"# {result['title']}\n\n{result['body']}\n\n" + "\n".join(result["references"]),
            file_name="literature_review.md",
        )
    with col2:
        st.download_button("Download BibTeX", data=to_bibtex(items), file_name="references.bib")
    with col3:
        csv = df.to_csv(index=False)
        st.download_button("Download CSV Corpus", data=csv, file_name="corpus.csv")
else:
    st.info("Enter a topic and click Generate.")
