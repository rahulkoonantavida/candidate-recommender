import streamlit as st
import pandas as pd
import PyPDF2
import re

from embeddings import embed_text
from preprocessing import resume_to_sections
from sklearn.metrics.pairwise import cosine_similarity

# ---- Page config ----
st.set_page_config(page_title="candidate-recommender", layout="wide", initial_sidebar_state="expanded")

# ---- Title ----
st.title("Candidate Recommendation Engine")

# ---- Sidebar inputs ----
st.sidebar.header("Inputs")
job_desc = st.sidebar.text_area(
    "Job Description",
    placeholder="Paste the job description here...",
    height=200,
)

input_method = st.sidebar.radio(
    "How would you like to provide resumes?",
    ("Upload PDFs/TXTs", "Paste as text"),
    horizontal=True
)

texts, ids = [], []

if input_method == "Upload PDFs/TXTs":
    uploaded_files = st.sidebar.file_uploader(
        "Upload candidate resumes",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )
    if uploaded_files:
        for f in uploaded_files:
            raw = ""
            if f.type == "application/pdf":
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    raw += page.extract_text() or ""
            else:
                raw = f.read().decode("utf-8")
            texts.append(raw)
            ids.append(f.name)

else:  # paste as text
    uploaded_files = st.sidebar.text_area(
        "Paste all resumes here",
        placeholder=(
            "Paste each resume one after another, and separate them with a line\n"
            "containing just three dashes, e.g.:\n\n"
            "Jane Doe\nResume text...\n---\nJohn Smith\nResume text…"
        ),
        height=300,
    )
    if uploaded_files:
        # split on lines that are exactly '---'
        parts = [p.strip() for p in re.split(r"(?m)^\-\-\-$", uploaded_files) if p.strip()]
        for i, part in enumerate(parts, start=1):
            # parse the first line as the candidate name
            lines = part.splitlines()
            name = lines[0].strip() if lines else f"Resume {i}"
            txt = "\n".join(lines[1:]) if len(lines) > 1 else ""
            ids.append(name)
            texts.append(txt)

if st.sidebar.button("Run Recommendation"):
    if not job_desc:
        st.sidebar.error("Please enter a job description.")
    elif not uploaded_files:
        st.sidebar.error("Please input at least one resume.")
    else: 

        # job desc embedding
        jd_embed = embed_text([job_desc])[0]

        # resume embeddings
        results = []
        for name, resume_text in zip(ids, texts):
            # new section-level approach
            sections = resume_to_sections(resume_text)
            if not sections:
                continue

            # MODIFIABLE WEIGHTS
            weights = {
                "experience":            2.0,
                "projects":              1.5,
                "skills":                1.0,
                "education":             1.0,
            }
            # MODIFIABLE WEIGHTS

            texts, ws = [], []
            for sec_name, text in sections.items():
                w = weights.get(sec_name, 1.0)
                if text and w > 0:
                    texts.append(text)
                    ws.append(w)

            embeds = embed_text(texts)

            # compute weighted average
            sims = cosine_similarity([jd_embed], embeds)[0]
            weighted_score = float((sims * ws).sum() / sum(ws))
            results.append((name, weighted_score))

        # rank by score descending
        df = pd.DataFrame(results, columns=["name","weighted_score"]).sort_values("weighted_score", ascending=False)

        # display top candidates
        top_k = min(10, len(df))
        st.subheader(f"Top {top_k} Candidates")
        st.table(df.head(top_k))

        # BONUS: OpenAI API call to GPT???

else:
    st.sidebar.info("Fill in the inputs and click **Run Recommendation**")

# ---- Footer ----
st.markdown("---")
st.markdown("Built with ❤️ and Streamlit")
st.markdown("Built by Rahul Koonantavida (R16)")