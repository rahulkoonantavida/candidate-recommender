import streamlit as st
import pandas as pd
import PyPDF2
import re

from embeddings import embed_text
from preprocessing import clean_text
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
        st.sidebar.error("Please upload at least one resume.")
    else:
        # extract text from each resume
        texts = []
        ids = []
        for f in uploaded_files:
            raw = ""
            if f.type == "application/pdf":
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    raw += page.extract_text() or ""
            else:
                raw = f.read().decode("utf-8")
            texts.append(clean_text(raw))
            ids.append(f.name)
            
        # build combined list: job_desc, then all resumes
        all_texts = [clean_text(job_desc)] + texts

        # embed everything in one go (model only loads once)
        embeddings = embed_text(all_texts)

        # split job embedding vs. resume embeddings
        job_emb     = embeddings[0].reshape(1, -1)   # shape (1, dim)
        resume_embs = embeddings[1:]                # shape (n_resumes, dim)

        # compute cosine similarities (returns array of shape (1, n_resumes))
        sims = cosine_similarity(job_emb, resume_embs)[0]

        # build DataFrame and sort candidates
        df = pd.DataFrame({
            "Candidate": ids,
            "Similarity": sims,
        }).sort_values("Similarity", ascending=False)

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