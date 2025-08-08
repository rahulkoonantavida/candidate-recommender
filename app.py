import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import openai
import streamlit as st
import pandas as pd
import PyPDF2
import re
from summary import summarize_fit
from embeddings import embed_text
from preprocessing import clean_text
from sklearn.metrics.pairwise import cosine_similarity

### === STREAMLIT WEBAPP SETUP ===
# ---- Page config ----
st.set_page_config(page_title="candidate-recommender", layout="wide", initial_sidebar_state="expanded")
# ---- Title ----
st.title("Search for the Most Competitive Candidates!")

# ---- Sidebar inputs ----
st.sidebar.header("Enter Input Details:")

# OPTIONAL OPENAI API KEY FOR LLM SUMMARIES
openai_api_key = st.sidebar.text_input(
    "🔑 OpenAI API Key (optional for AI summaries)",
    type="password",
    placeholder="sk-…",
)
if openai_api_key:
    openai.api_key = openai_api_key

# JOB DESCRIPTION INPUT FIELD
job_desc = st.sidebar.text_area(
    "Job Description",
    placeholder="Paste the job description here...",
    height=200,
)

# RESUME LIST INPUT FORMAT SELECTOR
input_method = st.sidebar.radio(
    "How would you like to provide resumes?",
    ("Upload PDFs/TXTs", "Paste as text"),
    horizontal=True
)

# to store cleaned resume text and resume ids
texts, ids = [], []
# to store full resume text for optional LLM summaries
fulltexts = []

# EXTRACT TEXT FROM UPLOADED RESUMES
# accepts .PDF/.TXT
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
            fulltexts.append(raw)
            texts.append(clean_text(raw))
            ids.append(f.name)

# EXTRACT TEXT FROM PLAINTEXT RESUMES
# expects '---' delimiter
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
            texts.append(clean_text(txt))
            fulltexts.append(txt)

# MAIN APP FUNCTIONALITY
# 1. compute embeddings for job description, and each resume
# 2. compute cosine similarity between job desc. and each resume
# 3. return top 10 resumes
# 4. (IF API KEY PROVIDED) return LLM summary for top 3 resumes

if st.sidebar.button("Run Recommendation"):
    if not job_desc:
        st.sidebar.error("Please enter a job description.")
    elif not uploaded_files:
        st.sidebar.error("Please upload at least one resume.")
    else:   
        # build combined list: job_desc, then all resumes
        all_texts = [clean_text(job_desc)] + texts

        # embed everything in one go (model only loads once)
        with st.spinner("Computing embeddings…"):
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

        # define a function that returns a CSS style string for each cell
        def color_similarity(val):
            if val > 0.8:
                color = "#2ecc71"   # dark green
            elif val > 0.6:
                color = "#a3e635"   # light green
            elif val > 0.4:
                color = "#facc15"   # yellow
            else:
                color = "#ef4444"   # red
            return f'background-color: {color}; color: black;'

        # display top-k
        top_k = min(10, len(df))
        top_df = df.head(top_k)
        styled = top_df.style.applymap(color_similarity, subset=["Similarity"])
        st.subheader(f"Top {top_k} Candidates")
        st.dataframe(styled, height=300)

        # retrieve dict of ids and full resume texts
        id2text = dict(zip(ids, fulltexts))

        # generate LLM summaries for top 3 candidates
        if openai_api_key:
            st.markdown("AI-Generated Summaries for Top 3 Candidates")
            for _, r in df.head(3).iterrows():
                cid = r["Candidate"]
                ctext = id2text[cid]
                with st.spinner(f"Summarizing fit for {cid}…"):
                    summary = summarize_fit(job_desc, ctext)
                st.write(f"**{cid}:** {summary}")
        else:
            st.info("Enter your OpenAI API key to see AI-generated summaries.")

else:
    st.sidebar.info("Fill in the inputs and click **Run Recommendation**")

# ---- Footer ----
st.markdown("---")
st.markdown("Built with ❤️ and Streamlit")
st.markdown("Built by Rahul Koonantavida (R16)")