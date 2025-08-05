import streamlit as st
import pandas as pd
import PyPDF2

# ---- Page config ----
st.set_page_config(page_title="candidate-recommender", layout="wide")

# ---- Title ----
st.title("Candidate Recommendation Engine")

# ---- Sidebar inputs ----
st.sidebar.header("Inputs")
job_desc = st.sidebar.text_area(
    "Job Description",
    placeholder="Paste the job description here...",
    height=200,
)

uploaded_files = st.sidebar.file_uploader(
    "Upload candidate resumes (PDF/TXT)",
    type=["pdf", "txt"],
    accept_multiple_files=True,
)

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
                # TO DO: extract text
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    raw += page.extract_text() or ""
            else:
                raw = f.read().decode("utf-8")
            texts.append(raw)
            ids.append(f.name)

        # TO DO: embeddings
        # TO DO: compute cosine similarities // sims = [...]

        # dummy data
        df = pd.DataFrame({
            "Candidate": ids,
            "Similarity": [0.0 for _ in ids],
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