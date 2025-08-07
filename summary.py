import streamlit as st
import openai
from typing import List

@st.cache_data(show_spinner=False)
def summarize_fit(job_desc: str, resume_text: str) -> str:
    prompt = [
        {"role": "system", "content": "You are an expert recruiter."},
        {
            "role": "user",
            "content": (
                "Here is a job description:\n\n"
                f"{job_desc}\n\n"
                "Here is a candidate’s resume text:\n\n"
                f"{resume_text}\n\n"
                "In one concise sentence, explain why this candidate is a great fit."
            ),
        },
    ]
    resp = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=prompt,
        temperature=0.2,
        max_tokens=70,
    )
    return resp.choices[0].message.content.strip()