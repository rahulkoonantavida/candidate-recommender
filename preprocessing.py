import re

SECTION_HEADERS = [
    "experience", "work experience", "professional experience",
    "education", "skills", "projects", "certifications",
]

def clean_text(raw: str) -> str:
    # Remove URLs
    text = re.sub(r'https?://\S+', '', raw)
    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)
    # Remove phone numbers (simple U.S. pattern)
    text = re.sub(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', '', text)
    # Remove “Page X of Y” footers
    text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)
    # Squash multiple blank lines
    text = re.sub(r'\n{2,}', '\n\n', text)
    return text.strip()

def extract_sections(text: str) -> dict[str, str]:
    """
    Returns a dict mapping section header → section text.
    Unrecognized text before the first known header goes under 'header'.
    """
    lines = text.splitlines()
    sections = {}
    current = "header"
    buffer = []

    for line in lines:
        lstr = line.strip().lower()
        if lstr in SECTION_HEADERS:
            # save the previous
            sections[current] = "\n".join(buffer).strip()
            current = lstr
            # save all forms of experience as "experience"
            if current == "work experience" or "professional experience":
                current = "experience"
            buffer = []
        else:
            buffer.append(line)
    sections[current] = "\n".join(buffer).strip()
    return sections

def preprocess_for_embedding(raw_resume: str) -> str:
    # clean text
    clean = clean_text(raw_resume)
    # extract sections
    secs  = extract_sections(clean)
    # 3. Reconstruct a weighted text blob
    weighted = (
        (secs.get("professional experience", "") * 2 or secs.get("experience", "") * 2 or secs.get("work experience", "") * 2)
      + (secs.get("projects", "")                     )
      + (secs.get("skills", "")                       )
      + (secs.get("education", "")                    )
    )
    return weighted or clean  # fallback to full clean text