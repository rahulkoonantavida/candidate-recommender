# [candidate-recommender](https://candidate-recommender-q2koucsop7gtgtzzw4kw3y.streamlit.app)

a simple web app that recommends the best candidates for a job based on relevance

## Project Structure

```
.
├── app.py             # Main Streamlit application
├── embeddings.py      # Generates and manages text embeddings
├── preprocessing.py   # Parses and preprocesses input data into clean text
├── summary.py         # Calls OpenAI API for generating candidate fit summaries
├── requirements.txt   # Project dependencies
├── ...
└── README.md
```

## Installation & Usage

1. **Clone the repository**
   
   ```
   git clone https://github.com/rahulkoonantavida/candidate-recommender.git
   cd candidate-recommender
   ```
3. **Create a virtual environment**
   
   ```
   python3 -m venv venv
   source venv/bin/activate
   ```
5. **Install dependencies**
   
   ```
   pip install -r requirements.txt
   ```
7. **Launch!**
   
   ```
   streamlit run app.py
   ```

## Approach

- Streamlit and its widgets enable an effortless way to accept text and file inputs for job descriptions and resumes
- ```re``` and ```nltk``` enable the removal of noise from the input data; stop words and other irrelevant information are excised
- ```SentenceTransformers``` enables the use of pretrained models to calculate embeddings for the cleaned data efficiently
- ```cosine_similarity``` provides a metric to quantize the semantic similarity between embeddings
- Using ```pandas```, the top 10 candidates are displayed with color coded similarity scores (ranked dark green, light green, yellow, red)
- Optionally, the OpenAI API is used to generate a concise summary regarding the fit of the top 3 candidates

## Assumptions

- Input is well formed (typical job descriptions and ATS-friendly resumes)
- User is able to locally manage input data (no database functionality for job descriptions or resumes)
- User is able to provide their own OpenAI API key, if AI summary functionality is desired

## Additional Notes

- attempted to implement section level embeddings and weighted embeddings to enhance candidate ranking capabilities, but encountered challenges with consistently parsing the various sections of a resume correctly (i.e. experience, projects, skills, etc.)
- on rare occurences, the OpenAI API generates a response on why a candidate is NOT a good fit for a particular role; this is not necessarily a bad thing, but should be leveraged in a more suitable context, rather than during a summary for the top 3 candidate recommendations

## Potential Improvements

- improve resume parsing capabilities to consistently divide resume sections
- implement section level embeddings and weighted embeddings
- test performance with slower, but more robust SBERT model (all-mpnet-base-v2)
- attempt to extract more detailed semantic information, rather than rely solely on cosine similarity
