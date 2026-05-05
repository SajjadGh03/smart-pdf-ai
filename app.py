import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from openai import OpenAI


client = OpenAI(base_url='https://api.gapgpt.app/v1', api_key='your API key')


# Load PDF and extract text
def load_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


# Split text into smaller chunks
def chunk_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]


# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2',local_files_only=True)

# Create FAISS index from embeddings
def create_index(chunks):
    embeddings = model.encode(chunks)
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    return index


# Retrieve relevant chunks based on query
def retrieve(query, index, chunks, k=3):
    q_emb = model.encode([query])
    D, I = index.search(np.array(q_emb), k)
    return [chunks[i] for i in I[0]]


# Generate answer using LLM
def ask_llm(context, question):
    prompt = f"""
    Answer based on context:

    {context}

    Question: {question}
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]

    )

    return response.choices[0].message.content


# Insight mode
def generate_insight(context):
    prompt = f"""
    Analyze this:

    {context}

    Give:
    - Summary
    - Key points
    - 3 questions
    - Ambiguities
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


# Streamlit UI
st.title("📄 Smart PDF AI")

file = st.file_uploader("Upload PDF")

if file:
    text = load_pdf(file)
    chunks = chunk_text(text)
    index = create_index(chunks)

    question = st.text_input("Ask a question")

    if question:
        context = retrieve(question, index, chunks)

        answer = ask_llm(context, question)
        insight = generate_insight(" ".join(context))

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Insight 🔥")
        st.write(insight)