import numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
import streamlit as st

@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedder = load_embedder()

def embed_text(text: str):
    return embedder.encode([text])[0]

def find_top_snippets(query: str, file_contents: list, top_k: int = 3):
    query_vec = embed_text(query)
    similarities = [
        np.dot(query_vec, embed_text(content)) / (norm(query_vec) * norm(embed_text(content)))
        for content in file_contents
    ]
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [(file_contents[i], similarities[i]) for i in top_indices]
