#vector matching for semantic search
import numpy as np
#used for cosine formula
from numpy.linalg import norm
#convert text to embeddings
from sentence_transformers import SentenceTransformer
#ui+caching
import streamlit as st



#Load the embedding model which convert text into vectorsand cache it to avoid reloading on every run
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")
embedder = load_embedder()

#Convert text into a numeric vector representation (embedding)
def embed_text(text: str):
    return embedder.encode(text)

#create embeddings only once(performance optimization) and cache it to avoid reloading on every run
def create_embeddings(file_contents: list):
    embeddings=[]
    for content in file_contents:
        emb = embed_text(content)
        embeddings.append(emb)
    return embeddings

#cosine similarity(clean, reusable function)
def cosine_similarity(vec1, vec2):
    if norm(vec1) == 0 or norm(vec2) == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))

#find top-k relevent snippets
def find_top_snippets(query: str, file_contents: list, embeddings: list, top_k: int = 3):
    query_vec = embed_text(query) #user question converted to vector
    similarities=[]
    #compare with precomputed embeddings
    for emb in embeddings:
        score = cosine_similarity(query_vec, emb)
        similarities.append(score)

    #argsort: sort indices, [::-1] reverse order, [:top_k] take top k (sort and pick top k )
    top_indices = np.argsort(similarities)[::-1][:top_k]
    #Filter + return results
    results = []
    for i in top_indices:
        if similarities[i] > 0.2:  # Only include snippets with a positive similarity score
            results.append((file_contents[i], similarities[i]))
        return results    
