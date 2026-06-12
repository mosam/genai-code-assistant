import streamlit as st
import os
from sentence_transformers import SentenceTransformer
import numpy as np
from google.genai import Client
from dotenv import load_dotenv
import random
import time
from numpy.linalg import norm

# Load environment variables
load_dotenv()
client = Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize embedding model (cached so it loads once)
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

with st.spinner("⚡ Loading embedding model..."):
    embedder = load_embedder()

# Functions
def embed_text(text):
    return embedder.encode([text])[0]

def find_top_snippets(query, file_contents, top_k=3):
    query_vec = embed_text(query)
    similarities = []
    #Dot product → just multiplies and sums vector values. It can be skewed by vector length
    #(longer code/text snippets produce bigger numbers even if they’re not semantically close).
    #similarities = [np.dot(query_vec, embed_text(content)) for content in file_contents]
    #Cosine similarity → normalizes by vector length and measures the angle between vectors.
    # That angle reflects how similar the meanings are, regardless of size.
    for content in file_contents:
        sim = np.dot(query_vec, embed_text(content)) / (norm(query_vec) * norm(embed_text(content)))
        similarities.append(sim)

    # Sort indices by similarity (highest first)
    top_indices = np.argsort(similarities)[::-1][:top_k]
    # Return the top-k snippets
    return [(file_contents[i], similarities[i]) for i in top_indices]

# --- UI ---
st.title("⚡ GenAI Codebase Assistant")

# Upload section
uploaded_files = st.file_uploader(
    "📂 Upload code files",
    type=["py","js","jsx","php","java","cpp","go","rs","sql","yml","md"],
    accept_multiple_files=True
)

file_contents = []
if uploaded_files:
    with st.expander("📄 Uploaded files preview"):
        for file in uploaded_files:
            content = file.read().decode("utf-8", errors="ignore")
            file_contents.append(content)
            st.write(f"✅ {file.name}")
            st.code(content[:200] + "...", language="text")

# Controls section
col1, col2 = st.columns(2)
with col1:
    model_choice = st.selectbox("🤖 Choose AI model:", 
                                ["models/gemini-flash-latest", "models/gemini-2.5-pro"])
with col2:
    style_choice = st.selectbox("📝 Answer style:", 
                                ["Concise", "Detailed", "Step-by-Step"])

query = st.text_input("💬 Ask a question about your codebase:")

# Map style to instructions
if style_choice == "Concise":
    style_instruction = "Answer briefly and focus only on the relevant part of the code."
elif style_choice == "Detailed":
    style_instruction = "Explain thoroughly with context, covering all important aspects."
else:
    style_instruction = "Explain step by step, breaking down the logic clearly."

# --- Processing ---
if query and file_contents:
    contexts = find_top_snippets(query, file_contents, top_k=3)
    st.markdown("### 🔎 Top 3 Relevant Snippets")
    for i, (snippet, score) in enumerate(contexts, 1):
        st.subheader(f"Snippet {i} (Similarity: {score:.3f})")
        st.code(snippet[:200] + "...", language="text")

    messages = [
        "🧠 Thinking hard about your code...",
        "🔧 Fixing bugs in my imagination...",
        "📚 Reading your code like a novel...",
        "⚡ Analyzing your code... please wait"
    ]

    try:
        with st.spinner(random.choice(messages)):
            response = None
            # Prepare context string with scores
            context_text = "\n\n".join(
                [f"Snippet {i+1} (Similarity: {score:.3f}):\n{snippet}" for i, (snippet, score) in enumerate(contexts)]
            )
            progress = st.progress(0)
            for attempt in range(3):
                progress.progress((attempt+1)/3)
                try:
                    response = client.models.generate_content(
                        model=model_choice,
                        contents=[
                            "You are a helpful coding assistant.",
                            f"Question:\n{query}",
                            f"Code snippet:\n" + context_text,
                            style_instruction
                        ]
                    )
                    break
                except Exception as e:
                    st.warning(f"⚠️ Attempt {attempt+1} failed: {e}")
                    time.sleep(5)

            if not response:
                fallback_model = ("models/gemini-2.5-pro" 
                                  if model_choice == "models/gemini-flash-latest" 
                                  else "models/gemini-flash-latest")
                st.info(f"⚡ Switching to fallback model ({fallback_model})...")
                response = client.models.generate_content(
                    model=fallback_model,
                    contents=[
                        "You are a helpful coding assistant.",
                        f"Question:\n{query}",
                        f"Code snippet:\n" + context_text,
                        style_instruction
                    ]
                )

        if response:
            st.success("✅ Analysis complete!")
            st.markdown("### 💡 AI Answer")
            st.markdown(response.text)
        else:
            st.error("❌ All retries failed. Please try again later.")

    except Exception as e:
        st.error(f"Error: {e}")
