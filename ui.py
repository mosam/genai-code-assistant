import streamlit as st
import random
import time
from embedding import create_embeddings, find_top_snippets
from gemini_client import ask_gemini

def run_ui():
    st.title("⚡ GenAI Codebase Assistant")

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

    col1, col2 = st.columns(2)
    with col1:
        model_choice = st.selectbox("🤖 Choose AI model:", 
                                    ["models/gemini-flash-latest", "models/gemini-2.5-pro"])
    with col2:
        style_choice = st.selectbox("📝 Answer style:", 
                                    ["Concise", "Detailed", "Step-by-Step"])

    query = st.text_input("💬 Ask a question about your codebase:")

    style_instruction = {
        "Concise": "Answer briefly and focus only on the relevant part of the code.",
        "Detailed": "Explain thoroughly with context, covering all important aspects.",
        "Step-by-Step": "Explain step by step, breaking down the logic clearly."
    }[style_choice]

    contexts = []
    if query and file_contents:
        if "embeddings" not in st.session_state:
            st.session_state["embeddings"] = create_embeddings(file_contents)  # Cache embeddings in session state
        embeddings = st.session_state["embeddings"]
        contexts = find_top_snippets(query, file_contents, embeddings)  # Find top relevant snippets based on precomputed embeddings
        if not contexts:
            st.warning("⚠️ No relevant code snippets found. Try rephrasing your question or uploading more files.")
        else:
            st.markdown("### 🔎 Top 3 Relevant Snippets")    

        for i, (snippet, score) in enumerate(contexts, 1):
            st.subheader(f"Snippet {i} (Similarity: {score:.3f})")
            st.code(snippet[:200] + "..." if len(snippet)>200 else snippet, language="text")

            with st.spinner(random.choice([
                "⚡ Analyzing your code... please wait"
            ])):
                try:
                    response = ask_gemini(model_choice, query, contexts, style_instruction)
                    st.success("✅ Analysis complete!")
                    st.markdown("### 💡 AI Answer")
                    st.markdown(response.text)
                except Exception as e:
                    st.error(f"Error: {e}")
