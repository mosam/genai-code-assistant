# ⚡ Description

I built a local, privacy-first Codebase RAG Assistant using Python and Streamlit. To optimize for speed and cost, I used a lightweight open-source embedding model (all-MiniLM-L6-v2) via SentenceTransformers to convert code files into vector embeddings locally, bypassing third-party embedding APIs. I implemented the cosine similarity search algorithm from scratch using NumPy to retrieve the most semantically relevant code snippets. Finally, I orchestrated the retrieved context alongside user queries into the google-genai SDK, allowing developers to chat with their codebase with customizable generation styles (Concise, Detailed, or Step-by-Step

# ⚡ GenAI Codebase Assistant
A Streamlit app that uses Google Gemini + SentenceTransformers to analyze codebases.  
Features:
- Cosine similarity for semantic matching
- Top-3 snippet retrieval
- Progress bar + retry logic
- Multiple answer styles (Concise, Detailed, Step-by-Step)

## 🚀 Setup

```bash
git clone https://github.com/<your-username>/genai-code-assistant.git
cd genai-code-assistant
pip install -r requirements.txt

---------------- Requirement understanding--------------
📖 Explanation
streamlit → for the UI.
sentence-transformers → embedding model (all-MiniLM-L6-v2).
numpy → vector math + cosine similarity.
python-dotenv → load .env file with your API key.
google-genai → Gemini client.
torch, torchvision, torchaudio → required by SentenceTransformers backend.
------------------------------------------------------------------------

Create a .env file in the project root:
GOOGLE_API_KEY=your_api_key_here
 Important: Do not commit .env to GitHub. Keep your API key private.

Run the App:
streamlit run app.py

Usage
Upload one or more code files (.py, .js, .php, .sql, etc.).
Enter a question about your codebase.
The app shows the top‑3 relevant snippets with similarity scores.

Gemini analyzes those snippets and rturns an answer.



Disclaimer
This project is for testing and learning purposes.
