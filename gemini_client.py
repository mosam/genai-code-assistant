import os
from dotenv import load_dotenv
from google.genai import Client

load_dotenv()
client = Client(api_key=os.getenv("GOOGLE_API_KEY"))

def ask_gemini(model_choice, query, contexts, style_instruction):
    context_text = "\n\n".join(
        [f"Snippet {i+1} (Similarity: {score:.3f}):\n{snippet}"
         for i, (snippet, score) in enumerate(contexts)]
    )
    return client.models.generate_content(
        model=model_choice,
        contents=[
            "You are a helpful coding assistant.",
            f"Question:\n{query}",
            f"Relevant code snippets with similarity scores:\n{context_text}",
            style_instruction
        ]
    )
