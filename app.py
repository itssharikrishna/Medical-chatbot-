from flask import Flask, render_template, request, jsonify
from pathlib import Path
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import pinecone
from langchain_ollama import OllamaLLM
import os
from dotenv import load_dotenv

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# -----------------------------
# Initialize Flask app
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Initialize Pinecone
# -----------------------------
pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
index_name = "medical-chatbot"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # embedding dimension must match model
        metric="cosine",
        spec=pinecone.ServerlessSpec(cloud="aws", region="us-east-1")
    )
pinecone_index = pc.Index(index_name)

# -----------------------------
# Load PDFs and embed
# -----------------------------
def load_pdf_files(folder_path):
    docs = []
    for pdf_file in Path(folder_path).glob("*.pdf"):
        reader = PdfReader(str(pdf_file))
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        docs.append({"text": text, "source": str(pdf_file)})
    return docs

def split_text_into_chunks(docs, chunk_size=200, chunk_overlap=50):
    chunks = []
    for doc in docs:
        text = doc["text"]
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            chunks.append({"text": chunk_text, "source": doc["source"]})
            start += chunk_size - chunk_overlap
    return chunks

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
extracted_data = load_pdf_files("data/")
text_chunks = split_text_into_chunks(extracted_data)
texts = [chunk["text"] for chunk in text_chunks]

# -----------------------------
# Upsert embeddings into Pinecone
# -----------------------------
batch_size = 50
vectors_to_upsert = [
    {
        "id": str(i),
        "values": embedding_model.encode([texts[i]], convert_to_numpy=True)[0].tolist(),
        "metadata": {"text": texts[i][:500], "source": text_chunks[i]["source"]}
    }
    for i in range(len(texts))
]

for i in range(0, len(vectors_to_upsert), batch_size):
    batch = vectors_to_upsert[i:i + batch_size]
    pinecone_index.upsert(vectors=batch)

print("Pinecone index ready with all documents!")

# -----------------------------
# Initialize Ollama LLM
# -----------------------------
llm = OllamaLLM(model="phi3:mini")
# Ensure Ollama server is running locally

# -----------------------------
# Question answering function
# -----------------------------
def answer_question(query, top_k=5):
    try:
        # Encode query
        query_vector = embedding_model.encode([query], convert_to_numpy=True)[0].tolist()

        # Query Pinecone
        results = pinecone_index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )

        matches = results.matches

        if not matches or len(matches) == 0:
            return "No relevant context found."

        # --- AUTOMATIC METADATA FIELD DETECTION ---
        first_meta = matches[0].metadata
        
        if not first_meta:
            return "No metadata found in vector store."

        # Pick the first text-like field (common keys)
        possible_keys = ["text", "chunk", "content", "document", "page_content"]

        # Pick a valid key if available, otherwise auto-pick the first one
        field = None
        for k in possible_keys:
            if k in first_meta:
                field = k
                break

        if field is None:
            # fallback: choose the first field in metadata
            field = list(first_meta.keys())[0]

        # Build context
        context = "\n".join([
            match.metadata.get(field, "")
            for match in matches
        ])
        # -----------------------------------------------------

        # Create prompt
        prompt = f"""
You are a helpful medical assistant.
Relevant Context:
{context}

User Question:
{query}

Answer clearly and medically accurate:
"""

        # LLM call
        response = llm.invoke(prompt)
        return response.strip()

    except Exception as e:
       print("Error in answer_question:", e)
       return f"Backend Error: {e}"




# -----------------------------
# Flask Routes
# -----------------------------
@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    try:
        msg = request.form.get("msg")
        if not msg:
            return jsonify({"answer": "Error: No message received"}), 400

        print("User Input:", msg)
        response = answer_question(msg)
        print("Response (first 200 chars):", response[:200])
        return jsonify({"answer": response})

    except Exception as e:
        print("Error in /get:", e)
        return jsonify({"answer": "Server error occurred"}), 500

# -----------------------------
# Run Flask App
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
