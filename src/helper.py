# -----------------------------
# 1️⃣ Imports
# -----------------------------
from langchain_community.llms import Ollama
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from typing import List

# -----------------------------
# 2️⃣ Load PDF Files
# -----------------------------
from PyPDF2 import PdfReader

def load_pdf_file(directory_path: str) -> List[dict]:
    """
    Load all PDFs from a directory and return a list of dicts with text and source.
    """
    pdf_texts = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory_path, filename)
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            pdf_texts.append({"text": text, "source": filename})
    return pdf_texts

# -----------------------------
# 3️⃣ Split Text into Chunks
# -----------------------------
def text_split(pdf_texts: List[dict], chunk_size=500, chunk_overlap=20) -> List[dict]:
    """
    Split each PDF text into smaller chunks with overlap.
    """
    chunks = []
    for doc in pdf_texts:
        text = doc["text"]
        source = doc["source"]
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            chunks.append({"text": chunk_text, "source": source})
            start += chunk_size - chunk_overlap
    return chunks

# -----------------------------
# 4️⃣ Create Embeddings
# -----------------------------
def create_embeddings(chunks: List[dict]):
    """
    Generate embeddings using SentenceTransformer.
    """
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
    return embeddings, chunks

# -----------------------------
# 5️⃣ Build FAISS Index
# -----------------------------
def build_faiss_index(embeddings):
    """
    Build FAISS index for similarity search.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

# -----------------------------
# 6️⃣ Initialize Ollama LLM
# -----------------------------
llm = Ollama(model="llama3")

# -----------------------------
# 7️⃣ Retrieve & Answer Questions
# -----------------------------
def answer_question(query: str, embeddings_model, index, chunks, top_k=3):
    # 1️⃣ Embed query
    query_vec = embeddings_model.encode([query], convert_to_numpy=True)
    
    # 2️⃣ Retrieve top-k similar chunks
    D, I = index.search(np.array(query_vec), top_k)
    context = " ".join([chunks[i]["text"] for i in I[0]])
    
    # 3️⃣ Build prompt
    system_prompt = (
        "You are a medical assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. "
        "Use three sentences maximum and keep the answer concise.\n\n"
        f"Context: {context}\nQuestion: {query}"
    )
    
    # 4️⃣ Generate answer
    result = llm.generate([system_prompt])  # ✅ list of strings
    
    # 5️⃣ Extract text
    answer = result.generations[0][0].text
    return answer

# -----------------------------
# ✅ Example Usage
# -----------------------------
# Load PDFs
pdf_texts = load_pdf_file("path_to_pdf_directory")  # your folder path

# Split into chunks
chunks = text_split(pdf_texts)

# Generate embeddings
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings, chunks = create_embeddings(chunks)

# Build FAISS index
faiss_index = build_faiss_index(embeddings)

# Ask a question
query = "What is Acne?"
response = answer_question(query, embedding_model, faiss_index, chunks)
print(response)
