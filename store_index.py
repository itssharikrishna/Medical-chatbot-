import os
from dotenv import load_dotenv
from pathlib import Path
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from langchain_ollama import OllamaLLM

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# -----------------------------
# 1️⃣ Load PDF files
# -----------------------------
def load_pdf_file(folder_path):
    docs = []
    for pdf_file in Path(folder_path).glob("*.pdf"):
        reader = PdfReader(str(pdf_file))
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        docs.append({"text": text, "source": str(pdf_file)})
    return docs

extracted_data = load_pdf_file("data/")  # folder containing PDFs

# -----------------------------
# 2️⃣ Split text into chunks
# -----------------------------
def split_text_into_chunks(docs, chunk_size=500, chunk_overlap=50):
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

text_chunks = split_text_into_chunks(extracted_data)
texts = [chunk["text"] for chunk in text_chunks]

# -----------------------------
# 3️⃣ Embeddings
# -----------------------------
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# -----------------------------
# 4️⃣ Initialize Pinecone (new SDK)
# -----------------------------
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"

# Create the index if it doesn't exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(index_name)

# -----------------------------
# 5️⃣ Insert documents into Pinecone in batches
# -----------------------------
def insert_docs_to_pinecone(chunks, embedding_model, batch_size=50):
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        vectors_to_upsert = []
        for j, chunk in enumerate(batch):
            vector = embedding_model.encode(chunk["text"]).tolist()
            vectors_to_upsert.append({
                "id": str(i + j),
                "values": vector,
                "metadata": {"text": chunk["text"][:500], "source": chunk["source"]}  # truncate if needed
            })
        index.upsert(vectors=vectors_to_upsert)
    print("Pinecone index ready with all documents!")

insert_docs_to_pinecone(text_chunks, embedding_model)

# -----------------------------
# 6️⃣ Inspect some vectors
# -----------------------------
print("\nInspecting some vectors from Pinecone...")
dummy_vector = [0.0] * 384
response = index.query(vector=dummy_vector, top_k=5, include_metadata=True)
for match in response['matches']:
    print("Vector ID:", match['id'])
    print("Metadata:", match['metadata'])
    print("-" * 50)

# -----------------------------
# 7️⃣ Initialize Ollama LLM
# -----------------------------
llm = OllamaLLM(model="llama3")  # Ensure Ollama server is running locally

# -----------------------------
# 8️⃣ RAG Query Function
# -----------------------------
def answer_question(query, top_k=5):
    # Embed query
    query_vector = embedding_model.encode([query], convert_to_numpy=True)[0].tolist()

    # Query Pinecone
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)

    # Combine retrieved text
    context = "\n".join([match['metadata']['text'] for match in results['matches']])

    # Build prompt
    prompt = (
        "You are a helpful medical assistant.\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{query}"
    )

    # Generate answer
    result = llm.generate([prompt])
    response = result.generations[0][0].text
    return response

# -----------------------------
# 9️⃣ Example Query
# -----------------------------
if __name__ == "__main__":
    query = "What is Acne?"
    answer = answer_question(query)
    print("\nAnswer:", answer)
