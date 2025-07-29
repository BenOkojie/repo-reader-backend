from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
import json
import numpy as np
import os
from gemini import get_llm_response

router = APIRouter()
# Load model and setup DB
model = SentenceTransformer("all-MiniLM-L6-v2")
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("MONGO_DB_NAME", "vector_store")
COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "chunks")

mongo = MongoClient(MONGO_URI)
collection = mongo[DB_NAME][COLLECTION_NAME]


# Helper: cosine similarity
def cosine_sim(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Search top K from MongoDB
def search_top_chunks(query_embedding, chunks, k=3, threshold=0.3):
    scored_chunks = []
    for chunk in chunks:
        score = cosine_sim(query_embedding, chunk['embedding'])
        if score >= threshold:
            scored_chunks.append((chunk, score))

    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in scored_chunks[:k]]

@router.post("/chat/stream")
async def rag_demo_stream(request: Request):
    data = await request.json()
    query = data.get("message")

    async def event_stream():
        yield json.dumps({"step": "Received query", "data": query}) + "\n"

        # 1. Embed query
        query_emb = model.encode(query).tolist()
        yield json.dumps({"step": "Embedded query", "data": str(query_emb[:5]) + "..."}) + "\n"

        # 2. Fetch and search chunks
        all_chunks = list(collection.find({}, {"_id": 0, "text": 1, "embedding": 1}))
        chunks = search_top_chunks(query_emb, all_chunks, k=3, threshold=0.3)

        if not chunks:
            yield json.dumps({
                "step": "No relevant chunks",
                "data": "Your question doesn't appear related to the uploaded documents."
            }) + "\n"
            return

        preview = [c["text"][:200] for c in chunks]
        yield json.dumps({"step": "Top chunks found", "data": preview}) + "\n"

        # 3. Construct prompt
        context = "\n---\n".join([c["text"] for c in chunks])
        new_prompt = f"Use the context below to answer the question.\n\nContext:\n{context}\n\nQuestion: {query}"
        yield json.dumps({"step": "Constructed new prompt", "data": new_prompt[:300] + "..."}) + "\n"

        # 4. LLM response
        response = get_llm_response(new_prompt)
        yield json.dumps({"step": "LLM response", "data": response}) + "\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

