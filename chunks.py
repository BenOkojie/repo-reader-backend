# chunks.py

from pathlib import Path
from dotenv import load_dotenv
import os
from typing import List
from langchain_community.document_loaders import TextLoader, DirectoryLoader
# from langchain.document_loaders import , TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from gemini import embed_chunk_with_gemini
from pymongo import MongoClient

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("MONGO_DB_NAME", "vector_store")
COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "chunks")


def load_code_files(directory: str) -> List[Document]:
    supported_extensions = [
        ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c", ".cs", ".go", ".rb",
        ".php", ".html", ".css", ".scss", ".json", ".yaml", ".yml", ".md", ".txt",
        ".ipynb", ".sh", ".xml", ".toml", ".ini", ".env", ".rs", ".swift"
    ]

    all_docs = []
    for ext in supported_extensions:
        loader = DirectoryLoader(
            path=directory,
            glob=f"**/*{ext}",
            loader_cls=TextLoader,
            recursive=True,
            show_progress=True
        )
        try:
            all_docs.extend(loader.load())
        except Exception as e:
            print(f"Warning: Failed to load some {ext} files — {e}")

    return all_docs


def split_documents(documents: List[Document], chunk_size=500, chunk_overlap=100) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)


def enrich_chunks_with_embeddings(chunks: List[Document]):
    enriched = []
    for chunk in chunks:
        embedding = embed_chunk_with_gemini(chunk.page_content)
        enriched.append({
            "text": chunk.page_content,
            "embedding": embedding,
            "metadata": chunk.metadata
        })
    return enriched


def store_to_mongodb(records: List[dict]):
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    collection.insert_many(records)
    print(f"✅ Stored {len(records)} records in MongoDB collection '{COLLECTION_NAME}'")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chunk, embed, and store code files.")
    parser.add_argument("--input", type=str, default="extracted/", help="Path to directory with code files")

    args = parser.parse_args()

    print("[1] Loading documents...")
    docs = load_code_files(args.input)

    print("[2] Splitting into chunks...")
    chunks = split_documents(docs)

    print("[3] Generating Gemini embeddings...")
    enriched_chunks = enrich_chunks_with_embeddings(chunks)

    print("[4] Storing in MongoDB...")
    store_to_mongodb(enriched_chunks)

    print("🎉 Done")
