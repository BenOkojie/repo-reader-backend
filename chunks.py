# chunks.py

from pathlib import Path
from dotenv import load_dotenv
import os
from typing import List
from langchain_community.document_loaders import TextLoader, DirectoryLoader, PyPDFLoader
# from langchain.document_loaders import , TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
from gemini import embed_chunk_with_gemini
from pymongo import MongoClient

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("MONGO_DB_NAME", "vector_store")
COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "chunks")


def load_code_files(directory: str) -> List[Document]:
    supported_extensions = [
        ".pdf", ".js", ".ts", ".jsx", ".tsx", ".java", ".cpp", ".c", ".cs", ".go", ".rb",
        ".php", ".html", ".css", ".scss", ".json", ".yaml", ".yml", ".md", ".txt",
        ".ipynb", ".sh", ".xml", ".toml", ".ini", ".env", ".rs", ".swift"
    ]

    all_docs = []
    path_obj = Path(directory)
    if path_obj.is_file():
        ext = path_obj.suffix
        if ext in supported_extensions:
            print(f"Loading single file: {path_obj.name}")
            try:
                if ext == ".pdf":
                    loader = PyPDFLoader(str(path_obj))
                else:
                    loader = TextLoader(str(path_obj))
                all_docs.extend(loader.load())
            except Exception as e:
                print(f"Warning: Failed to load file {path_obj.name} — {e}")
        else:
            print(f"Skipped unsupported file: {path_obj.name}")
    else:
        for ext in supported_extensions:
            print(f"Looking for *{ext} files...")
            if ext == ".pdf":
                loader_cls = PyPDFLoader
            else:
                loader_cls = TextLoader

            loader = DirectoryLoader(
                path=directory,
                glob=f"**/*{ext}",
                loader_cls=loader_cls,
                recursive=True,
                show_progress=True
            )
            try:
                all_docs.extend(loader.load())
            except Exception as e:
                print(f"Warning: Failed to load some {ext} files — {e}")
    print(all_docs)
    return all_docs


def split_documents(documents: List[Document], chunk_size=400, chunk_overlap=80) -> List[Document]:
    enriched_docs = []

    for doc in documents:
        # Normalize line breaks — PDF cleanup
        doc.page_content = doc.page_content.replace('\n', ' ').strip()

        # Add file-level metadata if not already present
        doc.metadata.setdefault("source_type", "pdf" if doc.metadata.get("source", "").endswith(".pdf") else "text")
        doc.metadata.setdefault("lang", "en")
        doc.metadata.setdefault("title", doc.metadata.get("source", "unknown"))

        enriched_docs.append(doc)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    split_chunks = splitter.split_documents(enriched_docs)

    # Add chunk-level metadata
    for i, chunk in enumerate(split_chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["token_count"] = len(chunk.page_content.split())

        # Preserve source page if present (e.g., PDFs)
        if "page" in chunk.metadata:
            chunk.metadata["source_page"] = f"{chunk.metadata.get('source', 'unknown')}#page={chunk.metadata['page']}"

    return split_chunks


def enrich_chunks_with_embeddings(chunks: List[Document]):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"Generating embeddings for {len(chunks)} chunks...")

    texts = [chunk.page_content for chunk in chunks]
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)

    enriched = []
    for chunk, embedding in zip(chunks, embeddings):
        enriched.append({
            "text": chunk.page_content,
            "embedding": embedding.tolist(),  # Needed for MongoDB storage
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
