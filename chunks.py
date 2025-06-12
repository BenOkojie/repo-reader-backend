# chunks.py

from pathlib import Path
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings  # or HuggingFaceEmbeddings
from langchain.vectorstores import FAISS  # or Chroma, etc.
from langchain.schema import Document
from langchain.chat_models import ChatGoogleGenerativeAI

gemini = ChatGoogleGenerativeAI(model="gemini-pro")

def enrich_metadata(chunks: list[Document]):
    enriched = []

    for doc in chunks:
        prompt = (
            f"Summarize the purpose of this code and name any key functions or classes:\n\n{doc.page_content}"
        )
        try:
            response = gemini.invoke(prompt)
            doc.metadata["summary"] = response.content
        except Exception as e:
            doc.metadata["summary"] = "Unknown"

        enriched.append(doc)

    return enriched


def load_code_files(directory: str, pattern: str = "**/*.py"):
    loader = DirectoryLoader(
        path=directory,
        glob=pattern,
        loader_cls=TextLoader,
        show_progress=True,
        recursive=True
    )
    return loader.load()


def split_documents(documents, chunk_size=500, chunk_overlap=100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)


def embed_chunks(chunks):
    embedding_model = OpenAIEmbeddings()  # replace if using a different provider
    return embedding_model.embed_documents([chunk.page_content for chunk in chunks])


def store_embeddings(chunks, embeddings, index_path="faiss_index"):
    vectorstore = FAISS.from_embeddings(
        embeddings,
        documents=chunks
    )
    vectorstore.save_local(index_path)
    return vectorstore


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chunk and embed code files.")
    parser.add_argument("--input", type=str, default="extracted/", help="Path to directory with code files")
    parser.add_argument("--index", type=str, default="faiss_index", help="Path to store FAISS index")

    args = parser.parse_args()

    print("[1] Loading documents...")
    docs = load_code_files(args.input)

    print("[2] Splitting into chunks...")
    chunks = split_documents(docs)

    print(f"[3] Generating embeddings for {len(chunks)} chunks...")
    embeddings = embed_chunks(chunks)

    print("[4] Storing in FAISS...")
    store_embeddings(chunks, embeddings, args.index)

    print("✅ Done: Embeddings stored in", args.index)
