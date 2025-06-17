from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os, zipfile, tempfile
from pathlib import Path

from chunks import (
    load_code_files,
    split_documents,
    enrich_chunks_with_embeddings,
    store_to_mongodb
)

app = FastAPI()

# CORS middleware for React frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173","*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def root():
    print("Root route hit!")
    return {"message": "Hello from root"}
@app.post("/collect.zip")
async def upload_zip(file: UploadFile = File(...)):
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip files are supported.")

    try:
        # 1. Read the uploaded zip file into memory
        contents = await file.read()

        # 2. Create a temporary workspace
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = Path(tmpdir) / "uploaded.zip"
            extract_dir = Path(tmpdir) / "extracted"

            # 3. Save and extract the ZIP
            with open(zip_path, "wb") as f:
                f.write(contents)

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

            # 4. Run your processing pipeline
            print("pipeline 1")
            docs = load_code_files(str(extract_dir))
            print("pipeline 2")
            chunks = split_documents(docs)
            print("pipeline 3")
            enriched = enrich_chunks_with_embeddings(chunks)
            print("pipeline 4")
            store_to_mongodb(enriched)

        return {
            "status": "success",
            "chunks_stored": len(enriched),
            "message": f"Zip processed and discarded. Chunks stored in MongoDB."
        }

    except Exception as e:
        import traceback
        print("An error occurred during ZIP processing:")
        # print(e)
        # traceback.print_exc()  # This prints the full traceback
        raise HTTPException(status_code=500, detail=f"Error processing ZIP:  why")
