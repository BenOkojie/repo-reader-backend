from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os, zipfile, tempfile
from pathlib import Path
from chat import router as chat_router
from chunks import (
    load_code_files,
    split_documents,
    enrich_chunks_with_embeddings,
    store_to_mongodb
)

app = FastAPI()

# CORS for local React dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(chat_router)
@app.get("/")
def root():
    return {"message": "Hello from FastAPI"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir)
            extracted_dir = temp_path / "extracted"
            extracted_dir.mkdir(exist_ok=True)

            if file.filename.endswith(".zip"):
                # Save and extract zip file
                zip_path = temp_path / "uploaded.zip"
                with open(zip_path, "wb") as f:
                    f.write(contents)

                with zipfile.ZipFile(zip_path, "r") as zip_ref:
                    zip_ref.extractall(extracted_dir)

                target_path = extracted_dir  # Process extracted files

            else:
                # Save single file
                file_path = extracted_dir / file.filename
                with open(file_path, "wb") as f:
                    f.write(contents)

                target_path = file_path  # Process the single file

            # Run pipeline
            print("Loading files...")
            docs = load_code_files(str(target_path))

            print("Splitting...")
            chunks = split_documents(docs)

            print("Embedding...")
            enriched = enrich_chunks_with_embeddings(chunks)

            print("Storing to MongoDB...")
            store_to_mongodb(enriched)

        return {
            "status": "success",
            "file_type": "zip" if file.filename.endswith(".zip") else "single file",
            "chunks_stored": len(enriched),
            "message": f"{file.filename} processed and stored."
        }

    except Exception as e:
        print("An error occurred during file processing:")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
