from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from zipfile import ZipFile, is_zipfile
from io import BytesIO

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000","http://localhost:5173"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.post("/collect.zip")
async def process_zip(file: UploadFile = File(...)):
    # Read the file bytes into memory
    contents = await file.read()

    # Verify it's a valid zip
    if not is_zipfile(BytesIO(contents)):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid .zip archive")

    # Open the zip file in memory
    with ZipFile(BytesIO(contents)) as zip_file:
        file_list = zip_file.namelist()  # list of files inside the zip

        # Optionally, you can read or extract files like:
        # with zip_file.open(file_list[0]) as f:
        #     content = f.read()

    return JSONResponse(content={"filename": file.filename, "files_in_zip": file_list})
