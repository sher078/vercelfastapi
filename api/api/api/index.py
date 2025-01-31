from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import os
import openai
import pandas as pd
from pydantic import BaseModel

# Initialize FastAPI application
app = FastAPI()

# Directory to store uploaded files
UPLOAD_DIR = "uploaded_documents"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Set OpenAI API key securely via environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

# Data model for handling user queries
class UserQuery(BaseModel):
    file_name: str
    question: str

def extract_text_from_excel(file_path: str) -> str:
    """Reads an Excel file and extracts text content as a single string."""
    df = pd.read_excel(file_path)
    return "\n".join(df.astype(str).values.flatten())

@app.post("/upload/")
async def upload_document(file: UploadFile = File(..., max_length=100_000_000)):  # 100MB max limit
    """Handles file uploads and saves them on the server."""
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    
    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    
    return {"file_name": file.filename, "message": "File uploaded successfully!"}

@app.post("/query/")
async def process_query(user_query: UserQuery):
    """Processes a user's question based on the uploaded document."""
    file_path = os.path.join(UPLOAD_DIR, user_query.file_name)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    # Extract document content
    if user_query.file_name.endswith(".xlsx"):
        document_content = extract_text_from_excel(file_path)
    else:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                document_content = file.read()
        except (OSError, UnicodeDecodeError) as e:
            raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")
    
    # Generate response using OpenAI's model with error handling
    try:
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=f"Answer the following based on this document:\n\n{document_content}\n\nQuestion: {user_query.question}",
            max_tokens=200
        )
        answer = response.choices[0].text.strip()
    except openai.error.OpenAIError as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")
    
    return {"answer": answer}
