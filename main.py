from fastapi import FastAPI
from fastapi import status
app = FastAPI(title="RAG Chatbot Backend API", version="0.1.0")

@app.get("/")
async def read_root():
    return {"message": "Welcome to the RAG Chatbot Backend!"}

@app.get("/health")
async def health_check():
    return status.HTTP_200_OK

