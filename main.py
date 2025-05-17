from fastapi import FastAPI
from fastapi import status
app = FastAPI(title="RAG Chatbot Backend API", version="0.1.0")

@app.get("/")
async def read_root():
    return {"message": "Welcome to the RAG Chatbot Backend!"}

@app.get("/health")
async def health_check():
    return status.HTTP_200_OK

# To run this (from your terminal, in the project root where main.py is):
# uvicorn main:app --reload
# Then open http://127.0.0.1:8000 in your browser
# And http://127.0.0.1:8000/docs for automatic API documentation