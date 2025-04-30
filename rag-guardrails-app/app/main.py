from fastapi import FastAPI
from pydantic import BaseModel
from app.services.rag import rag_pipeline
from langfuse import Langfuse
from langfuse.decorators import observe
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Initialize Langfuse
langfuse = Langfuse(
  secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
  public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
  host=os.environ.get("LANGFUSE_HOST")
)

class QueryRequest(BaseModel):
    question: str


@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/query")
@observe()
async def query_rag(request: QueryRequest):
    raw_answer = await rag_pipeline(request.question)
    
    return {
        "question": raw_answer,
    }