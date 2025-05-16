from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import uvicorn

from app.services.verification_service import verify_title_service

app = FastAPI(
    title="Title Verification API",
    description="Government database service for validating proposed titles",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TitleRequest(BaseModel):
    title: str

class SimilarTitle(BaseModel):
    title: str
    similarity: int

class VerificationResponse(BaseModel):
    similarTitles: List[SimilarTitle]
    matchScore: int
    disallowedWords: List[str]
    status: str
    approvalProbability: int

@app.get("/")
def read_root():
    return {"message": "Welcome to the Title Verification API"}

@app.post("/verify-title", response_model=VerificationResponse)
async def verify_title(request: TitleRequest):
    if not request.title or not request.title.strip():
        raise HTTPException(status_code=400, detail="Title cannot be empty")
    
    # Call the service that handles verification logic
    result = await verify_title_service(request.title)
    return result

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True) 