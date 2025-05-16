"""
Data models for title verification.
"""

from typing import List, Dict, Optional, Any, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime
import re

class TitleBase(BaseModel):
    """Base model for title data."""
    title: str = Field(..., min_length=3, max_length=150)
    
    @validator('title')
    def title_must_be_valid(cls, v):
        """Validates that the title contains valid characters."""
        if not v.strip():
            raise ValueError("Title cannot be empty or just whitespace")
        
        # Check for invalid characters
        if re.search(r'[^\w\s\-.,&:;\'\"!?()[\]{}]', v, re.UNICODE):
            raise ValueError("Title contains invalid characters")
        
        return v.strip()

class TitleRequest(TitleBase):
    """Request model for title verification."""
    language: Optional[str] = Field("en", description="ISO 639-1 language code of the title")
    
    @validator('language')
    def validate_language(cls, v):
        """Validates that the language code is supported."""
        supported = ["en", "hi", "bn", "te", "ta", "mr", "gu", "kn", "ml", "pa", "or", "as"]
        if v not in supported:
            raise ValueError(f"Unsupported language code. Supported codes: {', '.join(supported)}")
        return v

class SimilarTitle(BaseModel):
    """Model for similar title results."""
    title: str
    similarity: float
    match_type: str = Field(..., description="Type of match: lexical, phonetic, semantic, or combined")
    language: Optional[str] = "en"
    status: Optional[str] = "active"
    id: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "title": "Daily News",
                "similarity": 85.5,
                "match_type": "lexical",
                "language": "en",
                "status": "active",
                "id": "title_12345"
            }
        }

class DisallowedWordMatch(BaseModel):
    """Model for disallowed word matches."""
    word: str
    context: str
    rule_type: str = Field(..., description="Type of rule: disallowed_word, prefix, suffix, or periodicity")
    
    class Config:
        schema_extra = {
            "example": {
                "word": "police",
                "context": "The word 'police' is not allowed in titles",
                "rule_type": "disallowed_word"
            }
        }

class CombinationMatch(BaseModel):
    """Model for detecting combined existing titles."""
    first_title: str
    second_title: str
    context: str
    
    class Config:
        schema_extra = {
            "example": {
                "first_title": "Hindu",
                "second_title": "Times",
                "context": "New title appears to combine two existing titles"
            }
        }

class VerificationResponse(BaseModel):
    """Response model for title verification."""
    title: str
    similar_titles: List[SimilarTitle] = []
    disallowed_words: List[DisallowedWordMatch] = []
    combined_titles: List[CombinationMatch] = []
    match_score: float = Field(..., description="Overall match score (0-100)")
    status: str = Field(..., description="Accepted or Rejected")
    approval_probability: float = Field(..., description="Probability of approval (0-100)")
    feedback: str = Field(..., description="Human-readable feedback message")
    request_id: str
    verification_time_ms: int
    
    class Config:
        schema_extra = {
            "example": {
                "title": "Daily News Express",
                "similar_titles": [
                    {
                        "title": "Daily News", 
                        "similarity": 85.5,
                        "match_type": "lexical",
                        "language": "en",
                        "status": "active",
                        "id": "title_12345"
                    }
                ],
                "disallowed_words": [],
                "combined_titles": [],
                "match_score": 85.5,
                "status": "Rejected",
                "approval_probability": 14.5,
                "feedback": "The title is too similar to existing title 'Daily News'",
                "request_id": "req_abc123",
                "verification_time_ms": 150
            }
        }

class TitleDocument(TitleBase):
    """Database model for a title document."""
    id: Optional[str] = None
    title_vector: Optional[List[float]] = None
    phonetic_code: Optional[str] = None
    normalized_form: Optional[str] = None
    language: str = "en"
    status: str = "active"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = {}

    class Config:
        schema_extra = {
            "example": {
                "id": "title_12345",
                "title": "Daily News",
                "title_vector": [0.1, 0.2, 0.3],  # Truncated for brevity
                "phonetic_code": "DL NS",
                "normalized_form": "daily news",
                "language": "en",
                "status": "active",
                "created_at": "2023-10-16T12:34:56.789Z",
                "metadata": {
                    "publisher": "Example Publishing House",
                    "application_id": "APP123456"
                }
            }
        } 