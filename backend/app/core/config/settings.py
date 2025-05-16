"""
Configuration settings for the PRGI Title Verification System.
"""

import os
from typing import Dict, List, Optional, Union
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Application settings."""
    
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "PRGI Title Verification System"
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"
    
    # CORS Settings
    BACKEND_CORS_ORIGINS: List[str] = ["*"]
    
    # Security
    API_KEY: Optional[str] = os.getenv("API_KEY")
    SECRET_KEY: str = os.getenv("SECRET_KEY", "prgi_development_secret_key_please_change_in_production")
    
    # Database configuration
    ELASTICSEARCH_HOST: str = os.getenv("ES_HOST", "localhost")
    ELASTICSEARCH_PORT: str = os.getenv("ES_PORT", "9200")
    ELASTICSEARCH_USER: Optional[str] = os.getenv("ES_USER")
    ELASTICSEARCH_PASSWORD: Optional[str] = os.getenv("ES_PASSWORD")
    ELASTICSEARCH_INDEX: str = os.getenv("ES_INDEX", "prgi_titles")
    
    # Redis configuration for caching
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_PASSWORD: Optional[str] = os.getenv("REDIS_PASSWORD")
    
    # Embedding Model Configuration
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    VECTOR_DIMENSION: int = 384  # For all-MiniLM-L6-v2
    
    # Verification Engine Settings
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "65.0"))
    HIGH_SIMILARITY_THRESHOLD: float = float(os.getenv("HIGH_SIMILARITY_THRESHOLD", "80.0"))
    PHONETIC_MATCH_WEIGHT: float = float(os.getenv("PHONETIC_MATCH_WEIGHT", "1.5"))
    TEXT_MATCH_WEIGHT: float = float(os.getenv("TEXT_MATCH_WEIGHT", "1.0"))
    SEMANTIC_MATCH_WEIGHT: float = float(os.getenv("SEMANTIC_MATCH_WEIGHT", "2.0"))
    
    # Performance Settings
    MAX_TITLES_RETURNED: int = int(os.getenv("MAX_TITLES_RETURNED", "5"))
    API_TIMEOUT_SECONDS: int = int(os.getenv("API_TIMEOUT_SECONDS", "10"))
    
    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Disallowed words and patterns
    DISALLOWED_WORDS: List[str] = [
        "police", "crime", "corruption", "cbi", "cid", "army", 
        "supreme court", "high court", "classified", "secret",
        "confidential", "patent"
    ]
    
    COMMON_PREFIXES: List[str] = ["the", "india", "bharat", "national", "hindu", "times", "samachar"]
    COMMON_SUFFIXES: List[str] = ["news", "times", "express", "today", "daily", "weekly", "monthly"]
    
    PERIODICITY_TERMS: List[str] = ["daily", "weekly", "monthly", "quarterly", "annual", "yearly"]
    
    # Language translation support
    TRANSLATION_API_KEY: Optional[str] = os.getenv("TRANSLATION_API_KEY")
    SUPPORTED_LANGUAGES: List[str] = ["en", "hi", "bn", "te", "ta", "mr", "gu", "kn", "ml", "pa", "or", "as"]
    
    # Model config
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

# Create global settings object
settings = Settings() 