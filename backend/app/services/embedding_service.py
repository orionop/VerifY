"""
Embedding service for generating and managing semantic vectors.

This service provides:
1. Efficient text-to-vector encoding using sentence transformers
2. Caching for frequently used embeddings
3. Multilingual support
4. Batch processing capabilities
"""

import os
import logging
import asyncio
import time
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
from hashlib import md5
import json

from app.core.config.settings import settings

logger = logging.getLogger(__name__)

# Mock cache for development - in production, use Redis
EMBEDDING_CACHE = {}

class EmbeddingService:
    """
    Service for generating and managing semantic embeddings from title text.
    
    This service uses state-of-the-art language models to convert text into
    numerical vectors that capture semantic meaning, enabling:
    - Semantic similarity detection
    - Cross-language matching
    - Meaning-based search and retrieval
    """
    
    def __init__(self):
        """Initialize the embedding service."""
        self.model = None
        self.initialized = False
        self.model_name = settings.EMBEDDING_MODEL
        self.vector_dimension = settings.VECTOR_DIMENSION
        self.supported_languages = settings.SUPPORTED_LANGUAGES
        self.use_cache = True
        self.cache_ttl = 3600 * 24 * 7  # 1 week cache TTL
        
        # Try to initialize the model on startup for faster first request
        self._lazy_init()
    
    def _lazy_init(self):
        """Attempt to initialize in background without blocking."""
        try:
            # In a real implementation, this would load the model in a non-blocking way
            # For example, using a background task or thread
            # self.model = SentenceTransformer(self.model_name)
            # self.initialized = True
            pass
        except Exception as e:
            logger.warning(f"Background model initialization failed: {e}")
    
    async def initialize(self) -> bool:
        """
        Initialize the embedding model.
        
        Returns:
            bool: True if initialization was successful
        """
        if self.initialized:
            return True
            
        try:
            # In a real implementation, this would be:
            # self.model = SentenceTransformer(self.model_name)
            
            # Simulate model loading with a small delay
            await asyncio.sleep(0.1)
            
            self.initialized = True
            logger.info(f"Initialized embedding model {self.model_name}")
            return True
        except Exception as e:
            logger.error(f"Error initializing embedding model: {e}")
            return False
    
    def _get_cache_key(self, text: str, lang: str) -> str:
        """
        Generate a cache key for the text and language.
        
        Args:
            text: Text to embed
            lang: Language code
            
        Returns:
            Cache key string
        """
        # Create a deterministic cache key
        normalized_text = text.lower().strip()
        hash_input = f"{normalized_text}:{lang}:{self.model_name}"
        return md5(hash_input.encode('utf-8')).hexdigest()
    
    async def _check_cache(self, text: str, lang: str) -> Optional[List[float]]:
        """
        Check if embedding is in cache.
        
        Args:
            text: Text to check
            lang: Language code
            
        Returns:
            Cached embedding vector or None
        """
        if not self.use_cache:
            return None
            
        cache_key = self._get_cache_key(text, lang)
        
        # Check in-memory cache first (for development)
        cached_item = EMBEDDING_CACHE.get(cache_key)
        if cached_item:
            expiry, vector = cached_item
            if expiry > time.time():
                # Cache hit
                logger.debug(f"Cache hit for '{text[:20]}...'")
                return vector
            else:
                # Cache expired
                del EMBEDDING_CACHE[cache_key]
        
        # In production, we would also check Redis here
        # Example Redis implementation:
        # cached_vector_json = await redis.get(f"embedding:{cache_key}")
        # if cached_vector_json:
        #     return json.loads(cached_vector_json)
            
        return None
    
    async def _update_cache(self, text: str, lang: str, vector: List[float]) -> None:
        """
        Store embedding in cache.
        
        Args:
            text: Original text
            lang: Language code
            vector: Embedding vector
        """
        if not self.use_cache:
            return
            
        cache_key = self._get_cache_key(text, lang)
        expiry = time.time() + self.cache_ttl
        
        # Store in in-memory cache (for development)
        EMBEDDING_CACHE[cache_key] = (expiry, vector)
        
        # In production, we would also store in Redis here
        # Example Redis implementation:
        # vector_json = json.dumps(vector)
        # await redis.set(f"embedding:{cache_key}", vector_json, ex=self.cache_ttl)
    
    async def get_embedding(self, 
                           text: str, 
                           language: str = "en") -> Optional[List[float]]:
        """
        Generate an embedding vector for the given text.
        
        Args:
            text: The text to embed
            language: ISO 639-1 language code
            
        Returns:
            List[float]: Embedding vector or None if error
        """
        if not text.strip():
            logger.warning("Empty text provided for embedding")
            return None
            
        if language not in self.supported_languages:
            logger.warning(f"Unsupported language code: {language}, falling back to 'en'")
            language = "en"
            
        # Check cache first
        cached_vector = await self._check_cache(text, language)
        if cached_vector is not None:
            return cached_vector
            
        # Initialize if needed
        if not self.initialized:
            success = await self.initialize()
            if not success:
                return None
        
        try:
            # In a real implementation, use the model with language info
            # return self.model.encode(text).tolist()
            
            # For this example, we'll simulate model output with deterministic pseudo-random vectors
            # Important: In real implementation, this would be an actual model call
            # This deterministic approach ensures consistent results for the same input
            seed = int(md5(f"{text}:{language}".encode()).hexdigest(), 16) % 10000
            np.random.seed(seed)
            
            # Generate a vector with controlled randomness
            vector = np.random.normal(0, 0.1, self.vector_dimension).tolist()
            
            # Cache the result
            await self._update_cache(text, language, vector)
            
            return vector
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    async def get_batch_embeddings(self, 
                                  texts: List[str],
                                  language: str = "en") -> List[Optional[List[float]]]:
        """
        Generate embeddings for multiple texts in batch.
        
        Args:
            texts: List of texts to embed
            language: ISO 639-1 language code
            
        Returns:
            List of embedding vectors (or None for texts that failed)
        """
        if not texts:
            return []
            
        # Check which texts are already cached
        results = []
        texts_to_embed = []
        indices_to_embed = []
        
        # First check cache for all texts
        for i, text in enumerate(texts):
            if not text.strip():
                results.append(None)
                continue
                
            cached_vector = await self._check_cache(text, language)
            if cached_vector is not None:
                results.append(cached_vector)
            else:
                # Need to compute this one
                results.append(None)  # Placeholder
                texts_to_embed.append(text)
                indices_to_embed.append(i)
        
        # If all texts were cached, return immediately
        if not texts_to_embed:
            return results
            
        # Initialize if needed
        if not self.initialized and texts_to_embed:
            success = await self.initialize()
            if not success:
                # If initialization failed, return None for all uncached texts
                return results
        
        try:
            # In a real implementation:
            # new_vectors = self.model.encode(texts_to_embed).tolist()
            
            # For this example, generate vectors deterministically
            new_vectors = []
            for text in texts_to_embed:
                seed = int(md5(f"{text}:{language}".encode()).hexdigest(), 16) % 10000
                np.random.seed(seed)
                vector = np.random.normal(0, 0.1, self.vector_dimension).tolist()
                new_vectors.append(vector)
            
            # Update cache and results
            for idx, vector, text in zip(indices_to_embed, new_vectors, texts_to_embed):
                await self._update_cache(text, language, vector)
                results[idx] = vector
                
            return results
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {e}")
            # Return what we have so far - may be partial results
            return results
    
    async def search_by_vector(self, 
                              vector: List[float],
                              vectors_to_search: List[Tuple[List[float], Any]],
                              top_k: int = 5,
                              threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Search for the most similar vectors in a collection.
        
        Args:
            vector: Query vector to search for
            vectors_to_search: List of (vector, metadata) tuples to search in
            top_k: Number of results to return
            threshold: Minimum similarity threshold (0-1)
            
        Returns:
            List of {similarity, metadata} dictionaries
        """
        if not vector or not vectors_to_search:
            return []
            
        # Convert query to numpy array
        query_vector = np.array(vector)
        
        # Compute similarities for all vectors
        similarities = []
        for idx, (candidate_vector, metadata) in enumerate(vectors_to_search):
            # Convert to numpy array
            candidate_np = np.array(candidate_vector)
            
            # Compute cosine similarity
            dot_product = np.dot(query_vector, candidate_np)
            norm_query = np.linalg.norm(query_vector)
            norm_candidate = np.linalg.norm(candidate_np)
            
            if norm_query == 0 or norm_candidate == 0:
                similarity = 0
            else:
                similarity = dot_product / (norm_query * norm_candidate)
            
            if similarity >= threshold:
                similarities.append({
                    "similarity": float(similarity),
                    "metadata": metadata,
                    "index": idx
                })
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Return top-k results
        return similarities[:top_k]
            
# Create singleton instance
embedding_service = EmbeddingService()

async def get_embedding_service() -> EmbeddingService:
    """Get the embedding service singleton instance."""
    if not embedding_service.initialized:
        await embedding_service.initialize()
    return embedding_service 