"""
Semantic matcher service for meaning-based similarity detection.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from time import time

from app.core.config.settings import settings
from app.services.embedding_service import embedding_service

logger = logging.getLogger(__name__)

class SemanticMatcher:
    """
    Advanced semantic matcher for meaning-based title similarity using embeddings.
    
    This matcher uses neural language models to detect titles with similar meanings
    regardless of the exact words used. It's especially useful for:
    1. Cross-language similarity detection (when embeddings are multilingual)
    2. Detecting paraphrased titles that use different terms for the same concept
    3. Understanding semantic similarities beyond lexical or phonetic matching
    """
    
    def __init__(self):
        """Initialize the semantic matcher with configuration settings."""
        self.similarity_threshold = settings.SIMILARITY_THRESHOLD
        self.max_results = settings.MAX_TITLES_RETURNED
        self.vector_dimension = settings.VECTOR_DIMENSION
    
    def cosine_similarity(self, 
                          vector1: List[float], 
                          vector2: List[float]) -> float:
        """
        Compute cosine similarity between two embedding vectors.
        
        Args:
            vector1: First embedding vector
            vector2: Second embedding vector
            
        Returns:
            Cosine similarity score (0-100)
        """
        if not vector1 or not vector2:
            return 0.0
            
        try:
            # Convert to numpy arrays
            v1 = np.array(vector1)
            v2 = np.array(vector2)
            
            # Compute cosine similarity
            dot_product = np.dot(v1, v2)
            norm_v1 = np.linalg.norm(v1)
            norm_v2 = np.linalg.norm(v2)
            
            if norm_v1 == 0 or norm_v2 == 0:
                return 0.0
                
            cosine_sim = dot_product / (norm_v1 * norm_v2)
            
            # Convert to percentage (0-100)
            return max(0, min(100, float(cosine_sim * 100)))
        except Exception as e:
            logger.error(f"Error computing cosine similarity: {e}")
            return 0.0
    
    async def get_embedding(self, 
                           title: str, 
                           language: str = "en") -> Optional[List[float]]:
        """
        Get embedding vector for a title.
        
        Args:
            title: The title to embed
            language: Language code of the title
            
        Returns:
            Embedding vector or None if error
        """
        start_time = time()
        
        # Get embedding from service
        vector = await embedding_service.get_embedding(title, language)
        
        end_time = time()
        logger.debug(f"Generated embedding for '{title}' in {(end_time - start_time) * 1000:.2f}ms")
        
        return vector
    
    async def find_semantic_matches(self, 
                                   title: str, 
                                   existing_titles: List[Dict[str, Any]], 
                                   language: str = "en",
                                   threshold: float = None) -> List[Dict[str, Any]]:
        """
        Find semantically similar titles (similar meaning regardless of exact wording).
        
        Args:
            title: The title to compare
            existing_titles: List of existing title documents to compare against
            language: Language code of the input title
            threshold: Minimum similarity score to consider (0-100)
            
        Returns:
            List of similar titles with similarity scores
        """
        if threshold is None:
            threshold = self.similarity_threshold
            
        # Get embedding for the input title
        title_vector = await self.get_embedding(title, language)
        if not title_vector:
            logger.error(f"Failed to generate embedding for title: {title}")
            return []
            
        results = []
        
        # Compare with each existing title
        for doc in existing_titles:
            existing_title = doc["title"]
            existing_vector = doc.get("title_vector")
            existing_lang = doc.get("language", "en")
            
            # Skip if no embedding available
            if not existing_vector:
                continue
                
            # For cross-language matching, adjust scores based on language mismatch
            lang_adjustment = 1.0
            if language != existing_lang:
                # Apply slight penalty for cross-language matches
                # This can be adjusted based on embedding model's multilingual capabilities
                lang_adjustment = 0.9
            
            # Compute similarity
            similarity = self.cosine_similarity(title_vector, existing_vector)
            
            # Apply language adjustment
            adjusted_similarity = similarity * lang_adjustment
            
            # Only include results above threshold
            if adjusted_similarity >= threshold:
                match_details = {
                    "title": existing_title,
                    "similarity": round(adjusted_similarity, 1),
                    "match_type": "semantic",
                    "id": doc.get("id"),
                    "language": existing_lang,
                    "status": doc.get("status", "active"),
                    "is_cross_language": language != existing_lang
                }
                results.append(match_details)
        
        # Sort by similarity score (descending)
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Return top N results
        return results[:self.max_results]
    
    async def detect_cross_language_duplicates(self,
                                              title: str,
                                              language: str,
                                              existing_titles: List[Dict[str, Any]],
                                              threshold: float = 85.0) -> List[Dict[str, Any]]:
        """
        Specifically detect titles with similar meanings in different languages.
        
        This is a specialized version of semantic matching that focuses on
        cross-language detection with a higher threshold.
        
        Args:
            title: The title to check
            language: Language code of the input title
            existing_titles: List of existing title documents
            threshold: Minimum similarity score for cross-language matches (0-100)
            
        Returns:
            List of cross-language matches with similarity scores
        """
        # Filter existing titles to only include those in different languages
        different_language_titles = [
            doc for doc in existing_titles
            if doc.get("language") and doc.get("language") != language
        ]
        
        # If no cross-language titles, return empty list
        if not different_language_titles:
            return []
            
        # Use semantic matching with higher threshold
        cross_lang_matches = await self.find_semantic_matches(
            title, 
            different_language_titles,
            language, 
            threshold
        )
        
        # Enhance match details for cross-language
        for match in cross_lang_matches:
            match["match_type"] = "cross_language"
        
        return cross_lang_matches

# Create singleton instance
semantic_matcher = SemanticMatcher() 