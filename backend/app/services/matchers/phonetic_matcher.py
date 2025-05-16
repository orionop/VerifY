"""
Phonetic matcher service for sound-alike similarity detection.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
import jellyfish
from metaphone import doublemetaphone
from rapidfuzz import fuzz

from app.services.matchers.lexical_matcher import LexicalMatcher
from app.core.config.settings import settings

logger = logging.getLogger(__name__)

class PhoneticMatcher:
    """
    Advanced phonetic matcher for similar-sounding titles regardless of spelling.
    
    This matcher uses multiple phonetic algorithms to find titles that sound similar:
    1. Double Metaphone (primary for English)
    2. Soundex (alternative algorithm)
    3. Refined Soundex (more precise than Soundex)
    4. Combination matching with partial phonetic codes
    """
    
    def __init__(self):
        """Initialize the phonetic matcher with configuration settings."""
        self.similarity_threshold = settings.SIMILARITY_THRESHOLD
        self.max_results = settings.MAX_TITLES_RETURNED
        self.lexical_matcher = LexicalMatcher()  # For normalization
    
    def generate_phonetic_codes(self, title: str) -> Dict[str, Any]:
        """
        Generate multiple phonetic encodings for a title.
        
        Args:
            title: The title to encode
            
        Returns:
            Dictionary of phonetic codes using different algorithms
        """
        # Normalize the title first
        normalized = self.lexical_matcher.normalize_title(title)
        
        # Split into tokens (words)
        tokens = normalized.split()
        
        # Generate phonetic codes for each algorithm
        metaphone_primary = []
        metaphone_secondary = []
        soundex_codes = []
        refined_soundex_codes = []
        
        for token in tokens:
            # Skip very short tokens
            if len(token) < 3:
                continue
                
            # Double Metaphone (returns primary and secondary codes)
            dm_result = doublemetaphone(token)
            if dm_result[0]:  # Primary code
                metaphone_primary.append(dm_result[0])
            if dm_result[1]:  # Secondary code (alternative pronunciation)
                metaphone_secondary.append(dm_result[1])
                
            # Soundex
            soundex = jellyfish.soundex(token)
            if soundex:
                soundex_codes.append(soundex)
                
            # Refined Soundex (more precise)
            refined = jellyfish.nysiis(token)
            if refined:
                refined_soundex_codes.append(refined)
        
        # Combine into single strings for whole-title comparison
        return {
            "metaphone_primary": " ".join(metaphone_primary),
            "metaphone_secondary": " ".join(metaphone_secondary) if metaphone_secondary else None,
            "soundex": " ".join(soundex_codes),
            "refined_soundex": " ".join(refined_soundex_codes),
            "tokens": tokens
        }
    
    def compute_phonetic_similarity(self, 
                                    title_codes: Dict[str, Any], 
                                    existing_codes: Dict[str, Any]) -> float:
        """
        Compute phonetic similarity between two sets of phonetic codes.
        
        Args:
            title_codes: Phonetic codes for the title being checked
            existing_codes: Phonetic codes for an existing title
            
        Returns:
            Similarity score (0-100)
        """
        similarities = []
        
        # Compare primary metaphone codes
        if title_codes["metaphone_primary"] and existing_codes["metaphone_primary"]:
            primary_similarity = fuzz.ratio(
                title_codes["metaphone_primary"], 
                existing_codes["metaphone_primary"]
            )
            similarities.append(primary_similarity * 0.4)  # Weight: 40%
        
        # Compare secondary metaphone codes if available
        if (title_codes["metaphone_secondary"] and existing_codes["metaphone_secondary"]):
            secondary_similarity = fuzz.ratio(
                title_codes["metaphone_secondary"], 
                existing_codes["metaphone_secondary"]
            )
            similarities.append(secondary_similarity * 0.2)  # Weight: 20%
        
        # Compare soundex codes
        if title_codes["soundex"] and existing_codes["soundex"]:
            soundex_similarity = fuzz.ratio(
                title_codes["soundex"], 
                existing_codes["soundex"]
            )
            similarities.append(soundex_similarity * 0.2)  # Weight: 20%
        
        # Compare refined soundex codes
        if title_codes["refined_soundex"] and existing_codes["refined_soundex"]:
            refined_similarity = fuzz.ratio(
                title_codes["refined_soundex"], 
                existing_codes["refined_soundex"]
            )
            similarities.append(refined_similarity * 0.2)  # Weight: 20%
        
        # Handle partial token matches for better detection of added/removed words
        token_level_score = self._compute_token_level_phonetic_similarity(
            title_codes["tokens"], 
            existing_codes["tokens"]
        )
        
        if token_level_score:
            similarities.append(token_level_score)
        
        # Calculate final score - if no valid comparisons, return 0
        if not similarities:
            return 0.0
            
        # Average the similarity scores
        return sum(similarities) / len(similarities)
    
    def _compute_token_level_phonetic_similarity(self, 
                                               title_tokens: List[str], 
                                               existing_tokens: List[str]) -> Optional[float]:
        """
        Compute phonetic similarity at the token (word) level.
        
        This helps detect cases where words are added, removed, or reordered
        but the titles still sound similar.
        
        Args:
            title_tokens: Tokens from the title being checked
            existing_tokens: Tokens from an existing title
            
        Returns:
            Token-level similarity score or None if not applicable
        """
        if not title_tokens or not existing_tokens:
            return None
        
        # Score each pair of tokens and find the best matches
        matches = []
        used_existing = set()
        
        for t_token in title_tokens:
            best_score = 0
            best_idx = -1
            
            # Find the best matching token in existing title
            for idx, e_token in enumerate(existing_tokens):
                if idx in used_existing:
                    continue
                    
                # Compare phonetically
                t_dm = doublemetaphone(t_token)[0]
                e_dm = doublemetaphone(e_token)[0]
                
                if t_dm and e_dm:
                    score = fuzz.ratio(t_dm, e_dm)
                    if score > best_score:
                        best_score = score
                        best_idx = idx
            
            # If we found a good match, add it
            if best_score > 70 and best_idx >= 0:
                matches.append(best_score)
                used_existing.add(best_idx)
        
        # Calculate token coverage and final score
        if not matches:
            return None
            
        # Compute coverage (how many tokens were matched)
        coverage = len(matches) / max(len(title_tokens), len(existing_tokens))
        
        # Combine average match quality with coverage
        avg_match_quality = sum(matches) / len(matches)
        
        # Final token-level score (weighted for importance)
        return (0.7 * avg_match_quality + 0.3 * (coverage * 100)) * 0.4  # Weight: 40%
    
    async def find_phonetic_matches(self, 
                                   title: str, 
                                   existing_titles: List[Dict[str, Any]], 
                                   threshold: float = None) -> List[Dict[str, Any]]:
        """
        Find phonetically similar titles (similar sounding regardless of spelling).
        
        Args:
            title: The title to compare
            existing_titles: List of existing title documents to compare against
            threshold: Minimum similarity score to consider (0-100)
            
        Returns:
            List of similar titles with similarity scores
        """
        if threshold is None:
            threshold = self.similarity_threshold
            
        # Generate phonetic codes for the input title
        title_codes = self.generate_phonetic_codes(title)
        results = []
        
        # Compare with each existing title
        for doc in existing_titles:
            existing_title = doc["title"]
            
            # Generate codes for the existing title (or use cached codes if available)
            existing_codes = doc.get("phonetic_codes")
            if not existing_codes:
                existing_codes = self.generate_phonetic_codes(existing_title)
            
            # Compute similarity
            similarity = self.compute_phonetic_similarity(title_codes, existing_codes)
            
            # Only include results above threshold
            if similarity >= threshold:
                match_details = {
                    "title": existing_title,
                    "similarity": round(similarity, 1),
                    "match_type": "phonetic",
                    "id": doc.get("id"),
                    "language": doc.get("language", "en"),
                    "status": doc.get("status", "active")
                }
                results.append(match_details)
        
        # Sort by similarity score (descending)
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Return top N results
        return results[:self.max_results]

# Create singleton instance
phonetic_matcher = PhoneticMatcher() 