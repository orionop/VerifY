"""
Lexical matcher service for string-based similarity detection.
"""

import re
import logging
from typing import List, Dict, Any, Tuple, Set
from rapidfuzz import fuzz, process
from unidecode import unidecode

from app.core.config.settings import settings

logger = logging.getLogger(__name__)

class LexicalMatcher:
    """
    Advanced lexical matcher for title similarity using multiple string similarity algorithms.
    
    This matcher handles:
    1. Exact string matching
    2. Fuzzy string matching with Levenshtein distance and other algorithms
    3. Token-based similarity (words in common, word order)
    4. Normalization to handle case, accents, and special characters
    5. Prefix/suffix matching with special handling for common terms
    """
    
    def __init__(self):
        """Initialize the lexical matcher with configuration settings."""
        self.similarity_threshold = settings.SIMILARITY_THRESHOLD
        self.common_prefixes = set(settings.COMMON_PREFIXES)
        self.common_suffixes = set(settings.COMMON_SUFFIXES)
        self.periodicity_terms = set(settings.PERIODICITY_TERMS)
        self.max_results = settings.MAX_TITLES_RETURNED
    
    @staticmethod
    def normalize_title(title: str) -> str:
        """
        Normalize a title for better comparison:
        - Convert to lowercase
        - Remove accents/diacritics
        - Remove punctuation
        - Normalize whitespace
        """
        # Convert to lowercase and remove accents
        normalized = unidecode(title.lower())
        
        # Remove punctuation except spaces
        normalized = re.sub(r'[^\w\s]', '', normalized)
        
        # Normalize whitespace
        normalized = ' '.join(normalized.split())
        
        return normalized
    
    @staticmethod
    def extract_tokens(title: str) -> List[str]:
        """Extract word tokens from a title."""
        # Normalize and split by whitespace
        return LexicalMatcher.normalize_title(title).split()
    
    def detect_prefixes_and_suffixes(self, 
                                     title: str, 
                                     existing_title: str) -> Tuple[Set[str], Set[str]]:
        """
        Detect common prefixes and suffixes between two titles.
        
        Args:
            title: The title being checked
            existing_title: The existing title to compare with
            
        Returns:
            Tuple containing sets of matching prefixes and suffixes
        """
        title_tokens = self.extract_tokens(title)
        existing_tokens = self.extract_tokens(existing_title)
        
        # Skip if either title has no tokens
        if not title_tokens or not existing_tokens:
            return set(), set()
        
        # Check for common prefixes
        matching_prefixes = set()
        if title_tokens[0].lower() in self.common_prefixes and \
           title_tokens[0].lower() == existing_tokens[0].lower():
            matching_prefixes.add(title_tokens[0].lower())
            
        # Check for common suffixes
        matching_suffixes = set()
        if title_tokens[-1].lower() in self.common_suffixes and \
           title_tokens[-1].lower() == existing_tokens[-1].lower():
            matching_suffixes.add(title_tokens[-1].lower())
            
        return matching_prefixes, matching_suffixes
    
    def detect_periodicity_modification(self, title: str, existing_title: str) -> bool:
        """
        Detect if a title is created by adding a periodicity term to an existing title.
        
        Args:
            title: The title being checked
            existing_title: The existing title to compare with
            
        Returns:
            True if the title appears to be a periodicity modification
        """
        title_tokens = self.extract_tokens(title)
        existing_tokens = self.extract_tokens(existing_title)
        
        # Check if the new title is longer
        if len(title_tokens) <= len(existing_tokens):
            return False
        
        # Look for periodicity terms in the new title
        periodicity_found = False
        for term in self.periodicity_terms:
            if term in title_tokens and term not in existing_tokens:
                periodicity_found = True
                break
                
        if not periodicity_found:
            return False
            
        # Check if the existing title is contained within the new title
        # after removing periodicity terms
        filtered_tokens = [t for t in title_tokens if t not in self.periodicity_terms]
        existing_title_str = ' '.join(existing_tokens)
        filtered_title_str = ' '.join(filtered_tokens)
        
        return existing_title_str in filtered_title_str or \
               fuzz.partial_ratio(existing_title_str, filtered_title_str) > 90
    
    async def find_similar(self, 
                           title: str, 
                           existing_titles: List[Dict[str, Any]], 
                           threshold: float = None) -> List[Dict[str, Any]]:
        """
        Find lexically similar titles using multiple fuzzy matching algorithms.
        
        Args:
            title: The title to compare
            existing_titles: List of existing title documents to compare against
            threshold: Minimum similarity score to consider (0-100)
            
        Returns:
            List of similar titles with similarity scores and match details
        """
        if threshold is None:
            threshold = self.similarity_threshold
            
        normalized_title = self.normalize_title(title)
        results = []
        
        for doc in existing_titles:
            existing_title = doc["title"]
            normalized_existing = self.normalize_title(existing_title)
            
            # Skip exact duplicates - they'll be caught by exact match check
            if normalized_title == normalized_existing:
                continue
                
            # Calculate multiple similarity metrics
            ratio = fuzz.ratio(normalized_title, normalized_existing)
            partial_ratio = fuzz.partial_ratio(normalized_title, normalized_existing)
            token_sort_ratio = fuzz.token_sort_ratio(normalized_title, normalized_existing)
            token_set_ratio = fuzz.token_set_ratio(normalized_title, normalized_existing)
            
            # Use a weighted combination of metrics - customized for title comparison
            # The weighting can be adjusted based on real-world effectiveness
            combined_score = (
                0.15 * ratio +           # Exact character matching
                0.25 * partial_ratio +   # Substring matching
                0.35 * token_sort_ratio + # Word order insensitive
                0.25 * token_set_ratio    # Common words regardless of order
            )
            
            # Check for prefix/suffix matches and periodicity
            matching_prefixes, matching_suffixes = self.detect_prefixes_and_suffixes(
                title, existing_title
            )
            
            is_periodicity_modification = self.detect_periodicity_modification(
                title, existing_title
            )
            
            # Adjust score based on special cases
            if matching_prefixes or matching_suffixes:
                # Increase score for matching prefixes/suffixes
                combined_score = min(100, combined_score + 5)
                
            if is_periodicity_modification:
                # Significant boost for periodicity modifications
                combined_score = min(100, combined_score + 20)
            
            # Only include results above threshold
            if combined_score >= threshold:
                match_details = {
                    "title": existing_title,
                    "similarity": round(combined_score, 1),
                    "match_type": "lexical",
                    "id": doc.get("id"),
                    "language": doc.get("language", "en"),
                    "status": doc.get("status", "active"),
                    "matching_prefixes": list(matching_prefixes) if matching_prefixes else None,
                    "matching_suffixes": list(matching_suffixes) if matching_suffixes else None,
                    "is_periodicity_modification": is_periodicity_modification
                }
                results.append(match_details)
        
        # Sort by similarity score (descending)
        results.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Return top N results
        return results[:self.max_results]
    
    async def check_exact_match(self, 
                                title: str, 
                                existing_titles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Check if there's an exact match for the title.
        
        Args:
            title: The title to check
            existing_titles: List of existing title documents
            
        Returns:
            Match details or None if no exact match
        """
        normalized_title = self.normalize_title(title)
        
        for doc in existing_titles:
            existing_title = doc["title"]
            normalized_existing = self.normalize_title(existing_title)
            
            if normalized_title == normalized_existing:
                return {
                    "title": existing_title,
                    "similarity": 100.0,
                    "match_type": "exact",
                    "id": doc.get("id"),
                    "language": doc.get("language", "en"),
                    "status": doc.get("status", "active")
                }
                
        return None

# Create singleton instance
lexical_matcher = LexicalMatcher() 