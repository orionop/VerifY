"""
Content rule checker service for enforcing PRGI title guidelines.

This service implements checks for:
1. Disallowed words
2. Combinations of existing titles
3. Prefix/suffix rule violations
4. Periodicity modifications
"""

import re
import logging
from typing import List, Dict, Any, Set, Tuple
from rapidfuzz import fuzz

from app.core.config.settings import settings
from app.services.matchers.lexical_matcher import LexicalMatcher
from app.models.title import DisallowedWordMatch, CombinationMatch

logger = logging.getLogger(__name__)

class ContentRuleChecker:
    """
    Rule-based checker for content guidelines enforcement.
    
    This checker implements various rules specified by PRGI for title verification,
    such as disallowed words, combination detection, etc.
    """
    
    def __init__(self):
        """Initialize the rule checker with configuration settings."""
        self.disallowed_words = [word.lower() for word in settings.DISALLOWED_WORDS]
        self.common_prefixes = set(word.lower() for word in settings.COMMON_PREFIXES)
        self.common_suffixes = set(word.lower() for word in settings.COMMON_SUFFIXES)
        self.periodicity_terms = set(word.lower() for word in settings.PERIODICITY_TERMS)
        self.lexical_matcher = LexicalMatcher()  # For normalization and token extraction
    
    async def check_disallowed_words(self, title: str) -> List[DisallowedWordMatch]:
        """
        Check if the title contains any disallowed words.
        
        Args:
            title: The title to check
            
        Returns:
            List of disallowed word matches found
        """
        normalized_title = self.lexical_matcher.normalize_title(title).lower()
        tokens = self.lexical_matcher.extract_tokens(title)
        results = []
        
        # Check for each disallowed word using word boundary regex patterns
        for word in self.disallowed_words:
            # For multi-word disallowed terms
            if ' ' in word:
                if word in normalized_title:
                    results.append(DisallowedWordMatch(
                        word=word,
                        context=f"The term '{word}' is not allowed in titles",
                        rule_type="disallowed_word"
                    ))
                continue
                
            # For single word checking with word boundaries
            pattern = r'\b' + re.escape(word) + r'\b'
            if re.search(pattern, normalized_title):
                results.append(DisallowedWordMatch(
                    word=word,
                    context=f"The word '{word}' is not allowed in titles",
                    rule_type="disallowed_word"
                ))
        
        return results
    
    def _get_token_combinations(self, tokens: List[str], min_len: int = 2) -> List[Tuple[List[str], List[str]]]:
        """
        Get all possible ways to split a list of tokens into two parts.
        
        Args:
            tokens: List of tokens (words) from a title
            min_len: Minimum length (in tokens) for each part
            
        Returns:
            List of tuple pairs, each containing two parts of the original token list
        """
        n = len(tokens)
        if n < min_len * 2:
            return []
            
        combinations = []
        
        # Try all possible ways to split the tokens
        for i in range(min_len, n - min_len + 1):
            left = tokens[:i]
            right = tokens[i:]
            combinations.append((left, right))
            
        return combinations
    
    async def detect_title_combinations(self, 
                                      title: str, 
                                      existing_titles: List[Dict[str, Any]]) -> List[CombinationMatch]:
        """
        Detect if the title is a combination of two existing titles.
        
        Args:
            title: The title to check
            existing_titles: List of existing title documents
            
        Returns:
            List of combination matches found
        """
        tokens = self.lexical_matcher.extract_tokens(title)
        combinations = self._get_token_combinations(tokens)
        
        if not combinations:
            return []
            
        # Normalize existing titles for matching
        existing_title_dict = {}
        for doc in existing_titles:
            existing_title = doc["title"]
            normalized = self.lexical_matcher.normalize_title(existing_title)
            existing_title_dict[normalized] = existing_title
        
        results = []
        
        # Check each possible combination
        for left_tokens, right_tokens in combinations:
            left_text = ' '.join(left_tokens)
            right_text = ' '.join(right_tokens)
            
            # Look for exact matches first
            left_match = None
            right_match = None
            
            # Check for exact matches
            for normalized, original in existing_title_dict.items():
                if left_text == normalized and not left_match:
                    left_match = original
                if right_text == normalized and not right_match:
                    right_match = original
                    
                if left_match and right_match:
                    break
            
            # If no exact matches, try fuzzy matching
            if not left_match or not right_match:
                for normalized, original in existing_title_dict.items():
                    # Try fuzzy match for left part
                    if not left_match and fuzz.ratio(left_text, normalized) >= 85:
                        left_match = original
                        
                    # Try fuzzy match for right part
                    if not right_match and fuzz.ratio(right_text, normalized) >= 85:
                        right_match = original
                        
                    if left_match and right_match:
                        break
            
            # If both parts match existing titles, this is a combination
            if left_match and right_match:
                results.append(CombinationMatch(
                    first_title=left_match,
                    second_title=right_match,
                    context="Title appears to combine two existing titles"
                ))
                
                # Only report the first combination found to avoid excessive matches
                break
                
        return results
    
    async def check_prefix_suffix_rules(self, title: str) -> List[DisallowedWordMatch]:
        """
        Check for problematic prefix and suffix patterns.
        
        Args:
            title: The title to check
            
        Returns:
            List of rule violations found
        """
        tokens = self.lexical_matcher.extract_tokens(title)
        
        if len(tokens) < 2:
            return []
            
        results = []
        
        # Check first token (prefix)
        if tokens[0].lower() in self.common_prefixes:
            # Only flag common prefixes if they're being used with common patterns
            # This helps avoid false positives for legitimate uses
            rest_of_title = ' '.join(tokens[1:]).lower()
            
            # Check if prefix is paired with concerning patterns 
            # (Note: these patterns would be determined by PRGI guidelines)
            concerning_patterns = [
                r'\bnews\b', r'\btimes\b', r'\bexpress\b', r'\bdaily\b', 
                r'\bweekly\b', r'\bmonthly\b'
            ]
            
            if any(re.search(pattern, rest_of_title) for pattern in concerning_patterns):
                results.append(DisallowedWordMatch(
                    word=tokens[0].lower(),
                    context=f"Prefix '{tokens[0]}' with common title patterns may conflict with existing titles",
                    rule_type="prefix"
                ))
        
        # Check last token (suffix)
        if tokens[-1].lower() in self.common_suffixes:
            # Only flag common suffixes if they're being used with common patterns
            rest_of_title = ' '.join(tokens[:-1]).lower()
            
            # Check if suffix is paired with concerning patterns
            concerning_patterns = [
                r'\bindia\b', r'\bnational\b', r'\bdaily\b', r'\btimes\b', 
                r'\bexpress\b', r'\btoday\b'
            ]
            
            if any(re.search(pattern, rest_of_title) for pattern in concerning_patterns):
                results.append(DisallowedWordMatch(
                    word=tokens[-1].lower(),
                    context=f"Suffix '{tokens[-1]}' with common title patterns may conflict with existing titles",
                    rule_type="suffix"
                ))
                
        # Check periodicity terms
        for token in tokens:
            if token.lower() in self.periodicity_terms:
                # Note: detailed periodicity checks are also handled in lexical_matcher
                # This check is meant to catch standalone periodicity issues
                results.append(DisallowedWordMatch(
                    word=token.lower(),
                    context=f"Periodicity term '{token}' may create conflicts with existing titles",
                    rule_type="periodicity"
                ))
                break
                
        return results
    
    async def detect_periodicity_modifications(self, 
                                            title: str, 
                                            existing_titles: List[Dict[str, Any]]) -> List[DisallowedWordMatch]:
        """
        Detect if the title is an existing title with added periodicity.
        
        Args:
            title: The title to check
            existing_titles: List of existing title documents
            
        Returns:
            List of rule violations for periodicity modifications
        """
        title_tokens = self.lexical_matcher.extract_tokens(title)
        
        # Check if any periodicity terms exist in the title
        periodicity_tokens = [t for t in title_tokens if t.lower() in self.periodicity_terms]
        
        if not periodicity_tokens:
            return []
            
        results = []
        
        # Check each existing title to see if adding periodicity terms makes it match the new title
        for doc in existing_titles:
            existing_title = doc["title"]
            
            # Check if this is a periodicity modification
            if self.lexical_matcher.detect_periodicity_modification(title, existing_title):
                # Found a match - this title is a periodicity modification of an existing title
                periodicity_term = next((t for t in periodicity_tokens), "periodicity term")
                
                results.append(DisallowedWordMatch(
                    word=periodicity_term,
                    context=f"Adding '{periodicity_term}' to existing title '{existing_title}' is not allowed",
                    rule_type="periodicity_modification"
                ))
                
                # One violation is enough
                break
                
        return results
    
    async def check_all_rules(self, 
                            title: str, 
                            existing_titles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Run all content rule checks on the title.
        
        Args:
            title: The title to check
            existing_titles: List of existing title documents
            
        Returns:
            Dictionary with all rule check results
        """
        # Run all checks in parallel
        disallowed_word_matches = await self.check_disallowed_words(title)
        combination_matches = await self.detect_title_combinations(title, existing_titles)
        prefix_suffix_violations = await self.check_prefix_suffix_rules(title)
        periodicity_violations = await self.detect_periodicity_modifications(title, existing_titles)
        
        # Combine all prefix/suffix/periodicity violations
        all_word_violations = disallowed_word_matches + prefix_suffix_violations + periodicity_violations
        
        # Determine overall status
        is_rejected = bool(all_word_violations or combination_matches)
        
        return {
            "disallowed_words": all_word_violations,
            "combined_titles": combination_matches,
            "has_violations": is_rejected
        }

# Create singleton instance
content_rule_checker = ContentRuleChecker() 