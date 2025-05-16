"""
Main verification service for the PRGI Title Verification System.

This service orchestrates the entire verification process, coordinating the various
matcher services and applying business rules to determine the final verification result.
"""

import logging
import asyncio
import uuid
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import re

from app.core.config.settings import settings
from app.models.title import VerificationResponse, SimilarTitle, DisallowedWordMatch, CombinationMatch, TitleRequest
from app.services.matchers.lexical_matcher import lexical_matcher
from app.services.matchers.phonetic_matcher import phonetic_matcher
from app.services.matchers.semantic_matcher import semantic_matcher
from app.services.matchers.content_rule_checker import content_rule_checker
from app.services.elasticsearch_service import get_elasticsearch_service
from app.services.embedding_service import embedding_service

logger = logging.getLogger(__name__)

class VerificationService:
    """
    Core service for title verification orchestration.
    
    This service:
    1. Coordinates all the different verification checks
    2. Applies business rules to determine the final result
    3. Calculates verification probabilities
    4. Prepares detailed feedback for users
    """
    
    def __init__(self):
        """Initialize the verification service."""
        self.high_similarity_threshold = settings.HIGH_SIMILARITY_THRESHOLD
        self.text_match_weight = settings.TEXT_MATCH_WEIGHT
        self.phonetic_match_weight = settings.PHONETIC_MATCH_WEIGHT
        self.semantic_match_weight = settings.SEMANTIC_MATCH_WEIGHT
        self.max_results = settings.MAX_TITLES_RETURNED
    
    async def load_existing_titles(self, title: str, language: str) -> List[Dict[str, Any]]:
        """
        Load existing titles from the database for comparison.
        
        In a real implementation, this would fetch from Elasticsearch.
        For the prototype, we'll use a mock database.
        
        Args:
            title: Title being verified (for potential optimizations)
            language: Language of the title
            
        Returns:
            List of title documents
        """
        # In production, this would use Elasticsearch to efficiently retrieve
        # only the most relevant titles to check against
        
        # Mock database with sample titles
        mock_titles = [
            {
                "id": "title_001",
                "title": "Daily News",
                "language": "en",
                "status": "active",
                "created_at": datetime.now().isoformat()
            },
            {
                "id": "title_002",
                "title": "India Today",
                "language": "en",
                "status": "active",
                "created_at": datetime.now().isoformat()
            },
            {
                "id": "title_003",
                "title": "The Hindu",
                "language": "en",
                "status": "active",
                "created_at": datetime.now().isoformat()
            },
            {
                "id": "title_004",
                "title": "Times of India",
                "language": "en",
                "status": "active",
                "created_at": datetime.now().isoformat()
            },
            {
                "id": "title_005",
                "title": "Indian Express",
                "language": "en",
                "status": "active",
                "created_at": datetime.now().isoformat()
            },
            {
                "id": "title_006",
                "title": "The Telegraph",
                "language": "en",
                "status": "active",
                "created_at": datetime.now().isoformat()
            },
            {
                "id": "title_007",
                "title": "Hindustan Times",
                "language": "en",
                "status": "active",
                "created_at": datetime.now().isoformat()
            },
            {
                "id": "title_008",
                "title": "Financial Express",
                "language": "en",
                "status": "active",
                "created_at": datetime.now().isoformat()
            },
            {
                "id": "title_009",
                "title": "Mumbai Mirror",
                "language": "en",
                "status": "active",
                "created_at": datetime.now().isoformat()
            },
            {
                "id": "title_010",
                "title": "Dainik Jagran",
                "language": "hi",
                "status": "active",
                "created_at": datetime.now().isoformat()
            },
            {
                "id": "title_011",
                "title": "Dainik Bhaskar",
                "language": "hi",
                "status": "active",
                "created_at": datetime.now().isoformat()
            },
            {
                "id": "title_012",
                "title": "Amar Ujala",
                "language": "hi", 
                "status": "active",
                "created_at": datetime.now().isoformat()
            },
            {
                "id": "title_013",
                "title": "Patent News Daily",
                "language": "en",
                "status": "active",
                "created_at": datetime.now().isoformat()
            },
            {
                "id": "title_014",
                "title": "Police Times",
                "language": "en",
                "status": "active", 
                "created_at": datetime.now().isoformat()
            }
        ]
        
        # In production, we would add embeddings to each title
        # Here we'll generate them on demand
        for doc in mock_titles:
            # Add embedding vector if needed
            if not doc.get("title_vector"):
                doc["title_vector"] = await embedding_service.get_embedding(
                    doc["title"], doc["language"]
                )
                
        return mock_titles
    
    def _process_matcher_results(self, 
                                 results: List[Dict[str, Any]]) -> List[SimilarTitle]:
        """
        Process raw matcher results into standardized SimilarTitle objects.
        
        Args:
            results: Raw matcher results
            
        Returns:
            List of standardized SimilarTitle objects
        """
        similar_titles = []
        
        for result in results:
            similar_title = SimilarTitle(
                title=result["title"],
                similarity=result["similarity"],
                match_type=result["match_type"],
                id=result.get("id"),
                language=result.get("language", "en"),
                status=result.get("status", "active")
            )
            similar_titles.append(similar_title)
            
        return similar_titles
    
    def _deduplicate_similar_titles(self, 
                                    titles: List[SimilarTitle]) -> List[SimilarTitle]:
        """
        Deduplicate similar titles, keeping the highest similarity score.
        
        Args:
            titles: List of similar titles, potentially with duplicates
            
        Returns:
            Deduplicated list of similar titles
        """
        title_dict = {}
        
        for title in titles:
            key = title.title.lower()
            
            if key in title_dict:
                # Keep the higher similarity score
                if title.similarity > title_dict[key].similarity:
                    title_dict[key] = title
            else:
                title_dict[key] = title
                
        return list(title_dict.values())
    
    def _compute_overall_match_score(self, 
                                    similar_titles: List[SimilarTitle]) -> float:
        """
        Compute the overall match score based on all similar titles.
        
        This uses a weighted approach that considers:
        - The highest similarity across all match types
        - The type of match (lexical, phonetic, semantic)
        - Multiple high-scoring matches
        
        Args:
            similar_titles: List of similar titles
            
        Returns:
            Overall match score (0-100)
        """
        if not similar_titles:
            return 0.0
            
        # Group by match type
        match_types = {}
        for title in similar_titles:
            match_type = title.match_type
            if match_type not in match_types:
                match_types[match_type] = []
            match_types[match_type].append(title.similarity)
            
        # Get highest score for each match type
        highest_scores = {}
        for match_type, scores in match_types.items():
            highest_scores[match_type] = max(scores)
            
        # Apply weights by match type
        weighted_scores = []
        
        if "exact" in highest_scores:
            # Exact match gets highest priority
            return 100.0
            
        if "lexical" in highest_scores:
            weighted_scores.append(highest_scores["lexical"] * self.text_match_weight)
            
        if "phonetic" in highest_scores:
            weighted_scores.append(highest_scores["phonetic"] * self.phonetic_match_weight)
            
        if "semantic" in highest_scores:
            weighted_scores.append(highest_scores["semantic"] * self.semantic_match_weight)
            
        if "cross_language" in highest_scores:
            weighted_scores.append(highest_scores["cross_language"] * self.semantic_match_weight)
            
        if not weighted_scores:
            return 0.0
            
        # Use a weighted average based on the weights
        total_weight = (
            self.text_match_weight + 
            self.phonetic_match_weight + 
            self.semantic_match_weight
        )
        
        # Scale up to ensure high scores for strong matches
        scaling_factor = 1.2
        
        # Calculate weighted average and apply scaling
        max_score = max(weighted_scores)
        weighted_avg = sum(weighted_scores) / total_weight
        
        # Blend maximum score with weighted average
        blended_score = (0.7 * max_score) + (0.3 * weighted_avg)
        
        # Apply scaling, ensuring we don't exceed 100
        final_score = min(100, blended_score * scaling_factor)
        
        return round(final_score, 1)
    
    def _calculate_approval_probability(self, 
                                       match_score: float, 
                                       has_rule_violations: bool) -> float:
        """
        Calculate the probability of title approval.
        
        Args:
            match_score: Overall match score (0-100)
            has_rule_violations: Whether the title violates any rules
            
        Returns:
            Approval probability (0-100)
        """
        if has_rule_violations:
            # Title has rule violations - automatic rejection
            return 0.0
            
        if match_score >= 95:
            # Nearly identical to existing title
            return 0.0
            
        if match_score >= self.high_similarity_threshold:
            # High similarity - calculate probability based on match score
            # Higher match score = lower approval probability
            return round(100 - match_score, 1)
            
        # Moderate to low similarity - higher approval probability
        # Scale up the probability as match_score decreases
        base_probability = 100 - (match_score * 0.7)  # 70% influence from match score
        
        # Apply minimum probability floor
        min_probability = 30
        
        return round(max(min_probability, base_probability), 1)
    
    def _generate_feedback(self, 
                          title: str,
                          similar_titles: List[SimilarTitle],
                          disallowed_words: List[DisallowedWordMatch],
                          combined_titles: List[CombinationMatch],
                          match_score: float,
                          approval_probability: float) -> str:
        """
        Generate human-readable feedback about the verification result.
        
        Args:
            title: The title being verified
            similar_titles: List of similar titles found
            disallowed_words: List of disallowed word matches
            combined_titles: List of combined title matches
            match_score: Overall match score
            approval_probability: Approval probability
            
        Returns:
            Human-readable feedback message
        """
        # Start with basic message
        if approval_probability <= 5:
            if disallowed_words:
                primary_reason = f"disallowed {disallowed_words[0].rule_type} '{disallowed_words[0].word}'"
            elif combined_titles:
                primary_reason = f"combination of existing titles '{combined_titles[0].first_title}' and '{combined_titles[0].second_title}'"
            elif similar_titles:
                primary_reason = f"similarity to existing title '{similar_titles[0].title}' ({similar_titles[0].similarity}%)"
            else:
                primary_reason = "unknown issue"
                
            message = f"Title rejected due to {primary_reason}."
        elif approval_probability < 50:
            message = f"Title has a low approval probability of {approval_probability}%."
        else:
            message = f"Title has a good approval probability of {approval_probability}%."
        
        # Add details about similar titles if available
        if similar_titles and not disallowed_words and not combined_titles:
            top_match = similar_titles[0]
            if top_match.similarity > 80:
                message += f" Very similar to existing title '{top_match.title}'."
            elif top_match.similarity > 65:
                message += f" Moderately similar to existing title '{top_match.title}'."
        
        # Add advice for improving chances
        if disallowed_words:
            message += " Please remove disallowed words and try again."
        elif combined_titles:
            message += " Please create a more original title."
        elif approval_probability < 50 and similar_titles:
            message += " Consider a more distinctive title."
            
        return message
    
    async def verify_title(self, request: TitleRequest) -> VerificationResponse:
        """
        Verify a title against existing titles and rules.
        
        This is the main entry point for title verification, orchestrating
        all checks and determining the final verification result.
        
        Args:
            request: Title verification request
            
        Returns:
            Verification response with detailed results
        """
        start_time = time.time()
        title = request.title
        language = request.language
        request_id = str(uuid.uuid4())
        
        logger.info(f"Starting verification for title: '{title}' (language: {language})")
        
        # Load existing titles to check against
        existing_titles = await self.load_existing_titles(title, language)
        
        # Run checks in parallel for efficiency
        # 1. Exact match check (fastest, exits early if found)
        exact_match = await lexical_matcher.check_exact_match(title, existing_titles)
        
        if exact_match:
            logger.info(f"Exact match found for title: '{title}'")
            
            # Create verification response for exact match
            verification_time_ms = int((time.time() - start_time) * 1000)
            
            return VerificationResponse(
                title=title,
                similar_titles=[SimilarTitle(
                    title=exact_match["title"],
                    similarity=100.0,
                    match_type="exact",
                    id=exact_match.get("id"),
                    language=exact_match.get("language", "en"),
                    status=exact_match.get("status", "active")
                )],
                disallowed_words=[],
                combined_titles=[],
                match_score=100.0,
                status="Rejected",
                approval_probability=0.0,
                feedback=f"Title rejected: Exact match with existing title '{exact_match['title']}'.",
                request_id=request_id,
                verification_time_ms=verification_time_ms
            )
        
        # 2. Content rule checks and similarity checks in parallel
        lexical_results, phonetic_results, semantic_results, content_rule_results = await asyncio.gather(
            lexical_matcher.find_similar(title, existing_titles),
            phonetic_matcher.find_phonetic_matches(title, existing_titles),
            semantic_matcher.find_semantic_matches(title, existing_titles, language),
            content_rule_checker.check_all_rules(title, existing_titles)
        )
        
        # 3. Additional cross-language check if needed
        cross_lang_results = []
        if language != "en":  # Only do cross-language for non-English titles
            cross_lang_results = await semantic_matcher.detect_cross_language_duplicates(
                title, language, existing_titles
            )
        
        # Process and combine results
        all_similar_titles = (
            self._process_matcher_results(lexical_results) +
            self._process_matcher_results(phonetic_results) +
            self._process_matcher_results(semantic_results) +
            self._process_matcher_results(cross_lang_results)
        )
        
        # Deduplicate and sort similar titles
        deduplicated_titles = self._deduplicate_similar_titles(all_similar_titles)
        deduplicated_titles.sort(key=lambda x: x.similarity, reverse=True)
        
        # Calculate overall match score
        match_score = self._compute_overall_match_score(deduplicated_titles)
        
        # Extract content rule violations
        disallowed_words = content_rule_results["disallowed_words"]
        combined_titles = content_rule_results["combined_titles"]
        has_rule_violations = content_rule_results["has_violations"]
        
        # Determine verification status and probability
        approval_probability = self._calculate_approval_probability(
            match_score, has_rule_violations
        )
        
        status = "Accepted" if approval_probability > 50 else "Rejected"
        
        # Generate human-readable feedback
        feedback = self._generate_feedback(
            title,
            deduplicated_titles[:3],  # Top 3 similar titles
            disallowed_words,
            combined_titles,
            match_score,
            approval_probability
        )
        
        # Measure verification time
        verification_time_ms = int((time.time() - start_time) * 1000)
        
        # Limit the number of similar titles in the response
        top_similar_titles = deduplicated_titles[:self.max_results]
        
        # Create and return the final verification response
        verification_response = VerificationResponse(
            title=title,
            similar_titles=top_similar_titles,
            disallowed_words=disallowed_words,
            combined_titles=combined_titles,
            match_score=match_score,
            status=status,
            approval_probability=approval_probability,
            feedback=feedback,
            request_id=request_id,
            verification_time_ms=verification_time_ms
        )
        
        logger.info(
            f"Completed verification for '{title}'. "
            f"Status: {status}, Probability: {approval_probability}%, "
            f"Time: {verification_time_ms}ms"
        )
        
        return verification_response

# Create singleton instance
verification_service = VerificationService()

async def verify_title_service(title: str, language: str = "en") -> Dict[str, Any]:
    """
    Service function for verifying a title (used by API endpoints).
    
    Args:
        title: The title to verify
        language: ISO 639-1 language code
        
    Returns:
        Verification result as a dictionary
    """
    request = TitleRequest(title=title, language=language)
    result = await verification_service.verify_title(request)
    
    # Convert to dict for API response
    return result.dict() 