from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_bulk
import os
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# In production, these would come from environment variables
ES_HOST = os.environ.get("ES_HOST", "localhost")
ES_PORT = os.environ.get("ES_PORT", "9200")
ES_USER = os.environ.get("ES_USER", "")
ES_PASSWORD = os.environ.get("ES_PASSWORD", "")
ES_INDEX = os.environ.get("ES_INDEX", "titles")

class ElasticsearchService:
    """Service for interacting with Elasticsearch for title storage and retrieval."""
    
    def __init__(self):
        """Initialize the Elasticsearch client."""
        # In this example, we're not actually connecting to ES
        # In production, you would use real credentials and connection
        self.client = None
        self.connected = False
    
    async def connect(self) -> None:
        """Establish connection to Elasticsearch."""
        try:
            auth = None
            if ES_USER and ES_PASSWORD:
                auth = (ES_USER, ES_PASSWORD)
            
            self.client = AsyncElasticsearch(
                [f"http://{ES_HOST}:{ES_PORT}"],
                http_auth=auth,
                retry_on_timeout=True,
                max_retries=3
            )
            self.connected = True
            logger.info("Connected to Elasticsearch")
        except Exception as e:
            logger.error(f"Failed to connect to Elasticsearch: {e}")
            self.connected = False
    
    async def close(self) -> None:
        """Close the Elasticsearch connection."""
        if self.client and self.connected:
            await self.client.close()
            logger.info("Elasticsearch connection closed")
    
    async def create_index(self) -> None:
        """
        Create the titles index with appropriate mappings for text search.
        
        This sets up:
        - Standard text fields for exact and fuzzy matching
        - Phonetic analysis for sound-alike matching
        - Keyword fields for exact matches
        - Dense vector field for semantic similarity (via BERT embeddings)
        """
        if not self.connected or not self.client:
            logger.error("Not connected to Elasticsearch")
            return
        
        # Check if index already exists
        if await self.client.indices.exists(index=ES_INDEX):
            logger.info(f"Index {ES_INDEX} already exists")
            return
        
        # Define index settings with custom analyzers
        settings = {
            "settings": {
                "analysis": {
                    "analyzer": {
                        "phonetic_analyzer": {
                            "tokenizer": "standard",
                            "filter": ["lowercase", "phonetic_filter"]
                        }
                    },
                    "filter": {
                        "phonetic_filter": {
                            "type": "phonetic",
                            "encoder": "metaphone",
                            "replace": False
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "title": {
                        "type": "text",
                        "analyzer": "standard",
                        "fields": {
                            "keyword": {"type": "keyword"},
                            "phonetic": {
                                "type": "text",
                                "analyzer": "phonetic_analyzer"
                            }
                        }
                    },
                    "title_vector": {
                        "type": "dense_vector",
                        "dims": 384  # Dimension for all-MiniLM-L6-v2 embeddings
                    },
                    "created_at": {"type": "date"},
                    "status": {"type": "keyword"},
                    "metadata": {"type": "object"}
                }
            }
        }
        
        try:
            await self.client.indices.create(index=ES_INDEX, body=settings)
            logger.info(f"Created index {ES_INDEX} with phonetic analyzer")
        except Exception as e:
            logger.error(f"Error creating index: {e}")
    
    async def index_title(self, title: str, vector: List[float], metadata: Dict = None) -> bool:
        """
        Index a title document with its vector embedding.
        
        Args:
            title: The title text
            vector: BERT embedding vector for semantic search
            metadata: Additional metadata about the title
            
        Returns:
            bool: True if indexing was successful
        """
        if not self.connected or not self.client:
            logger.error("Not connected to Elasticsearch")
            return False
        
        doc = {
            "title": title,
            "title_vector": vector,
            "created_at": "now",
            "status": "active",
            "metadata": metadata or {}
        }
        
        try:
            await self.client.index(index=ES_INDEX, body=doc)
            return True
        except Exception as e:
            logger.error(f"Error indexing title: {e}")
            return False
    
    async def bulk_index_titles(self, titles: List[Dict[str, Any]]) -> bool:
        """
        Bulk index multiple title documents.
        
        Args:
            titles: List of title documents with vectors
            
        Returns:
            bool: True if bulk indexing was successful
        """
        if not self.connected or not self.client:
            logger.error("Not connected to Elasticsearch")
            return False
        
        # Prepare actions for bulk indexing
        actions = []
        for title_doc in titles:
            action = {
                "_index": ES_INDEX,
                "_source": {
                    "title": title_doc["title"],
                    "title_vector": title_doc["vector"],
                    "created_at": "now",
                    "status": "active",
                    "metadata": title_doc.get("metadata", {})
                }
            }
            actions.append(action)
        
        try:
            await async_bulk(self.client, actions)
            return True
        except Exception as e:
            logger.error(f"Error in bulk indexing: {e}")
            return False
    
    async def search_similar_titles(
        self, 
        title: str, 
        vector: Optional[List[float]] = None,
        min_score: float = 60.0,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search for similar titles using a multi-match approach.
        
        Args:
            title: Title to search for
            vector: Optional embedding vector for semantic search
            min_score: Minimum score threshold (0-100)
            limit: Maximum number of results to return
            
        Returns:
            List of similar titles with similarity scores
        """
        if not self.connected or not self.client:
            logger.error("Not connected to Elasticsearch")
            return []
        
        # Multi-dimensional search query
        query = {
            "size": limit,
            "query": {
                "bool": {
                    "should": [
                        # Fuzzy text matching
                        {
                            "match": {
                                "title": {
                                    "query": title,
                                    "fuzziness": "AUTO",
                                    "boost": 1.0
                                }
                            }
                        },
                        # Phonetic matching
                        {
                            "match": {
                                "title.phonetic": {
                                    "query": title,
                                    "boost": 1.5
                                }
                            }
                        }
                    ],
                    "minimum_should_match": 1
                }
            }
        }
        
        # Add vector search if embedding is provided
        if vector:
            vector_query = {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.vector, 'title_vector') + 1.0",
                        "params": {"vector": vector}
                    }
                }
            }
            query["query"]["bool"]["should"].append(vector_query)
        
        try:
            response = await self.client.search(index=ES_INDEX, body=query)
            
            # Process and normalize the results
            results = []
            for hit in response["hits"]["hits"]:
                # Normalize score to 0-100 range (ES returns scores typically 0-1 or higher)
                score = min(100, int(hit["_score"] * 20))  # Adjust scaling factor as needed
                
                results.append({
                    "title": hit["_source"]["title"],
                    "similarity": score,
                    "metadata": hit["_source"].get("metadata", {})
                })
            
            return results
        except Exception as e:
            logger.error(f"Error searching similar titles: {e}")
            return []

# Singleton instance
es_service = ElasticsearchService()

async def get_elasticsearch_service() -> ElasticsearchService:
    """Get or initialize the Elasticsearch service."""
    if not es_service.connected:
        await es_service.connect()
    return es_service 