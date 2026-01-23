"""Skill search service"""
from typing import List, Dict, Optional
import logging
import pymysql

from src.core.database import ElasticsearchClient, MySQLClient
from src.core.config import settings

logger = logging.getLogger(__name__)


class SkillSearchService:
    """Service for searching skills in ESCO taxonomy"""
    
    def __init__(self, elasticsearch_client: ElasticsearchClient, mysql_client: MySQLClient):
        self.es = elasticsearch_client.client
        self.mysql_client = mysql_client
        self.index = settings.ESCO_SKILLS_INDEX
    
    def search_by_name(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search skills by name using MySQL text search.
        MySQL has metadata (preferred_label, alternative_labels, description).
        Elasticsearch only has embeddings.
        
        Args:
            query: Search text
            limit: Maximum number of results
            
        Returns:
            List of skill dictionaries
        """
        try:
            conn = self.mysql_client.get_connection()
            cursor = conn.cursor(pymysql.cursors.DictCursor)
            
            # Full-text search in MySQL
            sql = """
                SELECT 
                    skill_id,
                    preferred_label,
                    description,
                    skill_type,
                    alternative_labels
                FROM esco_skills
                WHERE 
                    preferred_label LIKE %s 
                    OR alternative_labels LIKE %s
                    OR description LIKE %s
                ORDER BY 
                    CASE 
                        WHEN preferred_label LIKE %s THEN 1
                        WHEN alternative_labels LIKE %s THEN 2
                        ELSE 3
                    END
                LIMIT %s
            """
            
            search_pattern = f"%{query}%"
            exact_pattern = f"%{query}%"
            
            cursor.execute(sql, (
                search_pattern, search_pattern, search_pattern,
                exact_pattern, exact_pattern,
                limit
            ))
            
            results = []
            for row in cursor.fetchall():
                results.append({
                    "skill_id": row["skill_id"],
                    "label": row["preferred_label"],
                    "description": row.get("description", ""),
                    "skill_type": row.get("skill_type", ""),
                    "score": 1.0  # MySQL doesn't have relevance score
                })
            
            cursor.close()
            conn.close()
            
            logger.info(f"MySQL search '{query}' found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"MySQL search error: {e}", exc_info=True)
            return []
    
    def get_skill_by_id(self, skill_id: str) -> Optional[Dict]:
        """
        Get skill details by ID from MySQL.
        
        Args:
            skill_id: ESCO skill ID
            
        Returns:
            Skill dictionary or None if not found
        """
        try:
            conn = self.mysql_client.get_connection()
            cursor = conn.cursor(pymysql.cursors.DictCursor)
            
            sql = """
                SELECT 
                    skill_id,
                    preferred_label,
                    description,
                    skill_type,
                    alternative_labels
                FROM esco_skills
                WHERE skill_id = %s
            """
            
            cursor.execute(sql, (skill_id,))
            row = cursor.fetchone()
            
            cursor.close()
            conn.close()
            
            if row:
                return {
                    "skill_id": row["skill_id"],
                    "label": row["preferred_label"],
                    "description": row.get("description", ""),
                    "skill_type": row.get("skill_type", ""),
                }
            return None
            
        except Exception as e:
            logger.error(f"Get skill by ID error: {e}", exc_info=True)
            return None
    
    def search_by_vector(self, embedding: List[float], limit: int = 10) -> List[Dict]:
        """
        Search skills by embedding vector (semantic search).
        
        Args:
            embedding: Skill embedding vector
            limit: Maximum number of results
            
        Returns:
            List of skill dictionaries with similarity scores
        """
        try:
            search_query = {
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'vector') + 1.0",
                            "params": {"query_vector": embedding},
                        },
                    }
                },
                "size": limit,
                "_source": ["skill_id", "preferred_label", "description"],
            }
            
            response = self.es.search(index=self.index, body=search_query)
            hits = response["hits"]["hits"]
            
            results = []
            for hit in hits:
                source = hit["_source"]
                results.append({
                    "skill_id": source.get("skill_id"),
                    "label": source.get("preferred_label"),
                    "description": source.get("description"),
                    "score": hit["_score"],
                })
            
            logger.info(f"Vector search found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Vector search error: {e}", exc_info=True)
            return []
