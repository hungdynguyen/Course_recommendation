from typing import List, Dict
import logging
import pymysql
import os

from elasticsearch import Elasticsearch

logger = logging.getLogger(__name__)


class SkillSearchService:
    def __init__(self):
        # Elasticsearch for vector search
        es_host = os.getenv("ELASTICSEARCH_HOST", "http://elasticsearch:9200")
        self.es = Elasticsearch([es_host], request_timeout=30)
        self.index = "esco_skills"
        
        # MySQL for metadata search
        self.mysql_config = {
            'host': os.getenv('MYSQL_HOST', 'mysql'),
            'port': int(os.getenv('MYSQL_PORT', 3306)),
            'user': os.getenv('MYSQL_USER', 'vietcv'),
            'password': os.getenv('MYSQL_PASSWORD', 'yourpassword'),
            'database': os.getenv('MYSQL_DATABASE', 'vietcv'),
            'charset': 'utf8mb4'
        }

    def close(self):
        self.es.close()

    def search_by_name(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search skills by name using MySQL text search.
        MySQL has metadata (preferred_label, alternative_labels, description).
        Elasticsearch only has embeddings.
        """
        try:
            conn = pymysql.connect(**self.mysql_config)
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

    def search_by_vector(self, embedding: List[float], limit: int = 10) -> List[Dict]:
        """
        Search skills by embedding vector (semantic search).
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
                results.append(
                    {
                        "skill_id": source.get("skill_id"),
                        "label": source.get("preferred_label"),
                        "description": source.get("description"),
                        "score": hit["_score"],
                    }
                )

            return results
        except Exception as e:
            logger.error(f"Vector search error: {e}")
            return []