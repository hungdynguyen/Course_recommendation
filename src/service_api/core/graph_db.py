from neo4j import GraphDatabase
import logging
import os

logger = logging.getLogger(__name__)


class Neo4jClient:
    def __init__(self):
        # Load from environment variables
        neo4j_uri = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
        neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "password123")
        
        self.driver = GraphDatabase.driver(
            neo4j_uri, auth=(neo4j_user, neo4j_password)
        )

    def close(self):
        self.driver.close()

    def execute_query(self, query, parameters=None):
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters)
                return [record for record in result]
        except Exception as e:
            logger.error(f"Neo4j Query Error: {e}")
            raise


# Singleton instance
neo4j_client = Neo4jClient()