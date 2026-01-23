"""Database connection clients"""
from neo4j import GraphDatabase
from elasticsearch import Elasticsearch
import pymysql
import logging
from typing import Optional

from src.core.config import settings

logger = logging.getLogger(__name__)


class Neo4jClient:
    """Neo4j graph database client"""
    
    def __init__(self):
        try:
            self.driver = GraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD)
            )
            logger.info(f"Connected to Neo4j at {settings.NEO4J_URI}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise ConnectionError(f"Neo4j connection failed: {e}")
    
    def close(self):
        """Close the driver connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")
    
    def execute_query(self, query: str, parameters: Optional[dict] = None):
        """Execute a Cypher query"""
        try:
            with self.driver.session() as session:
                result = session.run(query, parameters or {})
                return [record for record in result]
        except Exception as e:
            logger.error(f"Neo4j query error: {e}")
            raise
    
    def verify_connection(self) -> bool:
        """Verify database connection"""
        try:
            with self.driver.session() as session:
                session.run("RETURN 1")
            return True
        except Exception:
            return False


class ElasticsearchClient:
    """Elasticsearch vector database client"""
    
    def __init__(self):
        try:
            self.client = Elasticsearch(
                [settings.ELASTICSEARCH_HOST],
                request_timeout=settings.ELASTICSEARCH_TIMEOUT
            )
            logger.info(f"Connected to Elasticsearch at {settings.ELASTICSEARCH_HOST}")
        except Exception as e:
            logger.error(f"Failed to connect to Elasticsearch: {e}")
            raise ConnectionError(f"Elasticsearch connection failed: {e}")
    
    def close(self):
        """Close the client connection"""
        if self.client:
            self.client.close()
            logger.info("Elasticsearch connection closed")
    
    def verify_connection(self) -> bool:
        """Verify database connection"""
        try:
            return self.client.ping()
        except Exception:
            return False


class MySQLClient:
    """MySQL database client"""
    
    def __init__(self):
        self.config = {
            'host': settings.MYSQL_HOST,
            'port': settings.MYSQL_PORT,
            'user': settings.MYSQL_USER,
            'password': settings.MYSQL_PASSWORD,
            'database': settings.MYSQL_DATABASE,
            'charset': 'utf8mb4'
        }
        logger.info(f"MySQL client configured for {settings.MYSQL_HOST}")
    
    def get_connection(self):
        """Get a new database connection"""
        try:
            return pymysql.connect(**self.config)
        except Exception as e:
            logger.error(f"Failed to connect to MySQL: {e}")
            raise ConnectionError(f"MySQL connection failed: {e}")
    
    def verify_connection(self) -> bool:
        """Verify database connection"""
        try:
            conn = self.get_connection()
            conn.close()
            return True
        except Exception:
            return False


# Singleton instances - will be initialized in dependencies
neo4j_client: Optional[Neo4jClient] = None
elasticsearch_client: Optional[ElasticsearchClient] = None
mysql_client: Optional[MySQLClient] = None


def get_neo4j_client() -> Neo4jClient:
    """Get or create Neo4j client instance"""
    global neo4j_client
    if neo4j_client is None:
        neo4j_client = Neo4jClient()
    return neo4j_client


def get_elasticsearch_client() -> ElasticsearchClient:
    """Get or create Elasticsearch client instance"""
    global elasticsearch_client
    if elasticsearch_client is None:
        elasticsearch_client = ElasticsearchClient()
    return elasticsearch_client


def get_mysql_client() -> MySQLClient:
    """Get or create MySQL client instance"""
    global mysql_client
    if mysql_client is None:
        mysql_client = MySQLClient()
    return mysql_client


def close_all_connections():
    """Close all database connections"""
    global neo4j_client, elasticsearch_client, mysql_client
    
    if neo4j_client:
        neo4j_client.close()
        neo4j_client = None
    
    if elasticsearch_client:
        elasticsearch_client.close()
        elasticsearch_client = None
    
    mysql_client = None
    
    logger.info("All database connections closed")
