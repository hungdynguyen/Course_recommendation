from elasticsearch import Elasticsearch
from src.config import settings

# Tăng timeout vì lần đầu connect có thể lâu
es_client = Elasticsearch(
    settings.ES_URL, 
    request_timeout=30
)

def create_index_if_not_exists(index_name: str):
    if not es_client.indices.exists(index=index_name):
        es_client.indices.create(index=index_name)
        print(f"Created index: {index_name}")