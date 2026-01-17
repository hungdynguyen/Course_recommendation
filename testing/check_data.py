"""
Check if data exists in databases (Elasticsearch, Neo4j)
"""

from elasticsearch import Elasticsearch
from neo4j import GraphDatabase
import os


def check_elasticsearch():
    """Check Elasticsearch has ESCO skills data"""
    print("="*80)
    print("1. Checking Elasticsearch")
    print("="*80)
    
    es_host = os.getenv("ELASTICSEARCH_HOST", "http://elasticsearch:9200")
    es = Elasticsearch([es_host], request_timeout=30)
    index = "esco_skills"
    
    try:
        # Check index exists
        if not es.indices.exists(index=index):
            print(f"❌ Index '{index}' does NOT exist!")
            print("\n💡 Run this to create index:")
            print("   docker exec -it vietcv_data_factory python -m src.data_factory.scripts.index_skills")
            es.close()
            return False
        
        # Count documents
        count_result = es.count(index=index)
        count = count_result['count']
        
        if count == 0:
            print(f"❌ Index '{index}' exists but is EMPTY (0 documents)")
            print("\n💡 Run this to populate index:")
            print("   docker exec -it vietcv_data_factory python -m src.data_factory.scripts.index_skills")
            es.close()
            return False
        
        print(f"✅ Index '{index}' exists with {count:,} documents")
        
        # Get sample
        response = es.search(
            index=index,
            body={"query": {"match_all": {}}, "size": 5, "_source": ["preferred_label", "skill_type"]}
        )
        
        print("\nSample skills:")
        for hit in response["hits"]["hits"]:
            label = hit["_source"].get("preferred_label", "N/A")
            skill_type = hit["_source"].get("skill_type", "N/A")
            print(f"  • {label} ({skill_type})")
        
        # Test search
        print("\nTesting search...")
        test_queries = ["insurance", "analysis", "management"]
        for query in test_queries:
            search_result = es.search(
                index=index,
                body={
                    "query": {
                        "multi_match": {
                            "query": query,
                            "fields": ["preferred_label^2", "alternative_labels", "description"],
                            "fuzziness": "AUTO"
                        }
                    },
                    "size": 1
                }
            )
            hits = search_result["hits"]["hits"]
            if hits:
                print(f"  ✅ '{query}' → {hits[0]['_source']['preferred_label']} (score: {hits[0]['_score']:.1f})")
            else:
                print(f"  ❌ '{query}' → No results")
        
        es.close()
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        es.close()
        return False


def check_neo4j():
    """Check Neo4j has courses and skills data"""
    print("\n" + "="*80)
    print("2. Checking Neo4j")
    print("="*80)
    
    neo4j_uri = os.getenv("NEO4J_URI", "bolt://neo4j:7687")
    neo4j_user = os.getenv("NEO4J_USERNAME", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password123")
    
    try:
        driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        
        with driver.session() as session:
            # Count skills
            result = session.run("MATCH (s:Skill) RETURN count(s) as count")
            skill_count = result.single()["count"]
            
            # Count courses
            result = session.run("MATCH (c:Course) RETURN count(c) as count")
            course_count = result.single()["count"]
            
            # Count TEACHES relationships
            result = session.run("MATCH ()-[r:TEACHES]->() RETURN count(r) as count")
            teaches_count = result.single()["count"]
            
            # Count REQUIRES relationships
            result = session.run("MATCH ()-[r:REQUIRES]->() RETURN count(r) as count")
            requires_count = result.single()["count"]
            
            print(f"Skills: {skill_count:,}")
            print(f"Courses: {course_count:,}")
            print(f"TEACHES edges: {teaches_count:,}")
            print(f"REQUIRES edges: {requires_count:,}")
            
            if skill_count == 0 or course_count == 0:
                print("\n❌ Graph is EMPTY!")
                print("\n💡 Run this to build graph:")
                print("   docker exec -it vietcv_data_factory python -m src.data_factory.scripts.build_graph")
                driver.close()
                return False
            
            print("\n✅ Neo4j has data")
            
            # Get sample courses
            result = session.run("""
                MATCH (c:Course)
                RETURN c.course_id as id, c.course_title as title
                LIMIT 5
            """)
            
            print("\nSample courses:")
            for record in result:
                print(f"  • {record['title']} ({record['id']})")
            
            # Test course-skill relationship
            result = session.run("""
                MATCH (c:Course)-[:TEACHES]->(s:Skill)
                RETURN c.course_title as course, s.preferred_label as skill
                LIMIT 3
            """)
            
            print("\nSample course-skill mappings:")
            records = list(result)
            if records:
                for record in records:
                    print(f"  • {record['course']} → {record['skill']}")
            else:
                print("  ❌ No TEACHES relationships found!")
                print("\n💡 Run this to map course skills:")
                print("   docker exec -it vietcv_data_factory python -m src.data_factory.scripts.map_course_skills")
        
        driver.close()
        return True
        
    except Exception as e:
        print(f"❌ Error connecting to Neo4j: {e}")
        return False


def main():
    print("\n" + "="*80)
    print("Database Data Check")
    print("="*80 + "\n")
    
    es_ok = check_elasticsearch()
    neo4j_ok = check_neo4j()
    
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    
    if es_ok and neo4j_ok:
        print("✅ All databases have data!")
        print("\nIf API still returns empty results, check:")
        print("1. API can connect to databases (check logs)")
        print("2. Search query matches existing skills")
        print("3. Skills in Neo4j match skills in Elasticsearch")
    else:
        print("❌ Some databases are missing data")
        print("\nRun pipelines in order:")
        print("1. docker exec -it vietcv_data_factory python -m src.data_factory.scripts.index_skills")
        print("2. docker exec -it vietcv_data_factory python -m src.data_factory.scripts.map_course_skills")
        print("3. docker exec -it vietcv_data_factory python -m src.data_factory.scripts.build_graph")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
