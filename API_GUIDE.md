# VietCV Course Recommendation API

FastAPI service for course recommendations based on skill gap analysis using GraphRAG.

## Quick Start

### 1. Start API Server

```bash
# In Docker
docker exec -it vietcv_data_factory python run_api.py

# Or with uvicorn directly
docker exec -it vietcv_data_factory uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Access API

- **API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## API Endpoints

### 1. Search Skills
```bash
curl -X POST "http://localhost:8000/api/v1/skills/search?query=python&limit=5"
```

Response:
```json
[
  {
    "skill_id": "http://data.europa.eu/esco/skill/...",
    "label": "Python programming",
    "description": "...",
    "skill_type": "skill",
    "score": 8.5
  }
]
```

### 2. Recommend Courses
```bash
curl -X POST "http://localhost:8000/api/v1/courses/recommend?max_courses=5" \
  -H "Content-Type: application/json" \
  -d '{
    "skill_names": ["python", "data analysis", "machine learning"]
  }'
```

Response:
```json
{
  "requested_skills": [
    {"skill_id": "...", "label": "Python", "description": "..."}
  ],
  "recommended_courses": [
    {
      "course_id": "CS101",
      "course_title": "Introduction to Python",
      "category": "Programming",
      "taught_skills": [...],
      "required_skills": [],
      "similarity_score": 0.85,
      "prerequisite_level": 0
    }
  ],
  "learning_path": ["CS101", "DS201", "ML301"]
}
```

### 3. Get Courses by Skill
```bash
curl "http://localhost:8000/api/v1/courses/by-skill/{skill_id}?limit=10"
```

### 4. Get Course Details
```bash
curl "http://localhost:8000/api/v1/courses/BHKT1102"
```

## Architecture

### GraphRAG Components:
- **Neo4j**: Graph database storing skills, courses, TEACHES, REQUIRES relationships
- **Elasticsearch**: Vector search for skill semantic matching
- **FastAPI**: REST API serving recommendations

### Recommendation Flow:
1. User provides target skills (names or IDs)
2. System searches ESCO taxonomy for matching skills
3. Neo4j graph query finds courses teaching those skills
4. Prerequisite analysis orders courses into learning path
5. Returns ranked, ordered course recommendations

## Environment Variables

Ensure `.env` file has:
```env
NEO4J_URI=bolt://neo4j:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password123
ELASTICSEARCH_HOST=http://elasticsearch:9200
```

## Testing

```bash
# Health check
curl http://localhost:8000/health

# Search skill
curl -X POST "http://localhost:8000/api/v1/skills/search?query=programming"

# Get recommendations
curl -X POST "http://localhost:8000/api/v1/courses/recommend" \
  -H "Content-Type: application/json" \
  -d '{"skill_names": ["programming"]}'
```
