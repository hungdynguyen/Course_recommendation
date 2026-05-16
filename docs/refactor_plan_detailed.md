# Refactor Plan: Baseline System (Single FastAPI + Course Engine)

## Tóm tắt

Refactor hệ thống hiện tại thành **2 phần rạch ròi**:

1. **FastAPI service**: Serve admin + user routes, internal course DOCX parsing, and recommendation retrieval
2. **Course Engine**: Build embeddings, manage Elasticsearch indices, and support cache/version operations

Không dùng: PostgreSQL (use MySQL instead), file-based embeddings (use Elasticsearch), 2-tier API split.

---

## 1. Current State Analysis

### 1.1 Cấu trúc hiện tại

```
src/
  service_api/
    api/v1/endpoints/
      admin_demo.py          # upload, queue, pipeline
      recommendations.py     # recommendation endpoint
      health.py
      jds.py
      skills.py
    services/
      admin_ingest_demo.py   # queue + worker logic
      course_recommendation.py
      skill_search.py
      gap_detection.py
    models/
      request.py
      response.py
    main.py
    config.py
    dependencies.py
  data_factory/
    # building graph, index skills
  shared/
    embeddings/
```

### 1.2 Vấn đề hiện tại

1. **Lẫn lộn giữa 2 phần**:
   - Admin upload/queue logic ở admin_ingest_demo.py nhưng chạy graph build (KG pipeline)
   - Recommendation logic ở recommendations.py nhưng chưa production-ready
   - Một service duy nhất FastAPI, không tách admin/user logic

2. **Thiếu proper database + vector storage**:
   - Không dùng MySQL để lưu metadata (course, version, batch)
   - Không dùng Elasticsearch cho vector search (dùng .npy files)
   - Không có S3/Minio để backup courses, snapshots, logs

3. **Không có scheduler + batch processing**:
   - Admin hiện tại manual trigger, không có Airflow DAG
   - Không có 8-task pipeline (parse → embed → validate → swap → notify)
   - Không có retry logic, monitoring, alerting

4. **Thiếu course docx parsing + embedding services**:
  - Không có parser riêng cho course docx (chỉ cần đọc docx, không cần OCR)
   - Không có course engine service (embeddings + indexing)
   - Không có cache versioning, rollback mechanism

5. **Dependencies KG cứ còn**:
   - Neo4jClient, MySQLClient legacy code vẫn ở codebase
   - skill_search.py, course_recommendation.py dùng KG
   - data_factory pipeline (build_graph.py) vẫn included

---

## 2. Target Architecture (Post-Refactor)

### 2.1 High-level single FastAPI service

```
┌─────────────────────────────────────────────────────────┐
│                   FASTAPI SERVICE                       │
│  2 route groups: /api/admin/*, /api/v1/recommend       │
│  (handles both course management + recommendations)     │
├─────────────────────────────────────────────────────────┤
│ - Admin routes: CRUD + pipeline trigger + versioning   │
│ - User routes: KNN search, cache info, health check    │
└─────────────────────────────────────────────────────────┘
           │
           └──────────────────┬────────────────────────────┐
                              │                            │
              ┌───────────────▼──────────────┐            │
              │   Shared Storage Layer       │            │
              ├──────────────────────────────┤            │
              │ - MySQL (metadata DB)        │            │
              │ - Elasticsearch (vectors)    │            │
              │ - S3/Minio (objects)         │            │
              │ - Redis (queue)              │            │
              └──────────────────────────────┘            │
                                                          │
              ┌───────────────────────────────────────────┴──┐
              │    Processing Layer (Airflow DAG)           │
              ├────────────────────────────────────────────┤
              │ - Course docx parser (internal, no OCR)     │
              │ - Course Engine (embed + index)             │
              │ - 8-task DAG (daily 01:00 UTC)              │
              └────────────────────────────────────────────┘
```

### 2.2 Folder structure (simplified - single FastAPI service)

```
src/
  service_api/                             # Single FastAPI service
    api/
      v1/
        endpoints/
          admin.py                         # POST/PUT/DELETE /api/admin/*
          recommendations.py               # POST /api/v1/recommend
          health.py                        # GET /health
          cache.py                         # GET /api/v1/cache/info
    services/
      elasticsearch_query.py               # ES KNN queries
      course_service.py                    # Course CRUD logic
      pipeline_service.py                  # Airflow DAG trigger
      version_service.py                   # Version management
      recommendation_ranker.py             # Ranking logic
    models/
      schemas.py
    main.py
    config.py

  course_engine/                           # Separate Course Engine
    api/
      routes.py
    services/
      embedding_builder.py
      es_indexer.py
      cache_manager.py
    main.py

  airflow/
    dags/
      daily_course_update_dag.py

  shared/
    models/
      course.py
    storage/
      mysql_client.py
      es_client.py
      s3_client.py
```

---

## 3. Components to Remove

### 3.1 Remove KG dependencies completely

Xóa hoàn toàn:
- `src/service_api/services/skill_search.py` (uses Neo4j KG)
- `src/service_api/services/course_recommendation.py` (uses Neo4j KG)
- `src/data_factory/scripts/build_graph.py` and all KG building logic
- Neo4jClient import from config
- Reference to Neo4j/graph endpoints

### 3.2 Replace outdated storage

Remove:
- `.npy` file-based embeddings (replace with Elasticsearch indices)
- `.jsonl` metadata files (replace with MySQL tables)
- `versions.json` file-based versioning (replace with MySQL course_versions table)

### 3.3 Remove Laravel (not needed initially)

Remove:
- Any separate Laravel/admin_api service folders
- Laravel dependencies from requirements
- Docker service for Laravel

Consolidate:
- All admin endpoints → FastAPI `/api/admin/*`
- All user endpoints → FastAPI `/api/v1/*`
- Single unified service (easier to deploy and maintain)

---

## 4. Components to Add

### 4.1 FastAPI Admin Endpoints

**File: `src/service_api/api/v1/endpoints/admin.py`**

```python
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import httpx

router = APIRouter(prefix="/api/admin", tags=["admin"])

@router.post("/courses/upload")
async def upload_courses(files: List[UploadFile] = File(...)):
    """Upload multiple course documents (docx/pdf)"""
    # 1. Validate file types
    # 2. Save to S3: s3://vietcv/uploads/{batch_id}/
    # 3. Create upload_batch in MySQL
    # 4. Trigger Airflow DAG
    # 5. Return batch_id + status
    pass

@router.get("/courses")
async def list_courses(skip: int = 0, limit: int = 100, version_id: str = None):
    """List all courses with optional version filter"""
    # Query MySQL courses table
    # Filter by version if provided
    # Return paginated results
    pass

@router.put("/courses/{course_id}")
async def update_course(course_id: str, updates: CourseUpdate):
    """Update course metadata"""
    # Update MySQL record
    # Create audit log entry
    # Return updated course
    pass

@router.delete("/courses/{course_id}")
async def delete_course(course_id: str):
    """Soft delete a course"""
    # Set deleted_at timestamp in MySQL
    # Don't delete from Elasticsearch (keep for history)
    pass

@router.post("/pipeline/trigger")
async def trigger_pipeline(manual: bool = False):
    """Manually trigger Airflow DAG"""
    # Call Airflow REST API: POST /api/v1/dags/daily_course_update/dagRuns
    # Return dag_run_id + status
    pass

@router.get("/pipeline/status/{dag_run_id}")
async def get_pipeline_status(dag_run_id: str):
    """Check Airflow DAG run status"""
    # Poll Airflow for task status
    # Return tasks + states
    pass

@router.get("/versions")
async def list_versions():
    """List all course cache versions"""
    # Query MySQL course_versions table
    # Return version list with metadata
    pass

@router.post("/versions/{version_id}/rollback")
async def rollback_version(version_id: str):
    """Rollback to previous cache version"""
    # Swap Elasticsearch index alias to old version
    # Update MySQL current_version record
    # Return success status
    pass
```

### 4.2 FastAPI User Endpoints

**File: `src/service_api/api/v1/endpoints/recommendations.py`**

```python
from fastapi import APIRouter
from elasticsearch import Elasticsearch

router = APIRouter(prefix="/api/v1", tags=["recommendations"])
es_client = Elasticsearch(hosts=["elasticsearch:9200"])

@router.post("/recommend")
async def recommend(request: RecommendRequest):
    """Get course recommendations for CV/JD pair"""
    # 1. Build combined query embedding
    # 2. KNN search on Elasticsearch (1024-dim dense_vector)
    # 3. Rank results by relevance score
    # 4. Return top_k courses with explanations
    # Latency target: p99 < 200ms
    pass

@router.get("/cache/info")
async def cache_info():
    """Get current cache version info"""
    # Query MySQL course_versions table
    # Return current_version + metadata
    pass
```

### 4.3 FastAPI Health Endpoint

**File: `src/service_api/api/v1/endpoints/health.py`**

```python
from fastapi import APIRouter

router = APIRouter(tags=["health"])

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}

@router.get("/health/ready")
async def readiness_check():
    """Readiness probe (all dependencies ready)"""
    # Check MySQL connection
    # Check Elasticsearch connection
    # Check S3 connectivity
    # Return 200 if all ok, 503 if not
    pass

@router.get("/health/live")
async def liveness_check():
    """Liveness probe (service is running)"""
    return {"status": "alive"}
```

### 4.4 FastAPI Service Layer

**File: `src/service_api/services/course_service.py`**

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

class CourseService:
    def __init__(self, db: Session):
        self.db = db
    
    def create_course(self, course_data: CourseCreate):
        # Insert into MySQL courses table
        # Return course_id
        pass
    
    def get_course(self, course_id: str):
        # Query MySQL by course_id
        pass
    
    def update_course(self, course_id: str, updates: CourseUpdate):
        # Update MySQL record
        # Create audit_log entry
        pass
    
    def delete_course(self, course_id: str):
        # Soft delete (set deleted_at)
        pass
    
    def list_courses(self, version_id: str = None, skip: int = 0, limit: int = 100):
        # Query MySQL with pagination
        # Filter by version if provided
        pass
```

**File: `src/service_api/services/pipeline_service.py`**

```python
import httpx

class PipelineService:
    def __init__(self, airflow_url: str):
        self.airflow_url = airflow_url
    
    def trigger_dag(self, dag_id: str = "daily_course_update"):
        # POST to Airflow REST API
        # Return dag_run_id
        pass
    
    def get_dag_status(self, dag_run_id: str):
        # GET Airflow dag run status
        # Return task statuses
        pass
```

**File: `src/service_api/services/elasticsearch_query.py`**

```python
from elasticsearch import Elasticsearch

class ElasticsearchQueryService:
    def __init__(self, es_client: Elasticsearch):
        self.es = es_client
    
    def knn_search(self, embedding: List[float], top_k: int = 10):
        # KNN search on dense_vector field
        # Return top_k courses with scores
        pass
    
    def swap_index(self, old_version: str, new_version: str):
        # Atomic: change alias from old to new index
        pass
```

### 4.5 Airflow DAG

**File: `src/airflow/dags/daily_course_update_dag.py`**

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

dag = DAG(
    'daily_course_update',
    default_args={
        'owner': 'fastapi_service',
        'retries': 2,
        'retry_delay': timedelta(minutes=5),
    },
    schedule_interval='0 1 * * *',  # 01:00 UTC daily
    start_date=datetime(2026, 4, 1),
    catchup=False,
)

def fetch_pending_courses():
    # Query MySQL for pending courses
    pass

def parse_course_docx():
  # Parse course docx internally with python-docx
    pass

def build_embeddings():
    # Call Course Engine: POST /build
    pass

def validate_quality():
    # Check quality metrics
    pass

def swap_cache_and_update_es():
    # Atomic index swap
    # Notify FastAPI to reload
    pass

def send_notification_success():
    # Send Slack/email notification
    pass

def archive_old_versions():
    # Compress old versions to S3
    pass

def handle_failure():
    # Error handling + notifications
    pass

task_fetch = PythonOperator(task_id='fetch', python_callable=fetch_pending_courses, dag=dag)
task_parse = PythonOperator(task_id='parse_course_docx', python_callable=parse_course_docx, dag=dag)
task_embed = PythonOperator(task_id='embed', python_callable=build_embeddings, dag=dag)
task_validate = PythonOperator(task_id='validate', python_callable=validate_quality, dag=dag)
task_swap = PythonOperator(task_id='swap', python_callable=swap_cache_and_update_es, dag=dag)
task_notify = PythonOperator(task_id='notify', python_callable=send_notification_success, dag=dag)
task_archive = PythonOperator(task_id='archive', python_callable=archive_old_versions, dag=dag)
task_error = PythonOperator(task_id='error_handler', python_callable=handle_failure, dag=dag)

# DAG flow
task_fetch >> task_parse >> task_embed >> task_validate >> task_swap >> [task_notify, task_archive]
```
        # Update symlink: current -> v_{version}
        # Notify recommendation service để reload
        pass
```

**File: `src/service_api/services/course_service.py`**

```python
class CourseMetadataService:
    def __init__(self, db):
        self.db = db  # PostgreSQL
    
    def create_course(self, course_data: Course):
        # Insert course
        pass
    
    def update_course(self, course_id: str, updates: Dict):
        # Update fields + increment version
        pass
    
    def delete_course(self, course_id: str):
        # Mark as deleted (soft delete)
        pass
    
    def list_courses(self, skip=0, limit=100):
        # List active courses
        pass
    
    def get_course_version(self, version: str):
        # Fetch courses included in this version
        pass
```

**File: `src/service_api/services/course_docx_parser.py`**

```python
class LLMEnrichService:
    def enrich_course(self, course_data: Dict) -> Dict:
        # Call LLM endpoint để:
        # 1. Parse docx -> structured format (nếu chưa parse)
        # 2. Extract skills từ content
        # 3. Generate embedding-friendly text
        pass
```

**File: `src/service_api/services/pipeline_service.py`**

```python
class AirflowTrigger:
    def trigger_daily_update(self):
        # Call Airflow REST API để trigger DAG
        pass
    
    def get_dag_status(self, dag_run_id: str):
        # Poll Airflow untuk task status
        pass
```

### 4.3 Shared models/storage

**File: `src/shared/models/course_model.py`**

```python
class Course(BaseModel):
    course_id: str
    title: str
    description: str
    skills: List[str]
    content_url: Optional[str]  # docx/pdf path
    version: str
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime] = None

class CourseVersion(BaseModel):
    version_id: str
    created_at: datetime
    course_count: int
    embeddings_path: str
    metadata_path: str
    checksum: str
    status: str  # "building", "ready", "archived"
```

**File: `src/shared/storage/cache_versioning.py`**

```python
class CacheVersionManager:
    def __init__(self, cache_root: Path):
        self.cache_root = cache_root
        self.versions_file = cache_root / "versions.json"
    
    def create_version(self, version_id: str) -> Dict:
        # Create versioned directory
        # Update versions.json
        pass
    
    def get_current_version(self) -> str:
        # Read symlink or config
        pass
    
    def set_current_version(self, version_id: str):
        # Atomic swap
        pass
    
    def list_versions(self):
        # All versions with metadata
        pass
    
    def rollback(self, version_id: str):
        # Switch back to old version
        pass
```

**File: `src/shared/storage/course_storage.py`**

```python
class CourseStorage:
    def save_uploaded_file(self, file: UploadFile, batch_id: str):
        # staging/{batch_id}/{filename}
        pass
    
    def move_to_catalog(self, batch_id: str, version_id: str):
        # staging/{batch_id} -> catalog/{version_id}
        pass
    
    def get_catalog_courses(self, version_id: str) -> List[Course]:
        # Read từ catalog/{version_id}/*.json
        pass
```



---

## 5. API Endpoints - Updated for Single FastAPI Service (Port 8000)

### 5.1 Recommendation API (User)

```
# Health check
GET /health
GET /health/ready
GET /health/live

# Recommendation
POST /api/v1/recommend
  input: {
    "cv_id": "cv_001",
    "jd_id": "jd_001",
    "top_k": 10
  }
  output: {
    "cv_id": "cv_001",
    "jd_id": "jd_001",
    "results": [
      {
        "rank": 1,
        "course_id": "course_001",
        "title": "Python Programming",
        "score": 0.92,
        "matched_skills": ["Python", "OOP"],
        "missing_skills": ["Web Scraping"]
      }
    ],
    "latency_ms": 45,
    "version": "v_20260430_001"
  }

# Cache info
GET /api/v1/cache/info
  output: {
    "current_version": "v_20260430_001",
    "course_count": 500,
    "last_updated": "2026-04-30T01:00:00Z",
    "embedding_model": "Qwen3-Embedding-0.6B",
    "es_health": "green"
  }
```

### 5.2 Admin API (FastAPI - port 8000)

```
# Courses CRUD
POST /api/admin/courses/upload
  input: multipart/form-data (multiple docx/pdf files)
  output: {
    "batch_id": "batch_001",
    "file_count": 5,
    "status": "queued"
  }

GET /api/admin/courses
  output: {
    "total": 500,
    "courses": [
      {
        "course_id": "course_001",
        "title": "Python Programming",
        "skills": ["Python", "OOP"],
        "version_id": "v_20260430_001",
        "created_at": "2026-04-15T00:00:00Z"
      }
    ]
  }

PUT /api/admin/courses/{course_id}
  input: {
    "title": "Advanced Python",
    "skills": ["Python", "OOP", "Design Patterns"],
    "description": "Updated course description"
  }
  output: {
    "course_id": "course_001",
    "updated_at": "2026-04-30T15:00:00Z"
  }

DELETE /api/admin/courses/{course_id}
  output: { "status": "deleted" }

# Pipeline management
POST /api/admin/pipeline/trigger
  input: { "batch_id": "optional" }
  output: {
    "dag_run_id": "run_20260430_001",
    "status": "running"
  }

GET /api/admin/pipeline/status/{dag_run_id}
  output: {
    "dag_run_id": "run_20260430_001",
    "status": "running",
    "progress": 60,
    "tasks": [
      { "task": "fetch_pending_courses", "status": "completed" },
      { "task": "parse_course_docx", "status": "running" },
      { "task": "build_embeddings", "status": "pending" },
      { "task": "validate_quality", "status": "pending" },
      { "task": "swap_cache_and_update_es", "status": "pending" }
    ]
  }

# Versioning & Cache Management
GET /api/admin/versions
  output: {
    "current_version": "v_20260430_001",
    "versions": [
      {
        "version_id": "v_20260430_001",
        "course_count": 520,
        "status": "ready",
        "created_at": "2026-04-30T01:00:00Z",
        "is_current": true
      },
      {
        "version_id": "v_20260429_001",
        "course_count": 500,
        "status": "ready",
        "created_at": "2026-04-29T01:00:00Z",
        "is_current": false
      }
    ]
  }

POST /api/admin/versions/{version_id}/rollback
  output: {
    "status": "ok",
    "previous_version": "v_20260430_001",
    "current_version": "v_20260429_001",
    "rolled_back_at": "2026-04-30T15:30:00Z"
  }
```

### 5.3 Health Endpoints

```
GET /health
  output: { "status": "ok" }

GET /health/ready
  output: { "status": "ready", "dependencies": {"mysql": "ok", "elasticsearch": "ok", "s3": "ok"} }

GET /health/live
  output: { "status": "alive" }
```

### 5.4 Course Document Parsing (internal, no separate service)

```
# Parse course docx during admin upload
POST /api/admin/courses/upload
  input: multipart/form-data (multiple docx files)
  output: {
    "batch_id": "batch_001",
    "file_count": 5,
    "status": "queued",
    "parsed_courses": 5
  }

# Internal parser module
course_docx_parser.py
  - python-docx for structured course docx
  - no OCR / no image recognition
  - extracts title, description, skills, credit hours, prerequisites
```

### 5.5 Course Engine (Python - port 8004)

```
# Build embeddings & index
POST /build
  input: {
    "version_id": "v_20260430_001",
    "courses": [...]
  }
  output: {
    "version_id": "v_20260430_001",
    "courses_processed": 520,
    "embedding_dimension": 1024,
    "es_index": "course_embeddings_v_20260430_001",
    "s3_backup": "s3://vietcv/cache/v_20260430_001/",
    "processing_time_ms": 45000
  }
```

---

## 6. Database Schema

### 6.1 MySQL Tables

**courses table**:
```sql
CREATE TABLE courses (
  course_id VARCHAR(255) PRIMARY KEY,
  title VARCHAR(500) NOT NULL,
  description TEXT,
  skills JSON,
  version_id VARCHAR(255),
  created_by VARCHAR(255),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  deleted_at TIMESTAMP NULL,
  FOREIGN KEY (version_id) REFERENCES course_versions(version_id),
  INDEX idx_version(version_id),
  INDEX idx_deleted(deleted_at)
);
```

**course_versions table**:
```sql
CREATE TABLE course_versions (
  version_id VARCHAR(255) PRIMARY KEY,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  course_count INT,
  status ENUM('building', 'ready', 'archived'),
  s3_backup_path VARCHAR(500),
  es_index VARCHAR(255),
  checksum VARCHAR(255)
);
```

**upload_batches table**:
```sql
CREATE TABLE upload_batches (
  batch_id VARCHAR(255) PRIMARY KEY,
  file_count INT,
  status ENUM('uploading', 'queued', 'processing', 'completed', 'failed'),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  completed_at TIMESTAMP NULL
);
```

**pipeline_runs table**:
```sql
CREATE TABLE pipeline_runs (
  dag_run_id VARCHAR(255) PRIMARY KEY,
  batch_id VARCHAR(255),
  status ENUM('running', 'success', 'failed'),
  started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  completed_at TIMESTAMP NULL,
  FOREIGN KEY (batch_id) REFERENCES upload_batches(batch_id)
);
```

**cv_profiles table**:
```sql
CREATE TABLE cv_profiles (
  cv_id VARCHAR(255) PRIMARY KEY,
  name VARCHAR(500),
  skills JSON,
  experience_years INT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**job_descriptions table**:
```sql
CREATE TABLE job_descriptions (
  jd_id VARCHAR(255) PRIMARY KEY,
  title VARCHAR(500),
  required_skills JSON,
  years_experience INT,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

**audit_logs table**:
```sql
CREATE TABLE audit_logs (
  id INT AUTO_INCREMENT PRIMARY KEY,
  action VARCHAR(50),
  resource_type VARCHAR(50),
  resource_id VARCHAR(255),
  details JSON,
  user_id VARCHAR(255),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  INDEX idx_resource(resource_type, resource_id),
  INDEX idx_created(created_at)
);
```

### 6.2 Elasticsearch Mapping

```json
{
  "mappings": {
    "properties": {
      "course_id": { "type": "keyword" },
      "title": { "type": "text" },
      "description": { "type": "text" },
      "skills": { "type": "keyword" },
      "embedding": {
        "type": "dense_vector",
        "dims": 1024,
        "index": true,
        "similarity": "cosine"
      },
      "version_id": { "type": "keyword" },
      "created_at": { "type": "date" }
    }
  }
}
```

---

## 7. Configuration Files

### 7.1 Environment Variables Structure

```env
# FastAPI Service (port 8000)
SERVICE_HOST=0.0.0.0
SERVICE_PORT=8000
SERVICE_NAME=vietcv_api

# Database
DB_HOST=mysql
DB_PORT=3306
DB_NAME=vietcv
DB_USER=vietcv_user
DB_PASSWORD=secure_password

# Elasticsearch
ES_HOST=elasticsearch
ES_PORT=9200
ES_INDEX=course_embeddings

# S3/Minio
S3_ENDPOINT=http://minio:9000
S3_BUCKET=vietcv
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin

# Embedding Model
EMBEDDING_MODEL_PATH=/app/models/qwen_embedding_finetuned
EMBEDDING_DEVICE=cuda
EMBEDDING_BATCH_SIZE=8

# Airflow
AIRFLOW_URL=http://airflow-webserver:8080/api/v1
AIRFLOW_USERNAME=airflow
AIRFLOW_PASSWORD=airflow

# Logging
LOG_LEVEL=INFO
```

### 7.2 `.env` cho Course Engine (Python - port 8004)

```env
SERVICE_PORT=8004
SERVICE_HOST=0.0.0.0

# Elasticsearch
ES_HOST=elasticsearch
ES_PORT=9200
ES_INDEX=course_embeddings

# Database
DB_HOST=mysql
DB_PORT=3306
DB_NAME=vietcv
DB_USER=vietcv_user
DB_PASSWORD=secure_password

# Embedding model
EMBEDDING_MODEL_PATH=/app/models/qwen_embedding_finetuned
EMBEDDING_DEVICE=cuda
EMBEDDING_BATCH_SIZE=8

# Storage
S3_ENDPOINT=http://minio:9000
S3_BUCKET=vietcv
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin

# Logging
LOG_LEVEL=INFO
```

### 7.3 Airflow configuration

```yaml
# airflow/config.yaml
airflow:
  core:
    dags_folder: /app/airflow/dags
    plugins_folder: /app/airflow/plugins
    base_log_folder: /app/logs/airflow
    executor: LocalExecutor
    database_uri: postgresql+psycopg2://airflow:airflow@airflow-postgres:5432/airflow_db
  
  scheduler:
    catchup_by_default: false
    dag_dir_list_interval: 300
    
  webserver:
    expose_config: true
    rbac: true
```

### 7.4 docker-compose.yml (services section - updated)

```yaml
services:
  mysql:
    image: mysql:8.3
    environment:
      MYSQL_ROOT_PASSWORD: root_password
      MYSQL_DATABASE: vietcv
      MYSQL_USER: vietcv_user
      MYSQL_PASSWORD: secure_password
    ports:
      - "3306:3306"
    volumes:
      - mysql_data:/var/lib/mysql

  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
    ports:
      - "9200:9200"
    volumes:
      - es_data:/usr/share/elasticsearch/data

  minio:
    image: minio/minio:latest
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - minio_data:/minio_data
    command: server /minio_data --console-address ":9001"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  airflow-postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: airflow_db
      POSTGRES_USER: airflow
      POSTGRES_PASSWORD: airflow
    volumes:
      - airflow_postgres_data:/var/lib/postgresql/data

  airflow-webserver:
    image: apache/airflow:2.8.0-python3.11
    environment:
      AIRFLOW__CORE__EXECUTOR: LocalExecutor
      AIRFLOW__CORE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@airflow-postgres:5432/airflow_db
    ports:
      - "8080:8080"
    volumes:
      - ./airflow/dags:/opt/airflow/dags
      - ./airflow/plugins:/opt/airflow/plugins
    depends_on:
      - airflow-postgres

  airflow-scheduler:
    image: apache/airflow:2.8.0-python3.11
    environment:
      AIRFLOW__CORE__EXECUTOR: LocalExecutor
      AIRFLOW__CORE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@airflow-postgres:5432/airflow_db
    volumes:
      - ./airflow/dags:/opt/airflow/dags
      - ./airflow/plugins:/opt/airflow/plugins
    depends_on:
      - airflow-postgres
    command: scheduler

  fastapi_service:
    build:
      context: .
      dockerfile: docker/Dockerfile.api
    environment:
      SERVICE_API_PORT: 8000
      SERVICE_HOST: 0.0.0.0
    ports:
      - "8000:8000"
    depends_on:
      - mysql
      - elasticsearch
      - minio
      - redis

  course_engine:
    build:
      context: ./src/course_engine
    environment:
      SERVICE_PORT: 8004
    ports:
      - "8004:8004"
    depends_on:
      - mysql
      - elasticsearch
      - minio

volumes:
  mysql_data:
  es_data:
  minio_data:
  airflow_postgres_data:
```

---

## 6. Database Schema (Metadata) - MySQL + Elasticsearch

### 6.1 MySQL tables

```sql
-- Courses
CREATE TABLE courses (
    id VARCHAR(50) PRIMARY KEY,
    title VARCHAR(255),
    description LONGTEXT,
    skills JSON,  -- Array of skill names
    content_url VARCHAR(500),
    version_id VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP NULL,
    INDEX idx_version_id (version_id),
    INDEX idx_deleted_at (deleted_at)
);

-- Course versions
CREATE TABLE course_versions (
    version_id VARCHAR(50) PRIMARY KEY,
    course_count INT,
    embeddings_path VARCHAR(500),
    metadata_path VARCHAR(500),
    checksum VARCHAR(64),
    status VARCHAR(20),  -- building, ready, archived
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    archived_at TIMESTAMP NULL,
    INDEX idx_status (status)
);

-- Upload batches
CREATE TABLE upload_batches (
    batch_id VARCHAR(50) PRIMARY KEY,
    file_count INT,
    total_size_bytes BIGINT,
    status VARCHAR(20),  -- pending, processing, completed, failed
    error_message LONGTEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP NULL,
    INDEX idx_status (status)
);

-- Pipeline runs
CREATE TABLE pipeline_runs (
    run_id VARCHAR(50) PRIMARY KEY,
    batch_id VARCHAR(50),
    status VARCHAR(20),  -- running, completed, failed
    version_id VARCHAR(50),
    progress_percent INT,
    error_log LONGTEXT,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP NULL,
    FOREIGN KEY (batch_id) REFERENCES upload_batches(batch_id),
    FOREIGN KEY (version_id) REFERENCES course_versions(version_id),
    INDEX idx_status (status),
    INDEX idx_version_id (version_id)
);

-- CV Profiles
CREATE TABLE cv_profiles (
    cv_id VARCHAR(50) PRIMARY KEY,
    filename VARCHAR(255),
    parsed_json JSON,
    skills JSON,
    experience_years INT,
    s3_path VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_created_at (created_at)
);

-- Job Descriptions
CREATE TABLE job_descriptions (
    jd_id VARCHAR(50) PRIMARY KEY,
    title VARCHAR(255),
    parsed_json JSON,
    required_skills JSON,
    s3_path VARCHAR(500),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_created_at (created_at)
);

-- Audit logs
CREATE TABLE audit_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    action VARCHAR(50),  -- create, update, delete, upload, process
    resource_type VARCHAR(50),  -- course, batch, version
    resource_id VARCHAR(50),
    user_id VARCHAR(50),
    details JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_resource (resource_type, resource_id),
    INDEX idx_created_at (created_at)
);
```

### 6.2 Elasticsearch indices

```json
{
  "index_name": "course_embeddings",
  "settings": {
    "number_of_shards": 2,
    "number_of_replicas": 1,
    "index.knn": true
  },
  "mappings": {
    "properties": {
      "course_id": { "type": "keyword" },
      "title": { "type": "text" },
      "description": { "type": "text" },
      "skills": { "type": "keyword" },
      "embedding": {
        "type": "dense_vector",
        "dims": 1024,
        "index": true,
        "similarity": "cosine"
      },
      "version_id": { "type": "keyword" },
      "created_at": { "type": "date" },
      "updated_at": { "type": "date" }
    }
  }
}
```

### 6.3 S3/Minio bucket structure

```
vietcv/
  uploads/
    {batch_id}/
      course_1.docx
      course_2.docx
  cache/
    {version_id}/
      courses.json
      embeddings.npy
      metadata.jsonl
  snapshots/
    {version_id}_snapshot.tar.gz
  logs/
    {date}/
      airflow_logs.txt
```

---

## 7. Configuration - Updated for MySQL, ES, S3, single FastAPI API

### 7.1 `.env` cho Recommendation API (service_api)

```env
# Database (MySQL - read-only for user routes)
DB_HOST=mysql
DB_PORT=3306
DB_NAME=vietcv
DB_USER=vietcv_user
DB_PASSWORD=secure_password

# Elasticsearch (for KNN search)
ES_HOST=elasticsearch
ES_PORT=9200
ES_INDEX=course_embeddings
ES_TIMEOUT=10

# Embedding model
EMBEDDING_MODEL_NAME=Qwen/Qwen3-Embedding-0.6B
EMBEDDING_MODEL_PATH=/app/models/qwen_embedding_finetuned
EMBEDDING_DEVICE=cuda
EMBEDDING_BATCH_SIZE=8

# API
SERVICE_API_HOST=0.0.0.0
SERVICE_API_PORT=8002
SERVICE_API_WORKERS=4

# S3/Minio (optional backup)
S3_ENDPOINT=http://minio:9000
S3_BUCKET=vietcv
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin

# Observability
LOG_LEVEL=INFO
METRICS_PORT=8003
```

### 7.2 `.env` cho FastAPI Service (port 8000)

```env
# FastAPI Service (unified admin + user)
SERVICE_HOST=0.0.0.0
SERVICE_PORT=8000
SERVICE_NAME=vietcv_api

# Database (MySQL)
DB_HOST=mysql
DB_PORT=3306
DB_NAME=vietcv
DB_USER=vietcv_user
DB_PASSWORD=secure_password

# Elasticsearch
ES_HOST=elasticsearch
ES_PORT=9200
ES_INDEX=course_embeddings

# S3/Minio
S3_ENDPOINT=http://minio:9000
S3_BUCKET=vietcv
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin

# Embedding model
EMBEDDING_MODEL_PATH=/app/models/qwen_embedding_finetuned
EMBEDDING_DEVICE=cuda
EMBEDDING_BATCH_SIZE=8

# Airflow (to trigger DAGs)
AIRFLOW_URL=http://airflow-webserver:8080/api/v1
AIRFLOW_USERNAME=airflow
AIRFLOW_PASSWORD=airflow

# Logging
LOG_LEVEL=INFO
```

### 7.3 `.env` cho Course Engine (Python - port 8004)

```env
# Service
SERVICE_HOST=0.0.0.0
SERVICE_PORT=8004
SERVICE_NAME=course_engine

# Database
DB_HOST=mysql
DB_PORT=3306
DB_NAME=vietcv
DB_USER=vietcv_user
DB_PASSWORD=secure_password

# S3
S3_ENDPOINT=http://minio:9000
S3_BUCKET=vietcv
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin

# Logging
LOG_LEVEL=INFO
```

### 7.4 `.env` cho Course Engine (Python - port 8004)

```env
# Service
SERVICE_HOST=0.0.0.0
SERVICE_PORT=8004
SERVICE_NAME=course_engine

# Database
DB_HOST=mysql
DB_PORT=3306
DB_NAME=vietcv
DB_USER=vietcv_user
DB_PASSWORD=secure_password

# Elasticsearch
ES_HOST=elasticsearch
ES_PORT=9200

# Embedding model
EMBEDDING_MODEL_PATH=/app/models/qwen_embedding_finetuned
EMBEDDING_DEVICE=cuda
EMBEDDING_BATCH_SIZE=8

# S3
S3_ENDPOINT=http://minio:9000
S3_BUCKET=vietcv
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin

# Logging
LOG_LEVEL=INFO
```

### 7.5 Airflow configuration

```yaml
# airflow/config.yaml
airflow:
  core:
    dags_folder: /app/airflow/dags
    plugins_folder: /app/airflow/plugins
    base_log_folder: /app/logs/airflow
    executor: LocalExecutor
    database_uri: mysql+pymysql://vietcv_user:secure_password@mysql:3306/airflow_db
  
  scheduler:
    catchup_by_default: false
    dag_dir_list_interval: 300
    
  webserver:
    expose_config: true
    rbac: true
    
  admin:
    default_user_password: airflow
```

### 7.6 docker-compose.yml (services section - updated)

```yaml
services:
  mysql:
    image: mysql:8.3
    environment:
      MYSQL_ROOT_PASSWORD: root_password
      MYSQL_DATABASE: vietcv
      MYSQL_USER: vietcv_user
      MYSQL_PASSWORD: secure_password
    ports:
      - "3306:3306"
    volumes:
      - mysql_data:/var/lib/mysql
      - ./migrations/schema.sql:/docker-entrypoint-initdb.d/schema.sql

  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.11.0
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
    ports:
      - "9200:9200"
    volumes:
      - es_data:/usr/share/elasticsearch/data

  minio:
    image: minio/minio:latest
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - minio_data:/minio_data
    command: server /minio_data --console-address ":9001"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  airflow-postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: airflow_db
      POSTGRES_USER: airflow
      POSTGRES_PASSWORD: airflow
    volumes:
      - airflow_postgres_data:/var/lib/postgresql/data

  airflow-webserver:
    image: apache/airflow:2.8.0-python3.11
    environment:
      AIRFLOW__CORE__EXECUTOR: LocalExecutor
      AIRFLOW__CORE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@airflow-postgres:5432/airflow_db
    ports:
      - "8080:8080"
    volumes:
      - ./airflow/dags:/opt/airflow/dags
      - ./airflow/plugins:/opt/airflow/plugins
    depends_on:
      - airflow-postgres

  airflow-scheduler:
    image: apache/airflow:2.8.0-python3.11
    environment:
      AIRFLOW__CORE__EXECUTOR: LocalExecutor
      AIRFLOW__CORE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@airflow-postgres:5432/airflow_db
    volumes:
      - ./airflow/dags:/opt/airflow/dags
      - ./airflow/plugins:/opt/airflow/plugins
    depends_on:
      - airflow-postgres
    command: scheduler

  fastapi_service:
    build:
      context: .
      dockerfile: docker/Dockerfile.api
    environment:
      SERVICE_API_PORT: 8000
      SERVICE_HOST: 0.0.0.0
    ports:
      - "8000:8000"
    depends_on:
      - mysql
      - elasticsearch
      - minio
      - redis

  course_engine:
    build:
      context: ./src/course_engine
    environment:
      SERVICE_PORT: 8004
    ports:
      - "8004:8004"
    depends_on:
      - mysql
      - elasticsearch
      - minio

volumes:
  mysql_data:
  es_data:
  minio_data:
  airflow_postgres_data:
```

---

## 8. Migration Steps - Simplified (Single FastAPI Service)

### Phase 1: Setup & Foundation (Week 1)

1. **Create folder structure**
   - `mkdir -p src/service_api/{api,services,models}`
   - `mkdir -p src/course_engine/{api,services}`
   - `mkdir -p src/airflow/{dags,tasks,plugins}`
   - `mkdir -p src/shared/{models,storage}`

2. **Setup MySQL database**
   - Run migrations: `mysql vietcv < migrations/schema.sql`
   - Create 8 tables

3. **Setup Elasticsearch**
   - Create index: `course_embeddings` with dense_vector mapping (1024 dims)
   - Enable KNN search
   - Setup aliases for versioning

4. **Setup Minio S3 + Redis + Airflow**
   - Create buckets: `uploads/`, `cache/`, `snapshots/`
   - Start Redis container
   - Initialize Airflow database
   - Create Airflow connections (MySQL, ES, S3)

### Phase 2: Build FastAPI Service (Week 2)

### Phase 2: Build FastAPI Service (Week 2)

1. **Create FastAPI app structure**
   - `/api/v1/endpoints/admin.py` (CRUD + pipeline)
   - `/api/v1/endpoints/recommendations.py` (recommend)
   - `/api/v1/endpoints/cache.py` (cache info)
   - `/api/v1/endpoints/health.py` (health check)

2. **Implement services**
   - `course_service.py` (CRUD logic)
   - `pipeline_service.py` (Airflow trigger)
   - `version_service.py` (versioning)
   - `elasticsearch_query.py` (KNN search)

3. **Create models**
   - Pydantic schemas for requests/responses
   - SQLAlchemy ORM models for MySQL

4. **Test FastAPI endpoints**
   - Unit tests for services
   - Integration tests with MySQL/Elasticsearch
   - Load testing (100 QPS target)

### Phase 3: Build Course Engine (Python - port 8004) (Week 2-3)

1. **Create FastAPI app**
   - POST `/build` endpoint
   - Implement EmbeddingBuilder
   - Implement EsIndexer
   - Implement CacheManager

2. **Implement services**
   - Batch encoding to embeddings (Qwen model)
   - Bulk indexing to Elasticsearch
   - S3 backup of indices
   - MySQL versioning

3. **Test Course Engine**
   - Build embeddings for 50 courses
   - Verify index created in Elasticsearch
   - Verify backup in S3
   - Verify MySQL version record

### Phase 4: Create Airflow DAG (8 tasks) (Week 3-4)

1. **Create DAG file**
   - `daily_course_update_dag.py` with 8 tasks
   - Schedule: `0 1 * * *` (01:00 UTC daily)

2. **Implement tasks**
   - fetch_pending_courses (query MySQL)
  - parse_course_docx (internal parser inside FastAPI)
   - build_embeddings (call Course Engine)
   - validate_quality (check metrics)
   - swap_cache_and_update_es (atomic swap)
   - send_notification_success (Slack/email)
   - archive_old_versions (cleanup)
   - handle_failure (error handling)

3. **Setup monitoring**
   - Airflow web UI monitoring
   - Alert on failure
   - SLA tracking

4. **Test DAG**
   - Manual trigger
   - Verify all tasks succeed
   - Verify cache updated
   - Verify recommendation API works

### Phase 5: Docker & Deployment (Week 4)

1. **Update docker-compose.yml**
  - 5 services: MySQL, Elasticsearch, Minio, Redis, Airflow + single FastAPI app
  - Single fastapi_service on port 8000 with internal course docx parser

2. **Create Dockerfiles**
  - Dockerfile.api (FastAPI - port 8000)
   - Dockerfile.course_engine (Course Engine - port 8004)

3. **Test end-to-end**
   - Upload courses via `/api/admin/courses/upload`
   - Monitor DAG in Airflow UI
   - Query recommendations via `/api/v1/recommend`
   - Verify all data flows correctly

### Phase 6: Testing & Documentation (Week 4)

1. **Unit + integration tests**
   - Course CRUD operations
   - KNN search accuracy
   - Pipeline orchestration

2. **Load testing**
   - 100 QPS on recommendations endpoint
   - Measure p99 latency (target: < 200ms)
   - Test embedding batch processing

3. **Documentation**
   - API documentation (OpenAPI/Swagger)
   - Architecture diagrams
   - Deployment guide
   - Operational runbooks
   - Troubleshooting guide

---

## 9. Backward Compatibility & Migration

### 9.1 Breaking changes

1. API endpoints change:
   - Old: `/api/v1/admin/*` (mixed with recommendations)
   - New: `/api/admin/*` (admin functions) + `/api/v1/*` (user/recommendations)

2. Single FastAPI service (port 8000):
   - Old: 2-tier (Admin 8001, User 8002)
   - New: Unified (FastAPI 8000)

3. Storage model:

### 9.2 Migration path for existing clients

1. **Phase 1**: Deploy both old and new systems (parallel)
   - Old system continues serving recommendations
   - New system accepts course uploads and builds cache

2. **Phase 2**: Gradual traffic migration (1-2 weeks)
   - Route 10% of recommendation requests to new FastAPI port 8000
   - Monitor metrics, adjust if needed
   - Increase to 50%, then 100%

3. **Phase 3**: Cutover (after 1 month stability)
   - Deprecate old endpoints
   - Old system enters maintenance mode
   - Send client migration notifications

4. **Phase 4**: Cleanup (after 6 weeks)
   - Remove old code from repository
   - Archive old data
   - Complete deprecation

---

## 10. Rollback Strategy

### 10.1 Deployment rollback

**FastAPI Service Rollback** (port 8000):
- Just restart previous Docker image
- All endpoints (admin + user) will work immediately
- Recommendation queries will work immediately (read-only on Elasticsearch)

**Course Cache Rollback**:
- Use MySQL course_versions table
- `POST /api/admin/versions/{version_id}/rollback`
- Atomic: Elasticsearch index alias swap back to previous version

### 10.2 Data recovery

**Courses** (if accidental deletion):
- MySQL soft delete (deleted_at timestamp)
- Restore: `UPDATE courses SET deleted_at = NULL WHERE course_id = '...'`

**Embeddings** (if ES corrupted):
- S3 backup: `s3://vietcv/cache/{version_id}/`
- Restore: Re-index from S3 backup
- Or: Trigger DAG to rebuild from courses

**Versions**:
- Last 3 versions kept in MySQL + ES
- Older versions archived (compressed ES indices)
- Can restore any recent version within 30 days

---

## 11. Monitoring & Observability

### 11.1 Metrics to track

**Recommendation API**:
- Request count, latency (p50, p95, p99)
- Cache hit/miss rate
- Model load time

**Course Manager**:
- Upload count, file size distribution
- Pipeline run duration per stage
- Cache build success rate
- DAG execution status

### 11.2 Logging

- Structured logs (JSON) cho debugging
- Separate log streams: API, manager, airflow
- ELK atau CloudWatch

### 11.3 Alerts

- DAG failed 2x
- Recommendation API latency > 500ms
- Cache build failed
- Disk space warning

---

## 12. Testing Strategy

### 12.1 Unit tests

**Admin API (FastAPI)**:
- CRUD operations (create, read, update, delete courses)
- Pipeline triggering (Airflow DAG)
- Version management

**User API (FastAPI)**:
- Elasticsearch KNN search
- Recommendation ranking logic
- Input validation
- Cache queries

**Services**:
- course_docx_parser: structured course docx parsing
- EmbeddingBuilder: batch encoding consistency
- EsIndexer: bulk indexing correctness
- CacheManager: version management

### 12.2 Integration tests

- Admin upload → MySQL insert → Airflow trigger
- Airflow fetch → course docx parse → Course Engine embed → ES index
- ES KNN search → User API recommend endpoint
- Version rollback → old index becomes active

### 12.3 Performance tests

**Latency targets**:
- FastAPI `/api/v1/recommend`: p99 < 200ms (target: p99 < 150ms)
- FastAPI `/api/admin/courses/upload`: < 5s for 10 files
- Course docx parse: < 5s per 10-file upload

**Throughput targets**:
- Recommendation API: QPS >= 100
- Admin API: QPS >= 10
- Airflow DAG: Complete in < 15 minutes for 500 courses

### 12.4 Load tests

- Simulate 100 concurrent recommendation requests
- Simulate 10 concurrent uploads
- DAG running with 5000 courses

### 12.5 Chaos tests

- Stop Elasticsearch → graceful degradation
- Stop MySQL → User API reads from cache, Admin API fails
- Stop course docx parser → upload fails fast, DAG retries the upload batch
- Network latency → retry logic engages

---

## 13. Timeline & Resources - Simplified (Single FastAPI Service)

| Phase | Focus | Effort | Timeline |
|-------|-------|--------|----------|
| 1. Foundation | DB schema, ES, S3, Redis, Airflow setup | 3 days | Week 1 |
| 2. FastAPI Service | CRUD + recommendations + course docx parsing + health endpoints | 6 days | Week 2-3 |
| 3. Course Engine | Embedding building, ES indexing, S3 backup | 4 days | Week 2-3 |
| 4. Airflow DAG | 8-task pipeline with monitoring | 3 days | Week 3-4 |
| 5. Docker & Deployment | docker-compose with 5 services, single FastAPI | 2 days | Week 4 |
| 6. Testing & Docs | E2E tests, monitoring, documentation | 3 days | Week 4-5 |
| **Total** | **Full production system** | **21 days** | **~4.2 weeks** |

**Resources**: 2-3 backend engineers (1 for FastAPI + course docx parsing, 1 for Course Engine, 1 for Airflow + DevOps)

---

## 14. Checklist

### Pre-migration
- [ ] Backup current database
- [ ] Backup model files
- [ ] Document current behaviors (API contracts)

### Post-migration
- [ ] All unit tests passing
- [ ] Integration tests passing
- [ ] Performance benchmarks met
- [ ] Monitoring/alerting setup
- [ ] Runbook written
- [ ] Team training completed
- [ ] Rollback procedure tested

---

## 15. Deliverables

1. **Code**
   - New folder structure với all services
   - Remove KG dependencies
   - Tests

2. **Documentation**
   - API spec (OpenAPI/Swagger)
   - Runbook (ops guide)
   - Architecture ADR (decision records)

3. **Docker & Infra**
   - docker-compose.yml
   - Dockerfile(s)
   - Kubernetes manifests (optional)

4. **Monitoring**
   - Prometheus config
   - Grafana dashboards
   - Alert rules

---

## Kết luận

Refactor này sẽ tạo ra hệ thống rõ ràng:
- **Recommendation API**: Pure inference, stateless, fast
- **Course Manager**: Handles ingest, versioning, scheduling
- **Airflow**: Orchestrates daily updates
- **Shared layer**: Model, storage, versioning logic

Khi xong, hệ thống sẽ production-ready, scalable, và dễ maintain.
