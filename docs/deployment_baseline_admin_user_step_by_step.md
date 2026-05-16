# Baseline-Only Deployment (Admin + User) - Step by Step

## 1. Scope đúng theo yêu cầu

Tài liệu này bám đúng luồng bạn muốn:

1. **Không dùng Knowledge Graph**.
2. **Không dùng Neo4j/MySQL cho recommendation runtime**.
3. **Chỉ deploy baseline embedding retrieval** (giống thí nghiệm baseline bạn đã chạy).
4. Hệ thống có 2 phần:
   - **Admin**: upload/drag-drop course -> chạy batch pipeline theo lịch.
   - **User**: chọn cặp CV/JD đã parse -> nhận top-k recommendation.
5. Giả sử đã có **LLM endpoint** có thể gọi để parse/enrich text khi cần.

## 2. Kiến trúc mục tiêu (baseline-only, production-ready)

### Frontend Layer
1. **Admin Frontend (NextJS)**
   - Upload courses (drag-drop)
   - View job queue + status
   - Configure schedule
   - Monitor logs

2. **User Frontend (NextJS)**
   - Select CV/JD pair
   - Trigger recommendation
   - View results + explanations

### API Layer (single FastAPI)
1. **Unified FastAPI service**
  - `POST /api/admin/courses/upload`
  - `GET /api/admin/courses`
  - `POST /api/admin/pipeline/trigger`
  - `GET /api/admin/pipeline/status/{dag_run_id}`
  - `GET /api/admin/versions`
  - `POST /api/admin/versions/{version_id}/rollback`
  - `POST /api/v1/recommend` (cv_id, jd_id, top_k)
  - `GET /api/v1/cache/info`
  - `GET /health`

### Storage Layer
1. **MySQL Metadata DB**
   - `courses` table (course_id, title, description, skills, version_id)
   - `course_versions` table (version_id, course_count, status, created_at)
   - `upload_batches` table (batch_id, file_count, status)
   - `pipeline_runs` table (run_id, batch_id, status, progress)
   - `audit_logs` table (action, resource_id, user_id, timestamp)

2. **Elasticsearch** (Vector DB)
   - Index: `course_embeddings`
   - Mapping: `{course_id, title, embedding: dense_vector[1024], skills[]}`
   - Used for: cosine similarity search via `_search` with `knn` query

3. **S3 / Minio** (Object Storage)
   - `uploads/{batch_id}/*.docx` - uploaded course files
   - `cache/v_{version_id}/embeddings_backup.json` - snapshot
   - `logs/pipeline_runs/{run_id}/` - pipeline logs

4. **Redis** (Optional, for Airflow/queue)
   - Queue: pending jobs
   - Cache: session, temp results

### Processing Layer
1. **Course docx parsing**
    - Input: course docx files
    - Process: structured extraction from docx tables/text
    - Output: structured JSON
    - Tech: `python-docx`

2. **Course Engine**
   - Embedding builder: encode courses → vector
   - Cache manager: store in ES + MySQL + backup to S3
   - Version controller: atomic swap logic

3. **Airflow Scheduler**
   - Daily 01:00 UTC trigger
   - DAG: `daily_course_update`
   - Tasks (8 steps - see section below)
   - Retry: 2x with 5min delay
   - Monitoring: Airflow UI + logs to S3

## 3. Data contracts

### 3.1 Input đã có sẵn

- CV parsed data: `{cv_id, user_id, skills[], requirements_text, created_at}`
- JD parsed data: `{jd_id, job_title, skills[], description, min_years_exp, created_at}`
- Lưu trong MySQL `cv_profiles` và `job_descriptions` tables

### 3.2 Course upload format (from Admin)

```json
{
  "course_id": "CNTT1234",
  "title": "Machine Learning Fundamentals",
  "description": "Learn ML algorithms and applications",
  "skills_outcomes": [
    {"skill_name": "Python", "proficiency_level": "intermediate"},
    {"skill_name": "Statistics", "proficiency_level": "beginner"}
  ],
  "duration_hours": 40,
  "level": "beginner",
  "prerequisites": ["CNTT1001"]
}
```

### 3.3 Course docx parsing output (docx → JSON)

```json
{
  "batch_id": "batch_20260430_001",
  "parsed_courses": [
    {
      "course_id": "extracted_from_docx",
      "title": "...",
      "description": "...",
      "skills_outcomes": [...],
    "parse_confidence": 0.95
    }
  ],
  "total_parsed": 50,
  "extraction_time_ms": 12000
}
```

### 3.4 Recommendation output

```json
{
  "cv_id": "cv_001",
  "jd_id": "jd_001",
  "top_k": 10,
  "results": [
    {
      "rank": 1,
      "course_id": "CNTT1234",
      "title": "Machine Learning",
      "score": 0.9123,
      "matched_skills": ["Python", "Statistics"],
      "missing_skills": ["TensorFlow"]
    }
  ],
  "latency_ms": 45
}

## 4. Những gì cần setup

### 4.1 Docker services (docker-compose.yml)

```yaml
services:
  # APIs
  fastapi_service:    # FastAPI, port 8000

  # Storage
  mysql:              # port 3306
  elasticsearch:      # port 9200
  redis:              # port 6379 (optional, for Airflow)
  minio:              # port 9000 (S3-compatible)

  # Processing
    course_engine:      # Python embedding builder, port 8004

  # Orchestration
  airflow_webserver:  # port 8080
  airflow_scheduler:  # background
  airflow_worker:     # background
```

### 4.2 Environment setup

```env
# Database
MYSQL_HOST=mysql
MYSQL_PORT=3306
MYSQL_DATABASE=vietcv
MYSQL_USER=vietcv
MYSQL_PASSWORD=xxx

# Elasticsearch
ES_HOST=http://elasticsearch:9200
ES_INDEX=course_embeddings

# Embedding model
EMBEDDING_MODEL_NAME=Qwen/Qwen3-Embedding-0.6B
EMBEDDING_MODEL_PATH=/app/models/qwen_embedding_finetuned
EMBEDDING_DEVICE=cuda
EMBEDDING_BATCH_SIZE=8

# S3/Minio
S3_ENDPOINT=http://minio:9000
S3_ACCESS_KEY=minioadmin
S3_SECRET_KEY=minioadmin
S3_BUCKET=vietcv

# Redis (optional)
REDIS_HOST=redis
REDIS_PORT=6379

# Airflow
AIRFLOW_HOME=/app/airflow
AIRFLOW__CORE__DAGS_FOLDER=/app/airflow/dags
AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql://airflow:airflow@postgres:5432/airflow
```

## 5. Những gì cần code (single FastAPI API architecture)

### 5.1 Admin routes (FastAPI) endpoints

```python
# app/api/v1/endpoints/admin.py
# POST /api/admin/courses/upload
# GET /api/admin/courses
# PUT /api/admin/courses/{course_id}
# DELETE /api/admin/courses/{course_id}
# POST /api/admin/pipeline/trigger
# GET /api/admin/pipeline/status/{dag_run_id}
# GET /api/admin/versions
# POST /api/admin/versions/{version_id}/rollback
```

**Services to implement (FastAPI + support services):**
- `CourseService` - CRUD with MySQL
- `PipelineService` - call Airflow REST API to trigger DAG
- `VersionService` - manage ES indices + S3 snapshots
- `S3Service` - upload/retrieve from Minio
- `CourseDocxParser` - parse structured course docx during upload

### 5.2 User routes (FastAPI) endpoints

```python
# app/api/v1/endpoints/recommendations.py
# POST /api/v1/recommend
# GET /api/v1/cache/info
# GET /health
```

**Services to implement (FastAPI):**
- `RecommendationService`:
  - Input: cv_id, jd_id, top_k
  - Load CV/JD from MySQL
  - Build query (missing skills)
  - Query ES with KNN
  - Return top-k courses
- `EsQueryService` - ES client wrapper
- `CacheService` - retrieve metadata from MySQL

### 5.3 MySQL schema

```sql
CREATE TABLE courses (
    id VARCHAR(50) PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    description LONGTEXT,
    skills JSON,
    duration_hours INT,
    level ENUM('beginner','intermediate','advanced'),
    version_id VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    deleted_at TIMESTAMP NULL,
    FOREIGN KEY (version_id) REFERENCES course_versions(id),
    INDEX idx_version (version_id),
    INDEX idx_deleted (deleted_at)
);

CREATE TABLE course_versions (
    id VARCHAR(50) PRIMARY KEY,
    course_count INT,
    es_index_name VARCHAR(100),
    snapshot_s3_path VARCHAR(500),
    status ENUM('building','ready','archived'),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    archived_at TIMESTAMP NULL
);

CREATE TABLE upload_batches (
    id VARCHAR(50) PRIMARY KEY,
    file_count INT,
    file_names JSON,
    s3_path VARCHAR(500),
    status ENUM('pending','processing','completed','failed'),
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP NULL,
    INDEX idx_status (status)
);

CREATE TABLE pipeline_runs (
    id VARCHAR(50) PRIMARY KEY,
    batch_id VARCHAR(50),
    status ENUM('running','completed','failed'),
    version_id VARCHAR(50),
    progress_percent INT,
    airflow_run_id VARCHAR(100),
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP NULL,
    FOREIGN KEY (batch_id) REFERENCES upload_batches(id),
    FOREIGN KEY (version_id) REFERENCES course_versions(id),
    INDEX idx_batch (batch_id),
    INDEX idx_status (status)
);

CREATE TABLE cv_profiles (
    id VARCHAR(50) PRIMARY KEY,
    user_id VARCHAR(50),
    skills JSON,
    requirements_text LONGTEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE job_descriptions (
    id VARCHAR(50) PRIMARY KEY,
    job_title VARCHAR(255),
    skills JSON,
    description LONGTEXT,
    min_years_exp INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE audit_logs (
    id INT AUTO_INCREMENT PRIMARY KEY,
    action VARCHAR(50),
    resource_type VARCHAR(50),
    resource_id VARCHAR(50),
    user_id VARCHAR(50),
    details JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_resource (resource_type, resource_id)
);
```

### 5.4 Elasticsearch index setup

```json
PUT /course_embeddings
{
  "settings": {
    "number_of_shards": 1,
    "number_of_replicas": 0,
    "index.vector.size": 1024
  },
  "mappings": {
    "properties": {
      "course_id": {"type": "keyword"},
      "title": {"type": "text"},
      "description": {"type": "text"},
      "skills": {"type": "keyword"},
      "embedding": {
        "type": "dense_vector",
        "dims": 1024,
        "index": true,
        "similarity": "cosine"
      },
      "version_id": {"type": "keyword"},
      "created_at": {"type": "date"}
    }
  }
}
```

### 5.5 S3 / Minio bucket structure

```
vietcv/
  uploads/
    batch_20260430_001/
      courses.docx
      references.pdf
  cache/
    v_20260430_001/
      embeddings_backup.json
  logs/
    pipeline_runs/
      run_20260430_001/
        dag_log.txt
        task_parse.log
        task_embed.log
```

### 5.6 Course docx parser (internal helper)

```python
# services/course_docx_parser.py
from docx import Document

class CourseDocxParser:
    def extract_courses(self, file_obj):
        document = Document(file_obj)
        # Parse structured course docx content
        return []
```

### 5.7 Course Engine (Python microservice)

```python
# services/course_engine/main.py
from services.embedding_builder import EmbeddingBuilder
from services.es_indexer import EsIndexer
from services.cache_manager import CacheManager

def build_course_cache(version_id, courses):
    builder = EmbeddingBuilder(model_path, device='cuda')
    es_indexer = EsIndexer(es_host)
    cache_mgr = CacheManager(s3_client, mysql_client)
    
    # 1. Encode courses → embeddings
    embeddings = builder.encode_batch(courses)
    
    # 2. Index to Elasticsearch
    es_indexer.index_courses(version_id, courses, embeddings)
    
    # 3. Backup to S3
    cache_mgr.backup_to_s3(version_id, courses, embeddings)
    
    # 4. Update MySQL
    cache_mgr.update_version_status(version_id, 'ready')
```

## 6. Airflow DAG: daily_course_update (01:00 UTC)

### 6.1 DAG definition

```python
# airflow/dags/daily_course_update_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'course_engine',
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'daily_course_update',
    default_args=default_args,
    description='Daily course cache build and Elasticsearch index update',
    schedule_interval='0 1 * * *',  # 01:00 UTC every day
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=['course_engine'],
)
```

### 6.2 Tasks (8 steps)

```
┌─────────────────────────────────────────────────────┐
│  fetch_pending_courses                              │
│  (Query MySQL for pending/updated courses)          │
└─────────────────────────────────────┬───────────────┘
                                      │
┌─────────────────────────────────────▼───────────────┐
│  parse_course_docx                                  │
│  (Parse uploaded course docx files internally)      │
└─────────────────────────────────────┬───────────────┘
                                      │
┌─────────────────────────────────────▼───────────────┐
│  build_embeddings                                   │
│  (Encode courses with embedding model)              │
└─────────────────────────────────────┬───────────────┘
                                      │
┌─────────────────────────────────────▼───────────────┐
│  validate_quality                                   │
│  (Check metrics: embedding dims, coverage, etc.)    │
├─────────────────────────────────────┬───────────────┤
│                                     │                │
│      ┌──────────────PASS────────────┘                │
│      │                                               │
│      └───────────────┐                               │
│                      │                               │
│    ┌────────FAIL─────▼───────┐                      │
│    │                         │                       │
└────▼──────────────────────────▼───────────────┐     │
│  swap_cache_and_update_es (PASS branch)     │     │
│  - Update ES index to v_{version}           │     │
│  - Update MySQL version status to 'ready'   │     │
│  - Notify recommendation service            │     │
└─────────────────────────────────────┬────────┘     │
                                      │              │
┌─────────────────────────────────────▼────────┐     │
│  send_notification_success                   │     │
│  (Email/Slack: build completed successfully)│     │
└─────────────────────────────────────┬────────┘     │
                                      │              │
                          ┌───────────┴────────┐     │
                          │                    │     │
┌─────────────────────────▼────────────────────▼─────┐
│  archive_old_versions                               │
│  (Move old ES indices to archive, keep last 3)      │
└─────────────────────────────────────┬───────────────┘
                                      │
                                      │ (FAIL)
                                      ▼
                    ┌─────────────────────────────┐
                    │  handle_failure             │
                    │  - Update MySQL status      │
                    │  - Send alert email         │
                    │  - Keep old ES index active │
                    └─────────────────────────────┘
```

### 6.3 Task implementations

```python
# Task 1: Fetch pending courses
def fetch_pending_courses(**context):
    mysql_client = context['mysql_client']
    courses = mysql_client.query(
        "SELECT * FROM courses WHERE version_id IS NULL OR updated_at > NOW() - INTERVAL 1 DAY"
    )
    context['task_instance'].xcom_push(key='courses', value=courses)
    return len(courses)

task_fetch = PythonOperator(
    task_id='fetch_pending_courses',
    python_callable=fetch_pending_courses,
    dag=dag,
)

# Task 2: Parse and enrich docx files
def parse_course_docx(**context):
    courses = context['task_instance'].xcom_pull(key='courses')
    enriched_courses = []
    for course in courses:
        if course['docx_path']:
            # Parse docx internally with python-docx
            course['parsed'] = True
        enriched_courses.append(course)
    
    context['task_instance'].xcom_push(key='enriched_courses', value=enriched_courses)
    return len(enriched_courses)

task_parse = PythonOperator(
    task_id='parse_course_docx',
    python_callable=parse_course_docx,
    dag=dag,
)

# Task 3: Build embeddings
def build_embeddings(**context):
    courses = context['task_instance'].xcom_pull(key='enriched_courses')
    
    builder = EmbeddingBuilder(model_path, device='cuda')
    course_engine_url = "http://course_engine:8004/build"
    
    version_id = f"v_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    response = requests.post(
        course_engine_url,
        json={
            'version_id': version_id,
            'courses': courses
        }
    )
    
    result = response.json()
    context['task_instance'].xcom_push(key='version_id', value=version_id)
    context['task_instance'].xcom_push(key='build_result', value=result)
    
    return {'status': 'ok', 'version_id': version_id}

task_embed = PythonOperator(
    task_id='build_embeddings',
    python_callable=build_embeddings,
    dag=dag,
)

# Task 4: Validate quality
def validate_quality(**context):
    result = context['task_instance'].xcom_pull(key='build_result')
    
    # Check thresholds
    embedding_dim = result.get('embedding_dim')
    course_count = result.get('course_count')
    avg_embedding_time = result.get('avg_embedding_time_ms')
    
    if embedding_dim != 1024:
        raise ValueError(f"Wrong embedding dimension: {embedding_dim}")
    if course_count == 0:
        raise ValueError("No courses indexed")
    if avg_embedding_time > 5000:  # 5 sec per course max
        raise ValueError(f"Embedding too slow: {avg_embedding_time}ms")
    
    return {'status': 'validated', 'course_count': course_count}

task_validate = PythonOperator(
    task_id='validate_quality',
    python_callable=validate_quality,
    dag=dag,
)

# Task 5: Swap cache and update ES
def swap_cache_and_update_es(**context):
    version_id = context['task_instance'].xcom_pull(key='version_id')
    es_client = context['es_client']
    mysql_client = context['mysql_client']
    
    # Create alias for new index
    es_client.indices.put_alias(index=f"course_embeddings_{version_id}", name="course_embeddings")
    
    # Update MySQL
    mysql_client.execute(
        "UPDATE course_versions SET status = 'ready' WHERE id = %s",
        (version_id,)
    )
    
    # Notify recommendation service
    notify_url = "http://fastapi_service:8000/api/v1/cache/info"
    requests.post(notify_url, json={'version_id': version_id})
    
    return {'status': 'swapped', 'version_id': version_id}

task_swap = PythonOperator(
    task_id='swap_cache_and_update_es',
    python_callable=swap_cache_and_update_es,
    dag=dag,
)

# Task 6: Send success notification
def send_notification_success(**context):
    version_id = context['task_instance'].xcom_pull(key='version_id')
    result = context['task_instance'].xcom_pull(key='build_result')
    
    message = f"""
    ✅ Course cache update completed successfully!
    Version: {version_id}
    Courses indexed: {result['course_count']}
    Time: {result['total_time_sec']}s
    """
    
    # Send email/Slack
    send_slack_notification(message)
    
    return {'status': 'notified'}

task_notify_ok = PythonOperator(
    task_id='send_notification_success',
    python_callable=send_notification_success,
    dag=dag,
)

# Task 7: Archive old versions
def archive_old_versions(**context):
    es_client = context['es_client']
    mysql_client = context['mysql_client']
    s3_client = context['s3_client']
    
    # Keep only last 3 versions
    old_versions = mysql_client.query(
        "SELECT id FROM course_versions WHERE status = 'ready' ORDER BY created_at DESC LIMIT -1 OFFSET 3"
    )
    
    for version in old_versions:
        # Archive ES index
        es_client.indices.put_settings(
            index=f"course_embeddings_{version['id']}",
            body={'index.codec': 'best_compression'}
        )
        
        # Mark in MySQL
        mysql_client.execute(
            "UPDATE course_versions SET status = 'archived', archived_at = NOW() WHERE id = %s",
            (version['id'],)
        )
    
    return {'archived_count': len(old_versions)}

task_archive = PythonOperator(
    task_id='archive_old_versions',
    python_callable=archive_old_versions,
    dag=dag,
)

# Task 8: Handle failure
def handle_failure(**context):
    error = context.get('exception')
    
    message = f"""
    ❌ Course cache update FAILED!
    Error: {str(error)}
    DAG: {context['dag'].dag_id}
    Execution date: {context['execution_date']}
    """
    
    # Keep old ES index active
    # Revert any partial changes
    # Send alert email
    send_slack_notification(message, severity='critical')
    
    return {'status': 'failure_handled'}

task_handle_fail = PythonOperator(
    task_id='handle_failure',
    python_callable=handle_failure,
    trigger_rule='one_failed',
    dag=dag,
)

# Define dependencies
task_fetch >> task_parse >> task_embed >> task_validate
task_validate >> [task_swap, task_handle_fail]
task_swap >> task_notify_ok >> task_archive
```

### 6.4 Monitoring Airflow

- Web UI: `http://airflow-webserver:8080`
- Check DAG status: every day 01:30 UTC
- Alerts: if DAG fails, notification to #course-engine Slack channel

## 7. Step-by-step triển khai

### Step 1: Chuẩn bị infrastructure

1. Docker Compose file với services:
   - MySQL 8.3
   - Elasticsearch 8.x
   - Minio (S3 compatible)
   - Redis 7.x (for Airflow)
   - Airflow (webserver + scheduler + worker)
  - FastAPI service (admin + user routes)
  - Course docx parser (internal in API)
   - Course Engine (Python)

2. Tạo `.env` với tất cả credentials
3. Chạy `docker-compose up -d`
4. Verify tất cả services healthy: `docker-compose ps`

### Step 2: Chuẩn bị database

1. Run MySQL migrations (Alembic):
   ```bash
   docker-compose exec mysql mysql -u vietcv -p vietcv < migrations/schema.sql
   ```

2. Create Elasticsearch index:
   ```bash
   curl -X PUT "http://localhost:9200/course_embeddings" -H "Content-Type: application/json" -d @es_mapping.json
   ```

3. Create S3 buckets:
   ```bash
   aws s3 mb s3://vietcv/uploads --endpoint http://localhost:9000
   aws s3 mb s3://vietcv/cache --endpoint http://localhost:9000
   ```

### Step 3: Implement FastAPI APIs

#### Admin routes (port 8000)
- Course CRUD endpoints
- Pipeline trigger/status endpoints
- Version management endpoints
- Services: CourseService, PipelineService, VersionService

#### User routes (port 8000)
- Recommendation endpoint (POST /api/v1/recommend)
- Cache info endpoint (GET /api/v1/cache/info)
- Services: RecommendationService, EsQueryService

Test:
```bash
curl -X POST http://localhost:8000/api/admin/courses/upload \
  -F "file=@courses.docx"

curl -X POST http://localhost:8000/api/v1/recommend \
  -H "Content-Type: application/json" \
  -d '{"cv_id":"cv_001", "jd_id":"jd_001", "top_k":10}'
```

### Step 4: Implement Python services

#### Course docx parser (internal)
- Dependency: python-docx
- Test: Upload course docx → get parsed JSON

#### Course Engine (port 8004)
- EmbeddingBuilder: encode courses
- EsIndexer: index to ES
- CacheManager: backup to S3
- Test: Build cache for 50 courses

### Step 5: Implement Airflow DAG

1. Create `airflow/dags/daily_course_update_dag.py` (8 tasks)
2. Configure schedule: `0 1 * * *` (01:00 UTC)
3. Set up connections:
   - MySQL connection
   - Elasticsearch connection
   - S3 connection
  - HTTP connection for Course Engine service

4. Validate DAG:
   ```bash
   airflow dags list
   airflow dags validate daily_course_update
   ```

5. Manual trigger test:
   ```bash
   airflow dags trigger daily_course_update
   ```

### Step 6: Deploy frontend

#### Admin FE (NextJS)
- Components: UploadForm, JobQueue, PipelineMonitor, VersionManager
- Connect to FastAPI admin routes (8000)

#### User FE (NextJS)
- Components: CvJdSelector, RecommendationTable, SkillMatcher
- Connect to FastAPI user routes (8000)

### Step 7: End-to-end testing

1. Admin upload 10 courses (docx format)
2. Monitor Airflow DAG (should run at 01:00 or manual trigger)
3. Verify ES index created with embeddings
4. Verify MySQL updated with version
5. User select CV/JD pair
6. User click Recommend
7. Verify top-10 courses returned with scores
8. Check latency < 200ms

### Step 8: Monitoring & Logging

1. Airflow UI: http://localhost:8080
2. ES monitoring: http://localhost:9200/_cat/indices
3. MySQL logs: `docker-compose logs mysql`
4. API logs: `docker-compose logs fastapi_service course_engine`

Setup alerts:
- Airflow DAG failed
- ES index missing
- Recommendation latency > 500ms

### Step 9: Rollback & Disaster recovery

1. Previous version rollback:
   ```bash
  curl -X POST http://localhost:8000/api/admin/versions/v_20260429_001/rollback
   ```

2. Backup restore:
   - MySQL: `mysqldump vietcv | gzip > backup.sql.gz`
   - ES: snapshot to S3
   - S3: versioning enabled

### Step 10: Performance tuning

1. Embedding model:
   - Batch size: tune based on GPU memory (default 8)
   - Device: use CUDA for speedup

2. Elasticsearch:
   - Tune `k` in KNN search (default 10)
   - Adjust `num_candidates` for recall

3. DAG scheduling:
   - Adjust retry count
   - Fine-tune retry delay
   - Monitor DAG duration

## 8. Acceptance checklist

- [ ] Docker services all healthy
- [ ] MySQL schema created with sample data
- [ ] Elasticsearch index created with sample embeddings
- [ ] Admin API endpoints working
- [ ] User API endpoints working
- [ ] Course docx parser handles uploads correctly
- [ ] Course Engine builds embeddings correctly
- [ ] Airflow DAG validates and runs
- [ ] Manual DAG trigger succeeds
- [ ] Recommendation latency acceptable
- [ ] ES query returns top-K results
- [ ] Versioning/rollback works
- [ ] Monitoring/alerts configured
- [ ] All tests passing (unit, integration, E2E)

## 9. Release checklist

- [ ] Benchmark metrics pass (Hit@K, MRR, nDCG)
- [ ] Load test: QPS >= 100, p99 latency < 200ms
- [ ] Staging deployment 24h stable
- [ ] Production deployment plan reviewed
- [ ] Rollback procedure tested
- [ ] Team trained
- [ ] Documentation complete
- [ ] Go-live approval

## 10. Conclusion

Hệ thống này là **production-ready baseline recommendation engine** với:
- Single FastAPI API architecture
- MySQL + Elasticsearch + S3 storage
- Daily Airflow DAG for cache update
- Course docx parsing + embedding service
- Admin + User separate frontends
- Monitoring + rollback capability
