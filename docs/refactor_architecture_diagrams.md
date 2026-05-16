# Refactor Architecture Diagrams

## 1. High-level System Architecture (Post-Refactor)

```mermaid
flowchart TB
  subgraph frontend["Frontend Layer"]
    admin["Admin UI\n(Upload, Schedule, Monitor)"]
    user["User UI\n(Select CV/JD, Get Recommendations)"]
  end

  subgraph api_layer["Recommendation API Service"]
    direction TB
    rec_api["FastAPI App\n:8000"]
    rec_endpoint["POST /recommend\nGET /cache/info\nGET /health"]
    recommender["BaselineRecommender\n(Query → Cosine → Top-K)"]
    embedding_loader["EmbeddingLoader\n(Load precomputed cache)"]
    pair_resolver["PairResolver\n(Resolve CV/JD)"]
  end

  subgraph manager_layer["Course Manager Service"]
    direction TB
    manager_api["FastAPI App\n:8001"]
    course_admin["POST /courses/upload\nGET /courses/list\nPUT/DELETE /courses/{id}"]
    pipeline_admin["POST /pipeline/trigger\nGET /pipeline/status"]
    ingest["CourseIngestService\n(Validation, Queue)"]
    metadata["CourseMetadataService\n(CRUD, DB)"]
  end

  subgraph airflow_layer["Airflow Scheduler"]
    direction TB
    airflow_dag["daily_course_update_dag\n(Schedule: 01:00 daily)"]
    task1["fetch_courses"]
    task2["parse_and_enrich\n(LLM call)"]
    task3["build_embeddings"]
    task4["validate_quality"]
    task5["swap_cache\n(Atomic)"]
    task6["notify"]
  end

  subgraph storage["Storage Layer"]
    direction TB
    postgres["PostgreSQL\n(courses, versions, batches)"]
    redis["Redis\n(queue, cache)"]
    disk["Disk Storage\n(staging, catalog, cache)"]
    llm["LLM Endpoint\n(parse/enrich)"]
  end

  %% Flows
  admin -->|upload courses| manager_api
  manager_api -->|route| ingest
  ingest -->|validate| redis
  ingest -->|persist| metadata
  metadata -->|write| postgres

  airflow_dag --> task1
  task1 --> task2
  task2 -->|call| llm
  task2 --> task3
  task3 --> task4
  task4 --> task5
  task5 -->|swap| disk
  task5 -->|signal reload| rec_api
  task5 --> task6

  rec_api -->|load| embedding_loader
  embedding_loader -->|read| disk
  rec_api -->|query encode| recommender
  recommender -->|cosine similarity| embedding_loader
  user -->|recommend request| rec_api
  rec_api -->|response| user

  ingest -->|read staging| disk
  ingest -->|read from| postgres
  metadata -->|query| postgres

  rec_endpoint -.->|part of| rec_api
  course_admin -.->|part of| manager_api
  pipeline_admin -.->|part of| manager_api
```

## 2. Data Flow: Upload → Cache → Recommend

```mermaid
sequenceDiagram
  participant Admin as Admin User
  participant ManagerUI as Manager UI
  participant ManagerAPI as Course Manager API
  participant Airflow as Airflow Scheduler
  participant LLM as LLM Endpoint
  participant Storage as Storage
  participant RecAPI as Recommendation API
  participant User as End User

  Admin->>ManagerUI: Upload courses (ZIP/JSON)
  ManagerUI->>ManagerAPI: POST /courses/upload
  ManagerAPI->>Storage: Save to staging/{batch_id}
  ManagerAPI->>Storage: Create queue record
  ManagerAPI-->>ManagerUI: {batch_id, status: queued}

  Note over Airflow: 01:00 Daily Trigger
  Airflow->>ManagerAPI: GET /courses/list?pending=true
  ManagerAPI->>Storage: Read staging files
  ManagerAPI-->>Airflow: [courses]

  Airflow->>LLM: POST /parse [courses]
  LLM-->>Airflow: [enriched_courses]

  Airflow->>Storage: Encode courses → course_embeddings.npy
  Airflow->>Storage: Write course_metadata.jsonl
  Airflow->>Storage: Create v_20260427_001/ version

  Airflow->>Storage: Validate quality metrics
  Airflow->>Storage: Atomic swap (symlink: current → v_20260427_001)
  Airflow->>RecAPI: Signal: reload cache

  RecAPI->>Storage: Reload embeddings from current/
  RecAPI-->>Admin: Notify: Cache updated

  User->>RecAPI: POST /recommend {cv_id, jd_id, top_k: 10}
  RecAPI->>Storage: Lookup CV/JD
  RecAPI->>RecAPI: Encode query
  RecAPI->>RecAPI: Cosine similarity search
  RecAPI-->>User: [{course_id, score, rank}, ...]
```

## 3. Course Manager Service Detail

```mermaid
flowchart TB
  subgraph endpoints["Endpoints"]
    upload["POST /courses/upload"]
    list["GET /courses/list"]
    get["GET /courses/{id}"]
    update["PUT /courses/{id}"]
    delete["DELETE /courses/{id}"]
    trigger["POST /pipeline/trigger"]
    status["GET /pipeline/status/{run_id}"]
    versions["GET /versions"]
    rollback["POST /versions/{id}/rollback"]
  end

  subgraph services["Services"]
    ingest["CourseIngestService"]
    metadata["CourseMetadataService"]
    builder["CourseCacheBuilder"]
    llm_client["LLMEnrichService"]
    airflow_client["AirflowTrigger"]
  end

  subgraph storage["Storage"]
    postgres[("PostgreSQL\n(metadata)")]
    redis["Redis\n(queue)"]
    staging["Disk: staging/"]
    catalog["Disk: catalog/"]
    cache["Disk: cache/"]
  end

  upload -->|validate| ingest
  list -->|query| metadata
  get -->|query| metadata
  update -->|update| metadata
  delete -->|soft delete| metadata
  trigger -->|start| airflow_client
  status -->|poll| airflow_client
  versions -->|list| builder
  rollback -->|swap symlink| builder

  ingest -->|save| staging
  ingest -->|persist| postgres
  ingest -->|queue| redis

  metadata -->|CRUD| postgres
  builder -->|compose text| metadata
  builder -->|encode| cache
  builder -->|swap| cache
  llm_client -->|call API| cache
```

## 4. Recommendation API Service Detail

```mermaid
flowchart TB
  subgraph endpoints["Endpoints"]
    recommend["POST /recommend\n{cv_id, jd_id, top_k}"]
    health["GET /health\nGET /health/ready\nGET /health/live"]
    info["GET /cache/info"]
  end

  subgraph services["Services"]
    resolver["PairResolver\n(Resolve CV/JD)"]
    recommender["BaselineRecommender\n(Encode + Cosine)"]
    loader["EmbeddingLoader\n(Version mgmt)"]
  end

  subgraph storage["Storage"]
    embeddings["course_embeddings.npy"]
    metadata["course_metadata.jsonl"]
    versions_json["versions.json"]
    cv_store["CV Store\n(files/db)"]
    jd_store["JD Store\n(files/db)"]
  end

  recommend -->|resolve| resolver
  resolver -->|fetch| cv_store
  resolver -->|fetch| jd_store
  resolver -->|return| recommender

  recommender -->|load cache| loader
  loader -->|read current| versions_json
  loader -->|load vectors| embeddings
  loader -->|load metadata| metadata
  recommender -->|cosine similarity| embeddings
  recommender -->|rank top-k| metadata
  recommend -->|response| user["Response"]
```

## 5. Airflow DAG Pipeline

```mermaid
flowchart LR
  start["START"] --> fetch["fetch_courses\n(Pending batches)"]
  fetch --> parse["parse_and_enrich\n(LLM call)"]
  parse --> build["build_embeddings\n(Encode courses)"]
  build --> validate["validate_quality\n(Check metrics)"]
  validate -->|pass| swap["swap_cache\n(Atomic switch)"]
  validate -->|fail| cleanup_fail["cleanup\n(Archive failed)"]
  swap --> notify["notify\n(Success)"]
  cleanup_fail --> end_fail["END FAILED"]
  notify --> end_ok["END SUCCESS"]

  style fetch fill:#e1f5ff
  style parse fill:#fff3e0
  style build fill:#f3e5f5
  style validate fill:#e8f5e9
  style swap fill:#fce4ec
  style notify fill:#e0f2f1
  style cleanup_fail fill:#ffebee
```

## 6. Cache Versioning & Rollback

```mermaid
flowchart TB
  v1["v_20260420_001\n(500 courses)\nStatus: ready\nCreated: 2026-04-20"]
  v2["v_20260421_001\n(510 courses)\nStatus: ready\nCreated: 2026-04-21"]
  v3["v_20260422_001\n(520 courses)\nStatus: ready\nCreated: 2026-04-22"]
  v4["v_20260426_001\n(530 courses)\nStatus: ready\nCreated: 2026-04-26\n**CURRENT**"]
  building["v_20260427_001\n(535 courses)\nStatus: building\nCreated: 2026-04-27"]

  versions_db["versions.json\n{\n  current: v_20260426_001\n  history: [...]\n  rollback_enabled: true\n}"]

  v1 -->|archive| archive["Archive\n(old versions)"]
  v2 -->|keep| kept["Available\n(versioned storage)"]
  v3 -->|keep| kept
  v4 -->|active| symlink["current/\n(symlink)"]
  building -->|building| in_progress["In progress...\n(not linked)"]

  symlink -->|ready| RecAPI["Recommendation API\nloads from current/"]

  versions_db -->|track| v1
  versions_db -->|track| v2
  versions_db -->|track| v3
  versions_db -->|track| v4

  Rollback["User action:\nPOST /rollback/v_20260425_001"] -->|update symlink| v3
  v3 -->|becomes| new_current["current/\n(swapped back)"]
```

## 7. Data Model Relationships

```mermaid
erDiagram
  COURSES ||--o{ COURSE_VERSIONS : "included_in"
  COURSES {
    string course_id PK
    string title
    string description
    string[] skills
    string version_id FK
    datetime created_at
    datetime updated_at
    datetime deleted_at "nullable"
  }

  COURSE_VERSIONS {
    string version_id PK
    int course_count
    string embeddings_path
    string metadata_path
    string checksum
    string status "building|ready|archived"
    datetime created_at
    datetime archived_at "nullable"
  }

  UPLOAD_BATCHES {
    string batch_id PK
    int file_count
    bigint total_size_bytes
    string status "pending|processing|completed|failed"
    string error_message "nullable"
    datetime created_at
    datetime completed_at "nullable"
  }

  PIPELINE_RUNS {
    string run_id PK
    string batch_id FK
    string status "running|completed|failed"
    string version_id FK
    int progress_percent
    string error_log "nullable"
    datetime started_at
    datetime completed_at "nullable"
  }

  UPLOAD_BATCHES ||--o{ PIPELINE_RUNS : "triggers"
  COURSE_VERSIONS ||--o{ PIPELINE_RUNS : "produces"

  AUDIT_LOGS {
    int id PK
    string action "create|update|delete|upload|process"
    string resource_type "course|batch|version"
    string resource_id
    string user_id
    json details
    datetime created_at
  }

  COURSES ||--o{ AUDIT_LOGS : "logged_in"
  COURSE_VERSIONS ||--o{ AUDIT_LOGS : "logged_in"
```

## 8. Deployment Topology (Docker Compose)

```mermaid
graph TB
  subgraph host["Host Machine"]
    volume1["Volume: /data/staging"]
    volume2["Volume: /data/catalog"]
    volume3["Volume: /data/cache"]
    volume4["Volume: /data/models"]
  end

  subgraph network["Docker Network"]
    rec_container["recommendation_api\n(FastAPI :8000)\nHealthy"]
    manager_container["course_engine\n(FastAPI :8004)\nHealthy"]
    airflow_web["airflow_webserver\n(:8080)"]
    airflow_sched["airflow_scheduler"]
    airflow_worker["airflow_worker"]
    postgres["postgres\n(:5432)"]
    redis["redis\n(:6379)"]
  end

  volume1 -.->|mount| manager_container
  volume2 -.->|mount| manager_container
  volume3 -.->|mount| rec_container
  volume3 -.->|mount| manager_container
  volume4 -.->|mount| rec_container

  rec_container -->|query| redis
  rec_container -->|read cache| volume3
  manager_container -->|write staging| volume1
  manager_container -->|write catalog| volume2
  manager_container -->|query/write| postgres
  manager_container -->|queue jobs| redis

  airflow_web -->|display| airflow_sched
  airflow_sched -->|schedule| airflow_worker
  airflow_worker -->|call API| manager_container
  airflow_worker -->|write cache| volume3
  airflow_worker -->|update DB| postgres

  rec_container -->|subscribe| redis
  redis -->|notify| rec_container
```

---

## Kết luận

Các diagrams này cho phép bạn:
1. **Hiểu rõ hệ thống sau refactor** (2 services rạch ròi)
2. **Theo dõi data flow** (upload → airflow → cache → recommend)
3. **Biết cách components kết nối** (services, storage, external systems)
4. **Lên kế hoạch deployment** (Docker, volumes, networking)
5. **Hỗ trợ communication** với team/stakeholders

Combine những diagrams này với [refactor_plan_detailed.md](refactor_plan_detailed.md) và [refactor_checklist.md](refactor_checklist.md) để có đầy đủ hướng dẫn triển khai.
