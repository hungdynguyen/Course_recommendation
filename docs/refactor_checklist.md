# Refactor Checklist - Single FastAPI Architecture

Dùng checklist này để track các file cần tạo/xóa/sửa trước khi implement.

---

## Phase 1: Foundation

### Tạo folder structure
- [ ] Create `src/service_api/api/v1/endpoints/`
- [ ] Create `src/service_api/services/`
- [ ] Create `src/service_api/models/`
- [ ] Create `src/service_api/scripts/`
- [ ] Create `src/course_engine/api/`
- [ ] Create `src/course_engine/services/`
- [ ] Create `src/shared/models/`
- [ ] Create `src/shared/storage/`
- [ ] Create `src/airflow/dags/`
- [ ] Create `src/airflow/tasks/`

### Database + infra
- [ ] Create Alembic migration for `courses`, `course_versions`, `upload_batches`, `pipeline_runs`, `cv_profiles`, `job_descriptions`, `audit_logs`
- [ ] Create Elasticsearch mapping for `course_embeddings`
- [ ] Create S3/Minio bucket structure (`uploads/`, `cache/`, `snapshots/`, `logs/`)
- [ ] Verify MySQL, ES, Redis, Minio, Airflow config in `.env`

---

## Phase 2: FastAPI Service

### Admin routes
- [ ] **CREATE** `src/service_api/api/v1/endpoints/admin.py`
  - `POST /api/admin/courses/upload`
  - `GET /api/admin/courses`
  - `PUT /api/admin/courses/{course_id}`
  - `DELETE /api/admin/courses/{course_id}`
  - `POST /api/admin/pipeline/trigger`
  - `GET /api/admin/pipeline/status/{dag_run_id}`
  - `GET /api/admin/versions`
  - `POST /api/admin/versions/{version_id}/rollback`

### User routes
- [ ] **MODIFY** `src/service_api/api/v1/endpoints/recommendations.py`
  - Use Elasticsearch KNN search
  - Keep `/api/v1/recommend`
  - Keep response schema lean

- [ ] **MODIFY** `src/service_api/api/v1/endpoints/health.py`
  - Add readiness/liveness probes

- [ ] **CREATE** `src/service_api/api/v1/endpoints/cache.py`
  - `GET /api/v1/cache/info`

### Services
- [ ] **CREATE** `src/service_api/services/course_docx_parser.py`
  - Parse structured course DOCX internally with `python-docx`

- [ ] **CREATE** `src/service_api/services/course_service.py`
  - CRUD + audit logging in MySQL

- [ ] **CREATE** `src/service_api/services/pipeline_service.py`
  - Trigger Airflow DAG + status lookup

- [ ] **CREATE** `src/service_api/services/version_service.py`
  - Version list + rollback logic

- [ ] **CREATE** `src/service_api/services/elasticsearch_query.py`
  - KNN search + alias swap helpers

- [ ] **CREATE** `src/service_api/services/embedding_loader.py`
  - Load precomputed cache metadata

### Models
- [ ] **CREATE** `src/service_api/models/request.py`
- [ ] **CREATE** `src/service_api/models/response.py`
- [ ] **CREATE** `src/service_api/models/schemas.py`

### Main/config
- [ ] **MODIFY** `src/service_api/main.py`
- [ ] **MODIFY** `src/service_api/config.py`
- [ ] **MODIFY** `src/service_api/dependencies.py`

### Testing
- [ ] Unit tests for admin routes
- [ ] Unit tests for recommendation route
- [ ] Integration tests with MySQL + Elasticsearch

---

## Phase 3: Course Engine

- [ ] **CREATE** `src/course_engine/main.py`
- [ ] **CREATE** `src/course_engine/services/embedding_builder.py`
- [ ] **CREATE** `src/course_engine/services/es_indexer.py`
- [ ] **CREATE** `src/course_engine/services/cache_manager.py`
- [ ] **CREATE** `src/course_engine/api/routes.py`
- [ ] Validate build/index/backup flow

---

## Phase 4: Airflow

- [ ] **CREATE** `src/airflow/dags/daily_course_update_dag.py`
- [ ] **CREATE** `src/airflow/tasks/fetch_courses.py`
- [ ] **CREATE** `src/airflow/tasks/parse_course_docx.py`
- [ ] **CREATE** `src/airflow/tasks/build_embeddings.py`
- [ ] **CREATE** `src/airflow/tasks/validate_quality.py`
- [ ] **CREATE** `src/airflow/tasks/swap_cache.py`
- [ ] **CREATE** `src/airflow/tasks/notify.py`
- [ ] **CREATE** `src/airflow/tasks/archive.py`

---

## Phase 5: Docker & Deployment

- [ ] **MODIFY** `docker-compose.yml`
  - single `fastapi_service`
  - `course_engine`
  - `mysql`, `elasticsearch`, `minio`, `redis`, `airflow-*`

- [ ] **CREATE/UPDATE** `docker/Dockerfile.api`
- [ ] **CREATE** `docker/Dockerfile.course_engine`
- [ ] **CREATE** `docker/Dockerfile.airflow`

---

## Phase 6: Cleanup

### Delete old components
- [ ] **DELETE** `src/service_api/services/skill_search.py`
- [ ] **DELETE** `src/service_api/services/course_recommendation.py`
- [ ] **DELETE** any standalone OCR service folder/files
- [ ] **DELETE** any Laravel/admin_api or user_api service files
- [ ] **DELETE** `src/data_factory/scripts/build_graph.py` if unused

### Update docs
- [ ] Update `docs/refactor_plan_detailed.md`
- [ ] Update `docs/design_update_summary.md`
- [ ] Update `docs/deployment_baseline_system.md`
- [ ] Update `docs/deployment_baseline_admin_user_step_by_step.md`
- [ ] Update `docs/refactor_architecture_diagrams.md`
- [ ] Update `docs/ARCHITECTURE_ALIGNMENT_COMPLETE.md`

---

## Verification Checklist

- [ ] No OCR service mentions in architecture docs
- [ ] No Laravel 2-tier API mentions in architecture docs
- [ ] No `port 8001/8002` API split remains
- [ ] CV/JD are documented as structured input
- [ ] Only course DOCX parsing is internal
- [ ] FastAPI single service is the only API layer
- [ ] Course Engine remains separate
- [ ] Airflow DAG remains daily 01:00 UTC
- [ ] MySQL + Elasticsearch + S3/Minio remain the storage core

---

## Progress Tracking

```
Foundation:      [ ] 0% [ ] 25% [ ] 50% [ ] 75% [ ] 100%
FastAPI API:     [ ] 0% [ ] 25% [ ] 50% [ ] 75% [ ] 100%
Course Engine:   [ ] 0% [ ] 25% [ ] 50% [ ] 75% [ ] 100%
Airflow:         [ ] 0% [ ] 25% [ ] 50% [ ] 75% [ ] 100%
Docker:          [ ] 0% [ ] 25% [ ] 50% [ ] 75% [ ] 100%
Docs:            [ ] 0% [ ] 25% [ ] 50% [ ] 75% [ ] 100%
```
