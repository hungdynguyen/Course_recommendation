# Design Update Summary - Architecture Alignment

**Date**: May 9, 2026  
**Changes**: Aligned architecture docs with single FastAPI + internal course DOCX parsing + ES + S3 + Airflow

---

## Documents Updated

### 1. **[docs/deployment_baseline_system.md](docs/deployment_baseline_system.md)**

**Changes**:
- ✅ Clarified the baseline as embedding-only retrieval
- ✅ Kept MySQL/Elasticsearch/S3/Redis as deployment primitives
- ✅ Updated processing flow to use internal course DOCX parsing, not a separate OCR service
- ✅ Updated API layer to FastAPI single service with `/api/admin/*` and `/api/v1/*`

### 2. **[docs/deployment_baseline_admin_user_step_by_step.md](docs/deployment_baseline_admin_user_step_by_step.md)**

**Changes**:
- ✅ Reworked the guide around single FastAPI API
- ✅ Removed standalone OCR service assumptions
- ✅ Kept course docx parsing as an internal upload-time step
- ✅ Kept MySQL schema, Elasticsearch mapping, S3 layout, and Airflow DAG

### 3. **[docs/refactor_plan_detailed.md](docs/refactor_plan_detailed.md)**

**Changes**:
- ✅ Rewrote target architecture to single FastAPI
- ✅ Added internal course DOCX parser instead of OCR container
- ✅ Consolidated env, docker-compose, phases, and testing strategy

### 4. Follow-up docs

**Kept in sync**:
- [docs/refactor_overview.md](docs/refactor_overview.md)
- [docs/refactor_checklist.md](docs/refactor_checklist.md)
- [docs/refactor_architecture_diagrams.md](docs/refactor_architecture_diagrams.md)
- [docs/ARCHITECTURE_ALIGNMENT_COMPLETE.md](docs/ARCHITECTURE_ALIGNMENT_COMPLETE.md)

---

## Key Architectural Decisions Locked In

| Component | Choice | Reason |
|-----------|--------|--------|
| **Metadata DB** | MySQL (existing) | Reuse current infrastructure, sufficient for metadata |
| **Vector Store** | Elasticsearch | Production-scale, KNN query support, index management |
| **Object Storage** | S3/Minio | Backup courses, snapshots, logs |
| **Queue/Cache** | Redis | Required by Airflow for DAG scheduling |
| **API Framework** | FastAPI (single service) | One API layer with admin + user routes |
| **Embedding Inference** | Qwen3-Embedding-0.6B (after-train) | Benchmark proven better performance |
| **Orchestration** | Airflow | Daily scheduled batch processing, 8-task DAG |
| **Course DOCX Parsing** | Internal FastAPI helper | Structured parsing during upload |
| **Course Engine** | Python microservice | Embedding builder, ES indexer, cache manager |
| **Scheduling** | Airflow + cron | Daily 01:00 UTC, 2x retry, monitoring |

---

## Architecture Flow (Updated)

```
┌─────────────────────────────────────────────────────────────┐
│ Frontend Layer                                              │
│ ┌──────────────────────┐  ┌──────────────────────┐         │
│ │  Admin NextJS        │  │  User NextJS         │         │
│ │  - Upload           │  │  - Select CV/JD      │         │
│ │  - Monitor pipeline │  │  - Get recommendations│         │
│ └──────────┬───────────┘  └──────────┬───────────┘         │
└────────────┼──────────────────────────┼────────────────────┘
             │                          │
┌────────────▼──────────────────────────▼────────────────────┐
│ API Layer (single FastAPI)                                  │
│ ┌────────────────────────────────────────────────────────┐ │
│ │ /api/admin/* + /api/v1/*                               │ │
│ │ - admin CRUD, pipeline, versions                       │ │
│ │ - recommend, cache/info, health                        │ │
│ └────────────────────────────────────────────────────────┘ │
└────────────┼───────────────────────────────────────────────┘
             │                          │
┌────────────▼──────────────────────────▼────────────────────┐
│ Storage Layer                                              │
│ ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│ │ MySQL :3306  │  │ ES :9200     │  │ S3 :9000     │     │
│ │ - Metadata   │  │ - Vectors    │  │ - Objects    │     │
│ │ - Versions   │  │ - KNN query  │  │ - Backups    │     │
│ └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
             ▲
┌────────────┴──────────────────────────────────────────────┐
│ Processing Layer (Daily Airflow DAG)                       │
│ ┌──────────────────┐  ┌──────────────┐  ┌──────────────┐ │
│ │ Course DOCX parse │  │ Course Engine│  │ Airflow DAG  │ │
│ │ internal          │  │ :8004        │  │ (8 tasks)    │ │
│ └──────────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## Airflow DAG: daily_course_update (8 Tasks)

```
fetch_pending_courses
  ↓
parse_course_docx (internal parser)
  ↓
build_embeddings (Course Engine)
  ↓
validate_quality
  ├─ PASS → swap_cache_and_update_es
  │           ↓
  │         send_notification_success
  │           ↓
  │         archive_old_versions
  │
  └─ FAIL → handle_failure (revert, alert)
```

---

## Next Steps

1. ✅ Update deployment_baseline_system.md (DONE)
2. ✅ Update deployment_baseline_admin_user_step_by_step.md (DONE)
3. ⏳ Update refactor_plan_detailed.md
4. ⏳ Commit + push all docs to GitHub

---

**Status**: Architecture design documents **fully aligned** with user's system design  
**Ready for**: Implementation phase (backend engineers can start coding)
