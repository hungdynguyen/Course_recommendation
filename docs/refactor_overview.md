# Refactor Plan - Complete Guide

Hệ thống này sắp được refactor hoàn toàn để tách rạch ròi 2 phần:
- **FastAPI API**: admin + user routes, internal course DOCX parsing, recommendation retrieval
- **Course Engine**: embedding building, ES indexing, cache versioning

---

## 📚 Core Documents

### 1. **[refactor_plan_detailed.md](refactor_plan_detailed.md)** - Kế hoạch chi tiết
  - Current state analysis (vấn đề hiện tại)
  - Target architecture (single FastAPI + Course Engine)
  - Components to remove/add
  - API endpoints (admin + user)
  - Database schema (MySQL + Elasticsearch)
  - Configuration files
  - **6 phases, ~4.2 weeks**
  - Timeline & resources

### 2. **[refactor_checklist.md](refactor_checklist.md)** - Actionable checklist
  - Danh sách từng file cần create/delete/modify
  - Checklist cho service_api + course_engine + airflow
  - Pre/post migration checklists
  - Progress tracking template

### 3. **[refactor_architecture_diagrams.md](refactor_architecture_diagrams.md)** - Diagrams & flows
  - High-level system architecture
  - Data flow: upload → parse → cache → recommend
  - FastAPI service detail
  - Recommendation API service detail
  - Airflow DAG pipeline
  - Cache versioning & rollback
  - Data model (ER diagram)
  - Deployment topology

---

## 🎯 Quick Start

### For Project Managers
1. Read: [refactor_plan_detailed.md](refactor_plan_detailed.md) → Section 1, 13, 14
2. View: [refactor_architecture_diagrams.md](refactor_architecture_diagrams.md) → Diagram 1, 5
3. Plan: **~4.2 weeks**, **2-3 engineers**

### For Engineers (Frontend to Backend)
1. Read: [refactor_checklist.md](refactor_checklist.md) → Choose your phase
2. Deep dive: [refactor_plan_detailed.md](refactor_plan_detailed.md) → Relevant section
3. Code: Follow checklist, tick boxes

### For DevOps/Infrastructure
1. Read: [refactor_plan_detailed.md](refactor_plan_detailed.md) → Section 7 (Config), 9 (Docker)
2. View: [refactor_architecture_diagrams.md](refactor_architecture_diagrams.md) → Diagram 8 (Deployment)
3. Build: docker-compose, Dockerfile(s), migrations

---


---

## 🔑 Key Changes

### What's NEW
- ✅ FastAPI service with unified admin + user routes
- ✅ Internal course DOCX parsing during upload
- ✅ Course Engine for embeddings + ES indexing
- ✅ Airflow DAG (daily scheduling)
- ✅ Cache versioning system (atomic swap, rollback)
- ✅ MySQL metadata DB
- ✅ CRUD operations (create, update, delete courses)

### What's REMOVED
- ❌ Neo4j (KG runtime)
- ❌ Separate OCR service container
- ❌ Laravel 2-tier API split
- ❌ Graph build pipeline
- ❌ Old admin_demo.py endpoints

### What's MIGRATED
- ✅ Embedding service (reuse, no breaking changes)
- ✅ Course data (from legacy to versioned system)
- ✅ Models (before/after train)

---

## 📁 Folder Structure (Post-Refactor)

```
src/
  service_api/               # ← Unified FastAPI API
    api/v1/endpoints/
      admin.py               # NEW: unified admin routes
      recommendations.py     # NEW: simplified
      health.py
      cache.py
    services/
      baseline_recommender.py        # NEW
      embedding_loader.py            # NEW
      pair_resolver.py               # NEW
    scripts/
      build_baseline_cache.py        # NEW

  course_engine/             # ← Course embedding/indexing
    api/
      routes.py
    services/
      embedding_builder.py
      es_indexer.py
      cache_manager.py
    main.py

  airflow/                   # ← NEW: Scheduling
    dags/
      daily_course_update_dag.py
    tasks/
      fetch_courses.py
      parse_and_enrich.py
      build_embeddings.py
      validate_quality.py
      swap_cache.py
      notify.py

  shared/
    models/
      course_model.py        # NEW
    storage/
      course_storage.py      # NEW
      cache_versioning.py    # NEW
```

---

## 🚀 Implementation Order

Recommended sequence:

1. **Phase 1 - Week 1**: Database + Airflow + folder structure
2. **Phase 2 - Week 2**: FastAPI service (admin + user routes)
3. **Phase 3 - Week 3**: Course Engine (depends on Phase 1)
4. **Phase 4 - Week 4**: Airflow DAG (depends on Phase 3)
5. **Phase 5 - Week 4-5**: Docker & E2E testing

---

## 📊 API Summary

### FastAPI API (single service)
```
POST /api/v1/recommend          # Main endpoint
GET  /api/v1/cache/info         # Cache metadata
GET  /health/ready              # Readiness probe
GET  /health/live               # Liveness probe
POST /api/admin/courses/upload   # Admin upload
POST /api/admin/pipeline/trigger # Trigger DAG
GET  /api/admin/versions         # Version history
```

---

## 💾 Data Model Overview

### MySQL Tables
- `courses` - Course metadata
- `course_versions` - Version tracking
- `upload_batches` - Upload history
- `pipeline_runs` - Airflow run logs
- `audit_logs` - Action tracking

### File System
```
data/
  uploads/                   # Temporary uploads
  catalog/                   # Organized course files per version
  processed/course_cache/    # Precomputed embeddings
    v_20260426_001/
      embeddings.npy
      metadata.jsonl
    current/ → v_20260426_001  # Symlink (atomic swap)
```

---

## ✅ Success Criteria

When refactor complete:

- [ ] All key FastAPI/Course Engine/Airflow files created or updated
- [ ] All KG-related files deleted
- [ ] All unit tests passing
- [ ] All integration tests passing
- [ ] API latency < 200ms p99
- [ ] Airflow DAG runs daily successfully
- [ ] Cache versioning + rollback works
- [ ] CRUD operations functional
- [ ] Docker compose brings up all services
- [ ] No KG dependencies remaining

---

## 🔗 Related Docs

Also read alongside these refactor docs:

- [docs/deployment_baseline_system.md](deployment_baseline_system.md) - Baseline architecture fundamentals
- [docs/system_design_layered.md](system_design_layered.md) - System design diagram
- [docs/deployment_baseline_admin_user_step_by_step.md](deployment_baseline_admin_user_step_by_step.md) - Admin/user flows

---

## 📞 Questions?

For specific guidance:
1. **Architecture questions** → [refactor_plan_detailed.md](refactor_plan_detailed.md) Sections 1-8
2. **What to code** → [refactor_checklist.md](refactor_checklist.md)
3. **How components fit** → [refactor_architecture_diagrams.md](refactor_architecture_diagrams.md)
4. **Timeline/resources** → [refactor_plan_detailed.md](refactor_plan_detailed.md) Section 13

---

## 🎓 Key Concepts

### Baseline-only system
- No Knowledge Graph at runtime
- Embedding-based cosine similarity retrieval
- Precomputed course vectors for speed
- Simplest, fastest, proven approach

### 2 Components
- **Recommendation API**: Read-only, fast, stateless
- **Unified FastAPI upload/admin routes**: Write-heavy, internal DOCX parsing, stateful

### Cache versioning
- Atomic swap (symlink)
- Rollback capability
- Track version history
- Quality validation before switch

### Airflow scheduling
- Daily 01:00 UTC
- Fault tolerance (retries)
- Task monitoring
- Lineage tracking

---

## 📅 Maintenance & Support

**After deployment:**
- Monitor Airflow DAG daily runs
- Track API metrics (latency, QPS)
- Review audit logs weekly
- Archive old cache versions monthly
- Test rollback quarterly

---

**Last updated**: April 2026
**Status**: Ready for implementation
**Owner**: Engineering Team
