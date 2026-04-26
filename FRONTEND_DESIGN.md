# Frontend Design – Course Recommendation System

## I. Tổng Quan

Xây dựng một **web frontend đơn giản** gồm 2 module:
1. **User Module**: Drag-drop CV → chọn JD → nhận course recommendations
2. **Admin Module**: Quản lý dữ liệu – drag-drop course files → tự động ingest vào KG

---

## II. Kiến Trúc Hệ Thống

```
┌──────────────────────────────────────────────────────────────────┐
│                        WEB FRONTEND                              │
│  ┌──────────────────────┐      ┌──────────────────────────────┐ │
│  │   User Module        │      │   Admin Module               │ │
│  │  - CV Upload         │      │  - Course Upload Area        │ │
│  │  - JD Selection      │      │  - Pending Queue Viewer      │ │
│  │  - Recommendations   │      │  - Run Ingest (manual demo)  │ │
│  │    Display           │      │  - Ingest Status Monitor     │ │
│  └──────────────────────┘      └──────────────────────────────┘ │
└────┬──────────────────────────────────────────────────┬───────────┘
     │ HTTP/REST                                        │
┌────▼──────────────────────────────────────────────────▼───────────┐
│                    FastAPI Backend                                │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │ NEW Endpoints (to be added):                                │ │
│  │ POST   /api/v1/admin/courses/upload    (bulk upload)        │ │
│  │ GET    /api/v1/admin/courses/queue     (pending list)       │ │
│  │ POST   /api/v1/admin/pipeline/run      (trigger ingest)     │ │
│  │ GET    /api/v1/admin/pipeline/status   (ingest status)      │ │
│  │ POST   /api/v1/recommendations/gap     (existing – user)    │ │
│  │ GET    /api/v1/jds/list                (NEW – JD list)      │ │
│  └──────────────────────────────────────────────────────────────┘ │
└────┬──────────────────────────────────────────────────────────────┘
     │ Shared Libraries & Infrastructure
┌────▼──────────────────────────────────────────────────────────────┐
│  Neo4j KG          │  Elasticsearch (skills)  │  File Storage     │
│  (CourseNode,      │  (CanonicalSkill vector)│  (Course staging) │
│   SkillNode,       │                         │                   │
│   TEACHES/REQUIRES)│                         │                   │
└────────────────────────────────────────────────────────────────────┘
```

---

## III. Module 1: User Interface

### 3.1 Luồng Người Dùng

```
┌─────────────┐      ┌─────────────┐      ┌──────────────┐     ┌──────────────┐
│  User Page  │──→   │ Drag-drop   │──→   │ Select JD    │──→  │ Show Results │
│             │      │ CV File     │      │ from List    │     │              │
└─────────────┘      └─────────────┘      └──────────────┘     └──────────────┘
                            │                     │                    ▲
                            └─────────────────────┴────────────────────┘
                                        │
                           (POST /api/v1/recommendations/gap)
                                        │
                           Input: CV text + JD title + Keywords
                           Output: Courses + Skills covered
```

### 3.2 Components & Functionality

| Component | Action | Details |
|-----------|--------|---------|
| **CV Upload** | Drag-drop or click | - Accept `.pdf`, `.docx`, `.txt` <br> - Extract text (backend service) <br> - Store in session (max 5MB) |
| **JD List** | Dropdown / Autocomplete | - Fetch từ `/api/v1/jds/list` <br> - Show JD title + brief description <br> - Allow search/filter |
| **JD Keywords** | Optional input | - User có thể thêm trending keywords <br> - Separated by comma/newline <br> - Used to enrich gap detection |
| **Recommend Button** | Click → Submit | - POST to `/api/v1/recommendations/gap` <br> - Show loading state <br> - Display results trong 2-5 sec |
| **Results Panel** | Display | - Recommended courses (sorted by skill coverage) <br> - Skills covered + Uncovered skills <br> - Course title + Category |

### 3.3 API Contract (User)

```json
POST /api/v1/recommendations/gap

Request:
{
  "cv_text": "5+ years Python...",
  "jd_title": "Senior Software Engineer",
  "jd_keywords": ["Python", "React", "Docker"],
  "gap_focus_terms": ["optional explicit gaps"],
  "max_courses": 10
}

Response:
{
  "gap_skills": [
    {
      "skill_id": "abc123",
      "label": "Docker",
      "description": "Containerization...",
      "category": "DevOps"
    }
  ],
  "recommended_courses": [
    {
      "course_id": "COURSE_001",
      "course_title": "Advanced Docker & Kubernetes",
      "category": "Khoa Công nghệ thông tin_2024",
      "covered_gaps": ["docker", "kubernetes"],
      "coverage_count": 2
    }
  ],
  "metadata": {
    "processing_time_ms": 1250,
    "total_gaps_detected": 15,
    "courses_returned": 8
  }
}
```

---

## IV. Module 2: Admin Interface

### 4.1 Luồng Admin

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Admin Page      │     │  Drag-drop       │     │  Auto-save to    │
│                  │──→  │  Course JSON     │──→  │  Staging Dir      │
│                  │     │  Files           │     │                  │
└──────────────────┘     └──────────────────┘     └────┬─────────────┘
        ▲                                              │
        │                                              │
        └──────────────────────────────────────────────┘
                    (Poll pending queue)
                            │
                            ▼
        ┌──────────────────────────────┐
        │  Display Pending Queue        │
        │  - File names                 │
        │  - Upload time                │
        │  - No. of courses             │
        └────────────┬──────────────────┘
                     │
                     │ [RUN INGEST] button
                     ▼
        ┌──────────────────────────────────────┐
        │  Execute Graph Build Pipeline         │
        │  (Async to ETL processor)            │
        │                                       │
        │  1. Move staging → processing         │
        │  2. Load + Embed + Dedup             │
        │  3. Index → ES + Build Neo4j KG      │
        │  4. Move to archive                  │
        └────────────┬─────────────────────────┘
                     │
                     ▼
        ┌──────────────────────────────────────┐
        │  Status Monitor                       │
        │  - Phase: Loading → Embed → Dedup    │
        │  - Progress bar (%)                   │
        │  - Duration per phase                │
        │  - Last completed: [time]            │
        │  - Next scheduled: [time] (batch)    │
        └──────────────────────────────────────┘
```

### 4.2 Staging Strategy (Tối ưu tài nguyên)

**Vấn đề**: Ingest nhiều course có thể dùng nhiều memory/compute, nên cần batching.

**Giải pháp**:

1. **Staging Directory** (`data/staging/pending/`)
   - Mỗi upload → lưu file tương ứng vào đây
   - JSON format: tuân theo LLM extraction schema
   - Admin có thể upload nhiều lần trước khi trigger

2. **Batch Ingest Strategy**
   - Accumulate courses trong staging folder
   - Khi [RUN INGEST] được click:
     - Move `pending/` → `processing/`
     - Trigger ETL pipeline (có thể async)
     - Move `processing/` → `archived/YYYY-MM-DD/`
   - Benefit: Ingest 1 lần cho nhiều files → tiết kiệm embedding overhead

3. **Incremental Update (Optional Phase 2)**
   - Không xóa toàn bộ KG (clear_existing=False)
   - Chỉ add new skills/courses, update existing as needed
   - Can enable later khi production

### 4.3 Components & Functionality

| Component | Action | Details |
|-----------|--------|---------|
| **Course Upload** | Drag-drop area | - Accept `.json` files <br> - Validate schema (have course_title, skills) <br> - Save to `data/staging/pending/` <br> - Show success/error toast |
| **Pending Queue** | Table viewer | - List all files in `staging/pending/` <br> - Columns: filename, upload time, # courses <br> - Auto-refresh every 3 sec |
| **Run Ingest Button** | Single click | - Move pending → processing <br> - Trigger ETL pipeline <br> - Show status monitor <br> - Disable button during ingest |
| **Status Monitor** | Real-time progress | - Polling `/api/v1/admin/pipeline/status` every 2 sec <br> - Show current phase (1-5) <br> - Progress % <br> - Duration/phase <br> - Completed time <br> - Success/Error message |

### 4.4 API Contracts (Admin)

```json
POST /api/v1/admin/courses/upload

Request:
{
  "course_data": [...json array...]
}

Response:
{
  "status": "saved",
  "file_id": "upload_20260415_143022",
  "courses_count": 45,
  "location": "data/staging/pending/upload_20260415_143022.json",
  "message": "45 courses queued for ingestion"
}

────────────────────────────────────────

GET /api/v1/admin/courses/queue

Response:
{
  "pending_files": [
    {
      "file_id": "upload_20260415_143022",
      "filename": "upload_20260415_143022.json",
      "upload_time": "2026-04-15T14:30:22Z",
      "courses_count": 45,
      "file_size_bytes": 102400
    }
  ],
  "total_pending_courses": 127
}

────────────────────────────────────────

POST /api/v1/admin/pipeline/run

Request:
{
  "clear_existing": false,            // Keep old KG, only add new
  "output_metrics_report": true       // Generate metrics JSON
}

Response:
{
  "run_id": "pipeline_run_20260415_143030",
  "status": "started",
  "message": "Ingestion pipeline started",
  "estimated_duration_sec": 180
}

────────────────────────────────────────

GET /api/v1/admin/pipeline/status?run_id=pipeline_run_20260415_143030

Response:
{
  "run_id": "pipeline_run_20260415_143030",
  "status": "running",                     // or "completed" | "failed"
  "current_phase": 2,                      // 1-load, 2-embed, 3-dedup, 4-es_index, 5-neo4j
  "phase_name": "Embedding skills",
  "progress_percent": 45,
  "started_at": "2026-04-15T14:30:30Z",
  "elapsed_time_sec": 90,
  "phase_durations": {
    "load": 5.2,
    "embed": 45.1
  },
  "metrics_report": null                   // populated when complete
}

────────────────────────────────────────

GET /api/v1/jds/list

Response:
{
  "jds": [
    {
      "jd_id": "jd_001",
      "title": "Senior Backend Engineer",
      "description": "5+ years Python/Go experience...",
      "category": "Technology"
    }
  ]
}
```

---

## V. Chiến Thuật Tối Ưu cho Ingest

### 5.1 Problem: Resource Usage

Quá trình build KG có các bước tốn resource:
- **Embedding**: Tính embedding cho mọi skill name (batch processing)
- **Dedup**: So sánh embedding/fuzzy (O(n²) nếu không optimize)
- **ES Indexing**: Bulk write vào Elasticsearch
- **Neo4j**: Add nodes + relationships + indexes

### 5.2 Solutions Implemented / Proposed

#### ✅ **Already in codebase:**
1. **Batch Processing** (embedding, ES ingestion)
   - Settings: `settings.embedding.batch_size`, `elasticsearch.batch_size`, `neo4j.batch_size`
   - Embedded trong `EmbeddingService`, `ElasticsearchIndexClient`, `Neo4jBatchClient`

2. **Dedup 2-pass Strategy** (efficient clustering)
   - Pass 1: Fuzzy matching (Jaro-Winkler) – fast, catches typos
   - Pass 2: Embedding similarity (cosine) – accurate, catches synonyms
   - Reduces dedup time từ O(n²) → O(n*k) where k = cluster size

3. **Metrics Reporting**
   - Pipeline output JSON report → analyze bottlenecks
   - Track input/output counts per phase

#### 🆕 **Propose for Frontend:**

1. **Incremental vs Full Ingest**
   - Checkbox: `clear_existing` (default: False)
   - False → add new courses, skip existing
   - True → full rebuild (use only for major updates)

2. **Queue Batching**
   - Accumulate uploads → single ingest run
   - Benefit: Share embedding model, dedup overhead across many files
   - Estimated 30% reduction in total time vs individual runs

3. **Async Processing**
   - Backend: trigger ETL in separate process (e.g., Celery, APScheduler)
   - Frontend: poll status endpoint → show progress
   - User không cần chờ UI bị block

4. **Smart Phase Scheduling** (Future)
   - Light phases (load, dedup): run immediately
   - Heavy phases (embed): batch multiple files → run at off-peak time (e.g., night)
   - Frontend: show "Scheduled for [time]" vs "Running now"

---

## VI. Data Flow & State Management

### 6.1 User Module Flow

```
CV Text (file/paste)
    │
    ├─→ [Backend extract text]
    │
    ├─→ Store in session (frontend cache)
    │
┌───┴─→ [User picks JD from list]
│
├─→ [Optional: add keywords]
│
├─→ [Click RECOMMEND]
│
└─→ POST /api/v1/recommendations/gap
    │
    ├─→ Gap Detection (embed CV+JD)
    ├─→ Skill Search (kNN in ES)
    ├─→ Rank Courses (Neo4j query)
    │
    └─→ Display Results
        - Courses sorted by coverage
        - Skill breakdown
```

### 6.2 Admin Module Flow

```
Courses (JSON files)
    │
    ├─→ [Drag-drop to staging]
    │
    ├─→ POST /api/v1/admin/courses/upload
    │
    ├─→ Save to data/staging/pending/
    │
    ├─→ Frontend auto-refresh queue display
    │
    ├─→ [Admin clicks RUN INGEST]
    │
    └─→ POST /api/v1/admin/pipeline/run
        │
        ├─→ Move pending/ → processing/
        │
        ├─→ Trigger ETL (async)
        │   - Phase 1: Load
        │   - Phase 2: Embed
        │   - Phase 3: Dedup
        │   - Phase 4: ES Index
        │   - Phase 5: Neo4j Build
        │
        ├─→ Frontend polls /api/v1/admin/pipeline/status
        │
        └─→ Move processing/ → archived/
            └─→ Show success + metrics
```

---

## VII. Directory Structure (Staging)

```
data/
├── staging/
│   ├── pending/        # Waiting for ingest
│   │   └── upload_20260415_143022.json
│   ├── processing/     # Currently ingesting
│   │   └── (empty until run)
│   └── archived/       # Completed runs
│       └── 2026-04-15/
│           └── upload_20260415_143022.json
├── neo4j/
├── elasticsearch/
└── ...
```

---

## VIII. Implementation Checklist

### Backend Endpoints (FastAPI)

- [ ] `POST /api/v1/admin/courses/upload` – bulk upload + save to staging
- [ ] `GET /api/v1/admin/courses/queue` – list pending files
- [ ] `POST /api/v1/admin/pipeline/run` – trigger ETL
- [ ] `GET /api/v1/admin/pipeline/status` – poll ingest progress
- [ ] `GET /api/v1/jds/list` – list available JDs (for user dropdown)
- [ ] Modify existing `POST /api/v1/recommendations/gap` to accept CV directly

### Frontend Components (React/Vue/Svelte)

- [ ] User: CV upload + text input
- [ ] User: JD selector (dropdown)
- [ ] User: Keywords input (optional)
- [ ] User: Recommend button + loading state
- [ ] User: Results display (course cards)
- [ ] Admin: Course upload area (drag-drop)
- [ ] Admin: Pending queue table (auto-refresh)
- [ ] Admin: Run Ingest button
- [ ] Admin: Status monitor (phase progress + duration)

### Infrastructure & Config

- [ ] Ensure `data/staging/pending/` directory exists
- [ ] Update `.env` with staging paths
- [ ] Configure batch sizes in `settings.yaml` (optimize for demo hardware)

---

## IX. Performance Targets (Demo)

| Metric | Target | Notes |
|--------|--------|-------|
| CV upload + parse | < 2 sec | Depends on file size |
| Gap detection + recommendation | 1-3 sec | With ES kNN search |
| Course upload (50 files) | < 1 sec | Just save to staging |
| Full ingest (100 new courses) | 2-5 min | Depends on batch size, embedding model |
| Status polling latency | < 500 ms | Real-time feedback |

---

## X. Demo Script (for Supervisor)

1. **User Flow** (2 min)
   - Upload sample CV (e.g., Python dev)
   - Select JD "Senior Backend Engineer"
   - Click Recommend
   - Show top 3 courses covering gaps

2. **Admin Flow** (3 min)
   - Upload 30 new course files (drag-drop)
   - Show pending queue
   - Click [RUN INGEST]
   - Watch progress from load → embed → neo4j
   - Show metrics report (# new skills, # new courses)
   - Verify courses appear in user recommendations

---

## XI. Future Enhancements (Phase 2)

- Scheduled ingest (run at specific times)
- Course update history (versioning)
- User feedback on recommendations (thumbs up/down)
- Analytics dashboard (most recommended skills, trending courses)
- Multi-language support (VN/EN)
- Batch download results (PDF report)

---

**Status**: Design ready for review. Proceed to implementation once approved.

---

## XII. Frontend Code Module Design

### 12.1 Muc tieu

Thiet ke module code cho frontend de:
- Code nhanh cho demo
- Tach biet ro User flow va Admin flow
- De mo rong thanh production sau nay

### 12.2 Tech Stack De Xuat

- Framework: React + TypeScript + Vite
- UI: Ant Design hoac MUI (chon 1 de dong bo)
- State/Data fetching: TanStack Query
- Routing: React Router
- File upload: react-dropzone
- Form handling: react-hook-form + zod

### 12.3 Cau truc thu muc frontend

```text
frontend/
├── src/
│   ├── app/
│   │   ├── App.tsx
│   │   ├── router.tsx
│   │   └── providers.tsx
│   │
│   ├── pages/
│   │   ├── user/
│   │   │   └── UserRecommendationPage.tsx
│   │   └── admin/
│   │       └── AdminIngestionPage.tsx
│   │
│   ├── features/
│   │   ├── user-recommendation/
│   │   │   ├── components/
│   │   │   │   ├── CvUploadPanel.tsx
│   │   │   │   ├── JdSelector.tsx
│   │   │   │   ├── KeywordInput.tsx
│   │   │   │   ├── RecommendButton.tsx
│   │   │   │   └── RecommendationResult.tsx
│   │   │   ├── hooks/
│   │   │   │   ├── useJdListQuery.ts
│   │   │   │   └── useRecommendMutation.ts
│   │   │   ├── schemas/
│   │   │   │   └── recommend.schema.ts
│   │   │   └── types.ts
│   │   │
│   │   └── admin-ingestion/
│   │       ├── components/
│   │       │   ├── CourseUploadDropzone.tsx
│   │       │   ├── PendingQueueTable.tsx
│   │       │   ├── PipelineRunButton.tsx
│   │       │   └── PipelineStatusPanel.tsx
│   │       ├── hooks/
│   │       │   ├── useUploadCoursesMutation.ts
│   │       │   ├── usePendingQueueQuery.ts
│   │       │   ├── useRunPipelineMutation.ts
│   │       │   └── usePipelineStatusPolling.ts
│   │       ├── schemas/
│   │       │   └── upload.schema.ts
│   │       └── types.ts
│   │
│   ├── shared/
│   │   ├── api/
│   │   │   ├── httpClient.ts
│   │   │   ├── recommendationApi.ts
│   │   │   └── adminApi.ts
│   │   ├── components/
│   │   │   ├── PageHeader.tsx
│   │   │   ├── ErrorState.tsx
│   │   │   ├── EmptyState.tsx
│   │   │   └── LoadingOverlay.tsx
│   │   ├── constants/
│   │   │   └── apiPaths.ts
│   │   └── utils/
│   │       ├── fileParser.ts
│   │       ├── date.ts
│   │       └── format.ts
│   │
│   ├── styles/
│   │   ├── tokens.css
│   │   └── global.css
│   │
│   └── main.tsx
├── index.html
└── package.json
```

### 12.4 Module Design - User Flow

#### A. Components

1. `CvUploadPanel`
- Nhan file `.pdf/.docx/.txt`
- Parse text local (neu co) hoac gui backend parse
- Day `cvText` len page state

2. `JdSelector`
- Goi `GET /api/v1/jds/list`
- Hien dropdown co search
- Tra ve `jdTitle` va `jdKeywords` mac dinh

3. `KeywordInput`
- User them keywords bo sung
- Chuan hoa thanh `string[]`

4. `RecommendButton`
- Trigger mutation `POST /api/v1/recommendations/gap`
- Disable khi request dang chay

5. `RecommendationResult`
- Render danh sach course da rank
- Hien thi `coverage_count`, `covered_gaps`

#### B. State shape

```ts
type UserRecommendationState = {
  cvText: string;
  selectedJdTitle: string;
  jdKeywords: string[];
  extraKeywords: string[];
  maxCourses: number;
};
```

#### C. API contract mapping

```ts
const payload = {
  cv_text: state.cvText,
  jd_title: state.selectedJdTitle,
  jd_keywords: [...state.jdKeywords, ...state.extraKeywords],
  max_courses: state.maxCourses,
};
```

### 12.5 Module Design - Admin Flow

#### A. Components

1. `CourseUploadDropzone`
- Drag-drop nhieu file json
- Validate format co `course_id/course_title/skills`
- Upload theo batch len `POST /api/v1/admin/courses/upload`

2. `PendingQueueTable`
- Query `GET /api/v1/admin/courses/queue`
- Refresh moi 3s
- Hien tong `total_pending_courses`

3. `PipelineRunButton`
- Trigger `POST /api/v1/admin/pipeline/run`
- Disable neu queue rong hoac run dang chay

4. `PipelineStatusPanel`
- Poll `GET /api/v1/admin/pipeline/status?run_id=...` moi 2s
- Hien phase, progress, elapsed, error

#### B. State shape

```ts
type AdminIngestionState = {
  currentRunId: string | null;
  isRunning: boolean;
  lastRunStatus: "idle" | "running" | "completed" | "failed";
  clearExisting: boolean;
};
```

#### C. Polling strategy

- Bat polling khi co `run_id`
- Dung polling khi `status` la `completed` hoac `failed`
- Sau khi xong: invalidate queue query de table cap nhat

### 12.6 Shared API Layer

`httpClient.ts`
- Tao axios instance
- Base URL tu `VITE_API_BASE_URL`
- Interceptor xu ly timeout, 4xx/5xx

`recommendationApi.ts`
- `getJdList(params)`
- `postGapRecommendation(payload)`

`adminApi.ts`
- `uploadCourses(payload)`
- `getPendingQueue()`
- `runPipeline(payload)`
- `getPipelineStatus(runId)`

### 12.7 Error Handling va UX Rules

- Loi upload file: show toast + giu lai file loi de admin sua
- Loi recommendation: show message ro nguyen nhan (missing CV/JD, timeout)
- Neu pipeline fail: hien `error_message` va cho phep re-run
- Moi action quan trong deu co loading state + disable button

### 12.8 Security va Validation (Demo level)

- Client-side validation file type va size
- Sanitize text input truoc khi gui API
- Khong render HTML raw tu response
- Gioi han kich thuoc file CV va course JSON

### 12.9 Ke hoach implement frontend

1. Scaffold app React + TypeScript
2. Dung khung routing voi 2 page: User/Admin
3. Implement shared api client + typed contracts
4. Implement User module end-to-end
5. Implement Admin module end-to-end
6. Hook polling va progress UI
7. Test happy path + common error path

### 12.10 Definition of Done (Frontend)

- User co the upload CV, chon JD, nhan recommendation
- Admin co the upload courses, xem queue, run ingest, theo doi status
- Khong co crash khi API tra loi hoac timeout
- UI responsive tren desktop va mobile
- Toan bo flow demo chay duoc voi backend hien tai
