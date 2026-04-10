# Technical Guide – Course Recommendation System (v2)

## 1. Tổng quan kiến trúc

Hệ thống gồm hai process độc lập dùng chung thư viện `shared/`:

```
┌─────────────────────────────────────────────────────────┐
│                    Infrastructure                        │
│   Neo4j (7687)     Elasticsearch (9200)                 │
└────────┬────────────────────┬────────────────────────────┘
         │                    │
┌────────▼──────────┐  ┌──────▼───────────────────────────┐
│  data_factory     │  │         service_api               │
│  (ETL – offline)  │  │      (FastAPI – online)           │
│                   │  │                                   │
│ 1. Load JSON      │  │ POST /api/v1/recommendations/gap  │
│ 2. Embed skills   │  │   1. Gap detection (embed JD/CV)  │
│ 3. Dedup cluster  │  │   2. Skill search (ES kNN)        │
│ 4. Index → ES     │  │   3. Course lookup (Neo4j)        │
│ 5. Build KG Neo4j │  │                                   │
└───────────────────┘  └───────────────────────────────────┘
         ▲                          ▲
         └──────── shared/ ─────────┘
              models, db clients,
              EmbeddingService
```

---

## 2. Cấu trúc thư mục `src/`

```
src/
├── requirements.txt                   # tất cả dependencies
├── .env.example                       # template biến môi trường
│
├── shared/                            # thư viện dùng chung
│   ├── models/
│   │   ├── skill.py                   # CanonicalSkill
│   │   └── course.py                  # CourseSkill, CourseNode, TeachesEdge, RequiresEdge
│   ├── db/
│   │   ├── neo4j_client.py            # Neo4jClient + Neo4jBatchClient
│   │   └── es_client.py               # ElasticsearchClient + ElasticsearchIndexClient
│   ├── embeddings/
│   │   └── embedding_service.py       # EmbeddingService (SentenceTransformer wrapper)
│   └── utils/
│       └── logging_utils.py
│
├── data_factory/
│   ├── settings.py                    # Settings dataclasses + YAML loader
│   ├── config/
│   │   └── settings.yaml              # cấu hình ETL (threshold, model, neo4j, es)
│   ├── io/
│   │   └── course_skill_loader.py     # đọc JSON từ LLM → List[CourseSkill]
│   ├── services/
│   │   └── skill_dedup_service.py     # greedy cosine clustering → CanonicalSkill
│   ├── pipelines/
│   │   └── graph_build_pipeline.py    # pipeline 5-phase chính
│   └── scripts/
│       └── build_graph.py             # ← entry point duy nhất của data_factory
│
└── service_api/
    ├── config.py                      # Settings từ env vars (pydantic-settings)
    ├── dependencies.py                # lazy-singleton DI factory
    ├── main.py                        # FastAPI app entry point
    ├── models/
    │   ├── request.py                 # GapRecommendationRequest, SkillSearchRequest
    │   └── response.py                # GapRecommendationResponse, SkillResponse, ...
    ├── services/
    │   ├── gap_detection.py           # GapDetectionService
    │   ├── skill_search.py            # SkillSearchService
    │   └── course_recommendation.py   # CourseRecommendationService
    └── api/v1/
        ├── api.py                     # router aggregator
        └── endpoints/
            ├── health.py              # GET  /api/v1/health
            ├── skills.py              # GET  /api/v1/skills/search
            │                          # GET  /api/v1/skills/{id}
            └── recommendations.py     # POST /api/v1/recommendations/gap
```

---

## 3. Domain Models (`shared/models/`)

### `CanonicalSkill` — `shared/models/skill.py`

Skill đã được dedup, lưu trong Elasticsearch và Neo4j.

| Field            | Type          | Mô tả                                              |
|------------------|---------------|----------------------------------------------------|
| `skill_id`       | `str`         | SHA-256[:16] của `canonical_label` (lowercase)     |
| `canonical_label`| `str`         | Tên đại diện được chọn theo `canonical_strategy`   |
| `aliases`        | `List[str]`   | Các tên gốc từ LLM bị merge vào skill này          |
| `description`    | `str \| None` | Mô tả (từ đầu tiên trong cluster)                  |
| `category`       | `str \| None` | Danh mục khoá học nguồn                            |

### `CourseSkill` — `shared/models/course.py`

Skill raw chưa dedup, load từ file JSON (output của LLM).

| Field               | Type    | Mô tả                                |
|---------------------|---------|--------------------------------------|
| `course_id`         | `str`   | ID khoá học                          |
| `skill_name`        | `str`   | Tên skill (text tự do từ LLM)        |
| `skill_type`        | `str`   | `"outcome"` hoặc `"entry"`           |
| `description`       | `str`   | Mô tả kết quả học                    |
| `category`          | `str`   | Nhóm/danh mục                        |

Ngoài ra còn có `CourseNode`, `TeachesEdge`, `RequiresEdge` dùng để build Neo4j graph.

---

## 4. Shared Infrastructure (`shared/db/`, `shared/embeddings/`)

### `EmbeddingService` — `shared/embeddings/embedding_service.py`

Wrapper duy nhất cho SentenceTransformer, dùng cho cả ETL lẫn serving.

```python
from shared.embeddings.embedding_service import EmbeddingService

emb = EmbeddingService(
    model_name="Qwen/Qwen3-Embedding-0.6B",
    model_path=None,        # None → dùng HuggingFace; str → local path
    device="cuda",          # "cuda" | "cpu"
    batch_size=16,
    normalize=True,
)

vecs = emb.encode(["Python", "Machine Learning"])  # → np.ndarray (2, 1024)
vec  = emb.encode_single("Python")                 # → np.ndarray (1024,)
```

### `Neo4jClient` / `Neo4jBatchClient` — `shared/db/neo4j_client.py`

| Class               | Dùng ở           | Chức năng                              |
|---------------------|------------------|----------------------------------------|
| `Neo4jClient`       | `service_api`    | `.query(cypher, params)` → read only   |
| `Neo4jBatchClient`  | `data_factory`   | Kế thừa `Neo4jClient` + batch MERGE   |

Methods của `Neo4jBatchClient`:
- `create_indexes()` — tạo constraints `skill_id`, `course_id`
- `clear_graph()` — xóa toàn bộ graph
- `batch_merge_skills(skills)` — MERGE :Skill nodes
- `batch_merge_courses(courses)` — MERGE :Course nodes
- `batch_merge_teaches(edges)` — MERGE (:Course)-[:TEACHES]->(:Skill)
- `batch_merge_requires(edges)` — MERGE (:Course)-[:REQUIRES]->(:Skill)

### `ElasticsearchClient` / `ElasticsearchIndexClient` — `shared/db/es_client.py`

| Class                        | Dùng ở          | Chức năng                            |
|------------------------------|-----------------|--------------------------------------|
| `ElasticsearchClient`        | `service_api`   | `vector_search()`, `text_search()`   |
| `ElasticsearchIndexClient`   | `data_factory`  | Kế thừa + `ensure_index()`, `bulk_index()` |

Schema Elasticsearch index `course_skills`:

```json
{
  "skill_id":        "keyword",
  "canonical_label": "text + keyword",
  "aliases":         "text",
  "description":     "text",
  "category":        "keyword",
  "vector":          "dense_vector (dim=1024, cosine)"
}
```

---

## 5. Data Factory — Logic chi tiết

### 5.1 Settings — `data_factory/settings.py`

Load từ `data_factory/config/settings.yaml`. Hỗ trợ expand `${ENV_VAR}`.

Key configs:

```yaml
deduplication:
  threshold: 0.92        # cosine similarity để merge hai skill (0–1)
  batch_size: 512        # batch khi tính cosine matrix trong RAM
  canonical_strategy: first  # first | longest | most_common

embedding:
  model_path: null       # null → HuggingFace; đường dẫn → local fine-tuned model
  device: cuda
  batch_size: 16
```

### 5.2 `GraphBuildPipeline` — `data_factory/pipelines/graph_build_pipeline.py`

```
Phase 1 – Load
  └─ course_skill_loader.load_course_skills(course_catalog_dir)
     Đọc *.json trong data/Data_Courses_Json/**
     Extract "skill_outcomes" (type=outcome) và "entry_requirements.minimum_entry_skills" (type=entry)
     → List[CourseSkill]

Phase 2 – Embed
  └─ EmbeddingService.encode([cs.skill_name for cs in course_skills])
     → np.ndarray (N, 1024)

Phase 3 – Dedup  [SkillDedupService]
  └─ Greedy cosine clustering:
       for each unassigned skill i:
         tạo cluster mới với i là representative
         tìm j > i còn chưa gán có sim(i,j) ≥ threshold → ghép vào cluster
       pick canonical label theo strategy
       skill_id = SHA-256[:16](canonical_label)
     → List[CanonicalSkill], merge_map {raw_label→skill_id}

Phase 4 – Index ES
  └─ ElasticsearchIndexClient.ensure_index()
     encode(canonical_labels) → vectors
     bulk_index(canonical_skill + vector)

Phase 5 – Build Neo4j KG
  └─ Neo4jBatchClient.create_indexes()
     batch_merge_skills(canonical_skills)
     build_graph_edges(course_skills, merge_map)
       → CourseNode, TeachesEdge (type=outcome), RequiresEdge (type=entry)
     batch_merge_courses / batch_merge_teaches / batch_merge_requires
```

### 5.3 `SkillDedupService` — `data_factory/services/skill_dedup_service.py`

```python
dedup = SkillDedupService(config)
canonical_skills, merge_map = dedup.deduplicate(
    labels=["Python", "python programming", "Lập trình Python", ...],
    embeddings=np.ndarray,        # (N, D) normalized
    descriptions=[...],           # optional
    categories=[...],             # optional
)
# merge_map = {"Python": "abc123", "python programming": "abc123", ...}
```

**Lưu ý về threshold:**
- Threshold cao (0.95+) → ít merge → nhiều skill nodes trong KG → nguy cơ phân mảnh
- Threshold thấp (0.85-) → merge nhiều → có thể gộp nhầm skill khác nghĩa
- Khuyến nghị: bắt đầu với 0.92, quan sát sau khi build

---

## 6. Service API — Logic chi tiết

### 6.1 Config — `service_api/config.py`

Load 100% từ environment variables (`.env` file khi develop).

| Env var                     | Default                        | Mô tả                              |
|-----------------------------|--------------------------------|------------------------------------|
| `NEO4J_URI`                 | `bolt://neo4j:7687`            |                                    |
| `ELASTICSEARCH_HOSTS`       | `http://elasticsearch:9200`    | comma-separated nếu nhiều host     |
| `ELASTICSEARCH_INDEX`       | `course_skills`                | phải khớp với data_factory         |
| `EMBEDDING_MODEL_NAME`      | `Qwen/Qwen3-Embedding-0.6B`    |                                    |
| `EMBEDDING_MODEL_PATH`      | `None`                         | override bằng local path           |
| `EMBEDDING_DEVICE`          | `cpu`                          | `cuda` nếu có GPU                  |
| `GAP_SIMILARITY_THRESHOLD`  | `0.80`                         | ngưỡng gap detection               |
| `SKILL_SEARCH_LIMIT`        | `3`                            | canonical skills tìm per gap       |

### 6.2 Dependency Injection — `service_api/dependencies.py`

Tất cả clients và services được khởi tạo lazy (lần đầu được gọi) và cache suốt lifetime process.

```
get_neo4j()        → Neo4jClient (singleton)
get_es()           → ElasticsearchClient (singleton)
get_embedding()    → EmbeddingService (singleton, load model một lần)
  ↓
get_gap_detection_service()    → GapDetectionService
get_skill_search_service()     → SkillSearchService
get_recommendation_service()   → CourseRecommendationService
```

### 6.3 Request Flow — `POST /api/v1/recommendations/gap`

**Input:**
```json
{
  "jd_skills": ["Python", "Machine Learning", "SQL", "Communication"],
  "cv_skills": ["Python", "Django", "PostgreSQL"],
  "max_courses": 10,
  "gap_threshold": null
}
```

**Bước 1 — `GapDetectionService.find_gaps()`**  
- Embed `jd_skills` → matrix `(M, D)`  
- Embed `cv_skills` → matrix `(K, D)`  
- Tính `sim_matrix = jd_vecs @ cv_vecs.T` → `(M, K)`  
- `max_sim[i] = max(sim_matrix[i])` — mức độ "có trong CV"  
- `gaps = [jd_skills[i] for i if max_sim[i] < threshold]`
- **Kết quả:** `["Machine Learning", "Communication"]`

**Bước 2 — `SkillSearchService.search_batch()`**  
- Embed từng gap  
- KNN search ES `course_skills` index  
- Trả về canonical skill\_ids tương ứng  
- **Kết quả:** `{gap_name → [CanonicalSkill dicts]}`

**Bước 3 — `CourseRecommendationService.recommend_for_gaps()`**  
```cypher
MATCH (c:Course)-[:TEACHES]->(s:Skill)
WHERE s.skill_id IN $ids
RETURN c.course_id, c.course_title, c.category,
       collect(DISTINCT s.skill_id) AS covered_skill_ids
ORDER BY size(collect(DISTINCT s.skill_id)) DESC
LIMIT $limit
```
- Rank theo số gaps được cover  
- **Kết quả:** ordered list of courses

**Output:**
```json
{
  "jd_skill_count": 4,
  "cv_skill_count": 3,
  "raw_gaps": ["Machine Learning", "Communication"],
  "gap_skills": [
    {"skill_id": "abc123", "label": "Machine Learning", "category": "AI"}
  ],
  "recommended_courses": [
    {
      "course_id": "IT3190",
      "course_title": "Học máy",
      "covered_gaps": ["abc123"],
      "coverage_count": 1
    }
  ]
}
```

---

## 7. Knowledge Graph Schema (Neo4j)

### Node Labels

**`:Skill`**
```
skill_id:        String  (UNIQUE, SHA-256[:16] của canonical_label)
canonical_label: String
aliases:         List<String>
description:     String
category:        String
```

**`:Course`**
```
course_id:    String  (UNIQUE)
course_title: String
category:     String
source_file:  String
```

### Relationships

```
(:Course)-[:TEACHES]->(:Skill)
  skill_type: "outcome"
  source:     đường dẫn file JSON nguồn

(:Course)-[:REQUIRES]->(:Skill)
  skill_type: "entry"
  source:     đường dẫn file JSON nguồn
```

### Ví dụ query hữu ích

```cypher
-- Tìm khoá học dạy 1 skill
MATCH (c:Course)-[:TEACHES]->(s:Skill {canonical_label: "Machine Learning"})
RETURN c.course_title, c.category

-- Khoá học cover nhiều gap nhất
MATCH (c:Course)-[:TEACHES]->(s:Skill)
WHERE s.skill_id IN ["abc123", "def456"]
RETURN c.course_title, count(s) AS covered ORDER BY covered DESC

-- Xem tất cả skills của 1 khoá học
MATCH (c:Course {course_id: "IT3190"})-[r]->(s:Skill)
RETURN type(r), s.canonical_label, s.category
```

---

## 8. Khởi động hệ thống

### Bước 1 – Khởi động Infrastructure Services

```bash
# Từ thư mục gốc dự án
docker compose up -d elasticsearch neo4j

# Kiểm tra
curl http://localhost:9200/_cluster/health   # Elasticsearch
curl http://localhost:7687                   # Neo4j (hoặc vào http://localhost:7474)
```

> Neo4j credentials mặc định: `neo4j / password123`  
> Neo4j Browser: http://localhost:7474

### Bước 2 – Cài Python dependencies

```bash
cd /root/courses_rec
source venv/bin/activate        # hoặc tạo venv mới: python -m venv venv

pip install -r src/requirements.txt
```

### Bước 3 – Cấu hình môi trường

```bash
cp src/.env.example src/.env
# Chỉnh sửa src/.env nếu cần (credentials, model path, v.v.)
```

Nếu dùng local fine-tuned model:
```bash
# Trong src/.env
EMBEDDING_MODEL_PATH=models/qwen_embedding_finetuned
EMBEDDING_DEVICE=cuda
```

### Bước 4 – Build Knowledge Graph (data_factory)

```bash
cd /root/courses_rec
source venv/bin/activate

# Chạy pipeline (lần đầu hoặc khi dữ liệu thay đổi)
PYTHONPATH=src python src/data_factory/scripts/build_graph.py

# Giữ graph cũ, chỉ thêm mới (không xóa trước)
PYTHONPATH=src python src/data_factory/scripts/build_graph.py --no-clear

# Dùng config file khác
PYTHONPATH=src python src/data_factory/scripts/build_graph.py --config path/to/settings.yaml
```

**Thời gian chạy ước tính** (tùy số lượng khoá học + GPU):
- Embedding ~500 skills: 2–5 phút (CPU) / <1 phút (GPU)
- Build Neo4j graph: <30 giây

### Bước 5 – Khởi động API

```bash
cd /root/courses_rec
source venv/bin/activate

PYTHONPATH=src uvicorn service_api.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --reload \
  --env-file src/.env
```

Swagger UI: http://localhost:8000/docs  
Health check: http://localhost:8000/api/v1/health

---

## 9. Chạy bằng Docker Compose (Production)

Cần cập nhật `docker-compose.yml` để trỏ vào `src/`:

```yaml
# service api
command: >
  sh -c "PYTHONPATH=src uvicorn service_api.main:app --host 0.0.0.0 --port 8000"
volumes:
  - ./src:/app/src

# data_factory
command: >
  sh -c "PYTHONPATH=src python src/data_factory/scripts/build_graph.py"
environment:
  - PYTHONPATH=/app/src
```

Khởi động toàn bộ:
```bash
docker compose up -d elasticsearch neo4j
docker compose run --rm data_factory     # build KG một lần
docker compose up -d api                 # start API
```

---

## 10. Điều chỉnh tham số quan trọng

### Dedup threshold (ảnh hưởng lớn nhất đến chất lượng KG)

Trong `src/data_factory/config/settings.yaml`:

```yaml
deduplication:
  threshold: 0.92          # ← điều chỉnh tại đây
  canonical_strategy: first  # first | longest | most_common
```

| Threshold | Hành vi                          | Khi nào dùng                        |
|-----------|----------------------------------|-------------------------------------|
| 0.95+     | Gộp ít, nhiều skill nodes        | Dữ liệu sạch, ít biến thể           |
| 0.90–0.93 | Cân bằng (khuyến nghị)           | Mặc định                            |
| 0.85–     | Gộp nhiều, risk nhầm             | Muốn KG nhỏ gọn, dữ liệu nhiều noise|

### Gap detection threshold

Trong `src/.env` hoặc truyền qua request:

```bash
GAP_SIMILARITY_THRESHOLD=0.80   # tăng → phát hiện nhiều gap hơn
```

Hoặc override per-request:
```json
{ "jd_skills": [...], "cv_skills": [...], "gap_threshold": 0.75 }
```
