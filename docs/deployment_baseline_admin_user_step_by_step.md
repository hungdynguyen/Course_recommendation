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

## 2. Kiến trúc mục tiêu (baseline-only)

## 2.1 Thành phần bắt buộc

1. **API service (FastAPI)**
   - Cung cấp endpoint admin + user.
2. **Worker service (batch)**
   - Xử lý course upload theo queue/schedule.
   - Build lại course embedding cache.
3. **Scheduler**
   - Trigger job theo cron (ví dụ mỗi đêm 01:00).
4. **Storage tối thiểu**
   - Object/file storage cho file course upload.
   - Metadata DB (PostgreSQL hoặc SQLite) cho jobs, schedules, pairs.
5. **Embedding model service**
   - Model before/after train (ưu tiên after-train nếu benchmark tốt hơn).
6. **Course vector cache**
   - `course_embeddings.npy` + `course_metadata.jsonl`.

## 2.2 Thành phần không cần cho luồng này

1. Neo4j graph runtime
2. Elasticsearch skill index runtime cho baseline retrieval
3. KG pipeline

## 3. Data contracts

## 3.1 Input đã có sẵn

- CV parsed data (skills, keywords, text snippets)
- JD parsed data (skills, keywords, text snippets)

Giả định lưu trong DB/table hoặc file có ID ổn định:
- `cv_id`
- `jd_id`

## 3.2 Admin upload course format

Chuẩn hóa 1 schema JSON (chấp nhận list hoặc object):

```json
{
  "course_id": "CNTT1234",
  "title": "Machine Learning",
  "description": "...",
  "skill_outcomes": [
    {"skill_name": "Python", "outcome_description": "..."}
  ]
}
```

## 3.3 Recommendation output format

```json
{
  "cv_id": "cv_001",
  "jd_id": "jd_001",
  "top_k": 10,
  "results": [
    {
      "course_id": "CNTT1234",
      "course_title": "Machine Learning",
      "score": 0.9123
    }
  ]
}
```

## 4. Những gì cần setup

## 4.1 Environment

1. Python 3.10+ hoặc Docker runtime
2. `sentence-transformers`, `numpy`, `pandas`, `fastapi`, `uvicorn`
3. Queue/scheduler stack:
   - Option A: Celery + Redis + APScheduler
   - Option B (nhẹ): RQ + Redis + APScheduler
   - Option C (POC): thread worker + APScheduler

Khuyến nghị production: **Celery + Redis**.

## 4.2 Model paths

Đặt rõ 2 cấu hình:

- before-train: `Qwen/Qwen3-Embedding-0.6B`
- after-train: `/app/models/qwen_embedding_finetuned`

Env ví dụ:

```env
EMBEDDING_MODEL_NAME=Qwen/Qwen3-Embedding-0.6B
EMBEDDING_MODEL_PATH=/app/models/qwen_embedding_finetuned
EMBEDDING_DEVICE=cuda
EMBEDDING_BATCH_SIZE=8
```

## 4.3 Baseline artifacts

Tạo thư mục:

- `data/processed/baseline_cache/`

Artifacts cần có:

- `course_embeddings.npy`
- `course_metadata.jsonl`
- `cache_version.json`

## 5. Những gì cần code trong repo hiện tại

Các path dưới đây bám theo cấu trúc hiện tại trong `src/service_api`.

## 5.1 Baseline retrieval core

Tạo mới:

- [src/service_api/services/baseline_retriever.py](src/service_api/services)

Chức năng:

1. Load model 1 lần khi app startup.
2. Load `course_embeddings.npy` + metadata 1 lần.
3. Có method:
   - `recommend_from_text(query_text, top_k)`
   - `recommend_from_skills(skills, top_k)`
4. Score bằng cosine/dot product trên normalized vectors.

## 5.2 Course embedding builder (offline/batch)

Tạo mới script:

- [src/service_api/scripts/build_baseline_course_cache.py](src/service_api/scripts)

Chức năng:

1. Đọc course data từ staging/catalog.
2. Chuẩn hóa text input cho từng course.
3. Encode batch.
4. Ghi `course_embeddings.npy`, `course_metadata.jsonl`.
5. Update `cache_version.json`.

## 5.3 Admin APIs

Mở rộng endpoint admin hiện có (đang có demo tại [src/service_api/api/v1/endpoints/admin_demo.py](src/service_api/api/v1/endpoints/admin_demo.py)) thành production endpoints:

1. `POST /api/v1/admin/courses/upload`
2. `GET /api/v1/admin/courses/queue`
3. `POST /api/v1/admin/pipeline/run`
4. `POST /api/v1/admin/pipeline/schedule`
5. `GET /api/v1/admin/pipeline/status`

Tách service xử lý:

- [src/service_api/services/admin_ingest_service.py](src/service_api/services)

## 5.4 User APIs

Tạo endpoint user baseline:

- [src/service_api/api/v1/endpoints/baseline_recommendations.py](src/service_api/api/v1/endpoints)

API chính:

1. `POST /api/v1/baseline/recommend`
   - input: `jd_id`, `cv_id`, `top_k`
2. `GET /api/v1/baseline/recommend/{request_id}` (optional audit)

Logic:

1. Lấy parsed JD/CV từ store.
2. Build query text từ missing skills hoặc JD-required minus CV-skills.
3. Gọi `baseline_retriever`.

## 5.5 Pair resolver (CV/JD)

Tạo service:

- [src/service_api/services/pair_resolver.py](src/service_api/services)

Chức năng:

1. Resolve `cv_id`, `jd_id` -> parsed payload.
2. Validate pair tồn tại.
3. Trả structured skills cho recommender.

## 5.6 LLM endpoint integration

Tạo adapter:

- [src/service_api/services/llm_client.py](src/service_api/services)

Dùng cho:

1. Parse/enrich course upload (admin pipeline).
2. Fallback khi dữ liệu course thiếu trường.

## 6. Scheduler + Batch design

## 6.1 Job states

Một job pipeline nên có trạng thái:

- `queued`
- `running`
- `failed`
- `completed`

Lưu metadata:

- `run_id`
- `started_at`
- `finished_at`
- `input_files`
- `output_cache_version`
- `log_tail`

## 6.2 Batch flow

1. Admin upload JSON files.
2. Files vào `staging/pending`.
3. Scheduler trigger job.
4. Worker đọc batch, validate schema.
5. Gọi LLM endpoint (nếu cần enrich).
6. Build baseline cache.
7. Atomic swap cache (đổi symlink/current pointer).
8. Mark completed + notify frontend.

## 7. Step-by-step triển khai

## Step 1: Chuẩn bị branch và config

1. Tạo branch `deploy-baseline-admin-user`.
2. Tạo file env cho baseline-only.
3. Cấu hình model after-train mặc định.

## Step 2: Tạo baseline cache builder

1. Implement [src/service_api/scripts/build_baseline_course_cache.py](src/service_api/scripts).
2. Chạy script trên sample data.
3. Verify có `course_embeddings.npy` + metadata.

## Step 3: Implement baseline retriever

1. Implement [src/service_api/services/baseline_retriever.py](src/service_api/services).
2. Viết unit tests:
   - load cache
   - recommend top-k
   - deterministic ordering khi score bằng nhau

## Step 4: Implement user recommendation API

1. Tạo [src/service_api/api/v1/endpoints/baseline_recommendations.py](src/service_api/api/v1/endpoints).
2. Add router vào API v1.
3. Tạo request/response models tương ứng.
4. Test API bằng 5-10 pair CV/JD thực tế.

## Step 5: Implement admin ingest API + queue

1. Hardening từ demo service hiện tại:
   - [src/service_api/services/admin_ingest_demo.py](src/service_api/services/admin_ingest_demo.py)
2. Tách thành production service với persistent queue.
3. Add schedule endpoint.
4. Add audit logs.

## Step 6: Implement scheduler/worker

1. Setup Redis + Celery worker.
2. Tạo task `build_course_cache_task`.
3. Tạo cron schedule (ví dụ hàng ngày 01:00).
4. Test manual trigger + scheduled trigger.

## Step 7: Frontend admin

Admin UI cần:

1. Drag-drop upload component.
2. Queue table (pending/running/completed/failed).
3. Run now button.
4. Schedule config form.
5. Log/status panel.

## Step 8: Frontend user

User UI cần:

1. Dropdown chọn `cv_id`.
2. Dropdown chọn `jd_id`.
3. Nút Recommend.
4. Result table: rank, course_id, title, score.
5. Optional: explanation card (matched gaps/skills).

## Step 9: Docker deployment

Tối giản services:

1. `api`
2. `worker`
3. `redis`
4. (optional) `db` cho metadata

Không cần deploy Neo4j/MySQL/Elasticsearch cho baseline-only runtime.

## Step 10: Acceptance checklist

- [ ] Upload course qua admin UI thành công.
- [ ] Job batch chạy theo lịch thành công.
- [ ] Cache mới được swap không downtime.
- [ ] User chọn cv/jd nhận top-k recommendation.
- [ ] Latency endpoint trong ngưỡng mục tiêu.
- [ ] Logs/audit đủ để truy vết.
- [ ] Rollback cache version hoạt động.

## 8. Chạy benchmark và release gate

Dùng script benchmark hiện có để so sánh trước release:

- [scripts/run_serving_benchmarks.sh](scripts/run_serving_benchmarks.sh)

Release gate đề xuất:

1. Hit@1/3/5/10 không giảm so với mốc baseline đã chốt.
2. MRR@10 và nDCG@10 không giảm quá ngưỡng cho phép.
3. Smoke test API pass.

## 9. Rollout strategy

1. Deploy internal staging trước.
2. Chạy shadow traffic với một phần request.
3. So sánh online metrics baseline-old vs baseline-new.
4. Promote production sau 24-72h ổn định.

## 10. Kết luận

Để sẵn sàng deploy theo đúng mong muốn của bạn, trọng tâm là:

1. Chuẩn hóa luồng **baseline-only**.
2. Hoàn thiện 2 interface **admin** và **user**.
3. Xây batch pipeline có schedule + cache versioning.
4. Tách hoàn toàn khỏi KG runtime để hệ thống nhẹ và đúng với thí nghiệm baseline.
