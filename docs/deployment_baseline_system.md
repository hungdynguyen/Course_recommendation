# Baseline Deployment Guide

## 1. Mục tiêu

Tài liệu này mô tả cách deploy **baseline thực nghiệm** mà bạn đã benchmark trong repo này. Baseline ở đây là luồng **embedding-only retrieval**:

- nhập JD/CV hoặc skill gaps
- lấy text của gap làm query
- encode query bằng embedding model
- encode toàn bộ course texts bằng cùng embedding model
- tính cosine similarity giữa query vector và course vectors
- trả về top-k course có similarity cao nhất

Luồng này **không dùng Knowledge Graph, không Neo4j, không MySQL, không dùng designed flow**.

## 2. Luồng baseline thực sự

Theo script benchmark baseline, luồng thực tế là:

`groundtruth missing technical skills -> embedding query -> cosine search over embedded course texts`

Các thành phần chính:

1. **Embedding model**
   - Model trước train hoặc model sau train
   - Với bài toán deploy của bạn, ưu tiên model sau train nếu đã chứng minh tốt hơn

2. **Course catalog**
   - Lấy từ `data/Data_Courses_Filtered`
   - Mỗi course được ghép thành text từ title + skill outcomes + descriptions

3. **Precomputed course embeddings**
   - Encode toàn bộ course texts trước khi serve
   - Lưu vector matrix để dùng lại khi query

4. **Query encoder**
   - Encode skill gap hoặc JD text thành vector

5. **Cosine scorer**
   - So sánh query vector với course vectors bằng cosine similarity
   - Lấy top-k course

## 3. Kiến trúc deploy đề xuất

### 3.1 Thành phần tối thiểu

Để deploy baseline đúng như thí nghiệm, chỉ cần:

- **API service**: FastAPI để expose endpoint recommend
- **Embedding model**: sentence-transformers model
- **Course catalog file**: JSON course data
- **Course embedding cache**: file `.npy` hoặc `.npz`
- **Optional cache**: để tránh load lại vector mỗi request

### 3.2 Thành phần không cần

Không cần cho baseline này:

- Neo4j
- MySQL
- Elasticsearch
- Knowledge Graph
- graph build pipeline
- designed flow services

## 4. Repo components liên quan

### 4.1 API entrypoint

- [src/service_api/main.py](src/service_api/main.py)
- [src/service_api/api/v1/endpoints/recommendations.py](src/service_api/api/v1/endpoints/recommendations.py)
- [src/service_api/dependencies.py](src/service_api/dependencies.py)

### 4.2 Baseline evaluation script

- [src/service_api/scripts/evaluate_serving_embedding_baseline_groundtruth_gaps.py](src/service_api/scripts/evaluate_serving_embedding_baseline_groundtruth_gaps.py)

Script này phản ánh đúng baseline thí nghiệm:
- load labels/scenarios
- load course catalog
- embed course texts
- embed query text
- cosine similarity
- top-k ranking

### 4.3 Embedding service

- [src/shared/embeddings/embedding_service.py](src/shared/embeddings/embedding_service.py)
- [services/data_factory/src/embeddings/embedding_service.py](services/data_factory/src/embeddings/embedding_service.py)

### 4.4 Benchmark script

- [scripts/run_serving_benchmarks.sh](scripts/run_serving_benchmarks.sh)

## 5. Quyết định deploy

### 5.1 Chọn model nào

Nếu benchmark của bạn cho thấy baseline after-train tốt hơn, deploy bằng model fine-tuned local:

```yaml
embedding:
  provider: sentence_transformers
  model_name: Qwen/Qwen3-Embedding-0.6B
  model_path: /app/models/qwen_embedding_finetuned
  batch_size: 2
  device: cuda
  normalize: true
```

Nếu cần rollback nhanh, chỉ cần đặt:

```yaml
embedding:
  model_path: null
```

### 5.2 Cách serving baseline nên làm

Ở production, không nên encode toàn bộ course catalog mỗi request. Nên:

1. Precompute embeddings cho course catalog khi build/deploy
2. Load matrix embeddings lên RAM khi service start
3. Encode query text mỗi request
4. Tính cosine similarity qua dot product
5. Trả top-k

## 6. Mô hình runtime đề xuất

### 6.1 Offline build step

Chạy một job build để tạo cache:

- đọc `data/Data_Courses_Filtered`
- sinh `course_embeddings.npy`
- lưu `course_catalog_metadata.jsonl`

### 6.2 Online serving step

API server chỉ cần:

- load embedding model 1 lần khi start
- load course vectors 1 lần khi start
- serve `/recommend`

## 7. Kế hoạch code cần có

Nếu bạn muốn deploy baseline đúng cách, nên có 3 lớp code:

### 7.1 Course embedding builder

Một script offline, ví dụ:

- input: course catalog JSON
- output: course vectors + metadata

### 7.2 Baseline recommender service

Một service Python có interface kiểu:

- `recommend(query_text, top_k=10)`
- trả danh sách course + score

### 7.3 FastAPI endpoint

Ví dụ endpoint:

- `POST /api/v1/baseline/recommend`

Body:

```json
{
  "query_text": "Python SQL AWS",
  "top_k": 10
}
```

Response:

```json
{
  "query_text": "Python SQL AWS",
  "top_k": 10,
  "results": [
    {"course_id": "...", "course_title": "...", "score": 0.91}
  ]
}
```

## 8. Step-by-step setup

### Bước 1: Chuẩn bị môi trường

1. Cài Docker + Docker Compose.
2. Clone repo.
3. Tạo `.env`.
4. Mount model fine-tuned vào container nếu deploy after-train.

Ví dụ `.env`:

```env
EMBEDDING_MODEL_PATH=/app/models/qwen_embedding_finetuned
EMBEDDING_MODEL_NAME=Qwen/Qwen3-Embedding-0.6B
EMBEDDING_DEVICE=cuda
EMBEDDING_BATCH_SIZE=2
```

### Bước 2: Chuẩn bị data

Kiểm tra course catalog đã có sẵn:

- `data/Data_Courses_Filtered`

Nếu dùng bản after-train, đảm bảo model nằm ở:

- `/app/models/qwen_embedding_finetuned`

### Bước 3: Build course embeddings

Chạy một script build offline để encode course texts thành vectors.

Khuyến nghị lưu ra:

- `data/processed/baseline_cache/course_embeddings.npy`
- `data/processed/baseline_cache/course_metadata.jsonl`

### Bước 4: Chạy baseline benchmark để xác nhận chất lượng

Chạy script evaluation baseline:

```bash
docker exec -w /app/services/data_factory vietcv_data_factory python scripts/testing/evaluate_mapping_metrics.py
```

Hoặc nếu bạn muốn chạy đúng baseline serving benchmark:

```bash
docker exec -w /app/services/data_factory vietcv_data_factory python scripts/index_skills.py
```

Nhưng lưu ý: đây là pipeline ESCO embedding build, không phải baseline course retrieval. Với baseline deployment, bạn nên viết thêm script riêng cho course embedding cache.

### Bước 5: Implement baseline API

Tạo service baseline trong `src/service_api`:

- load `EmbeddingService`
- load course vectors
- compute cosine similarity
- return top-k

### Bước 6: Dockerize API

Build image cho API service và mount:

- model fine-tuned
- course cache
- config `.env`

### Bước 7: Smoke test

Test local endpoint:

- `GET /`
- `GET /docs`
- `POST /api/v1/baseline/recommend`

### Bước 8: Deploy production

- chạy `docker compose up -d`
- check logs
- check latency
- check top-k output

## 9. Cách tổ chức repository cho baseline deploy

Đề xuất cấu trúc thêm:

```text
src/
  service_api/
    services/
      baseline_recommender.py
    api/v1/endpoints/
      baseline.py
    scripts/
      build_baseline_cache.py
```

### baseline_recommender.py

Nên chứa:

- load model
- load course cache
- encode query
- score cosine
- rank top-k

### build_baseline_cache.py

Nên chứa:

- load course catalog
- compose text cho mỗi course
- encode toàn bộ course texts
- lưu embeddings + metadata

## 10. Những gì nên sửa trong code hiện tại

### 10.1 Bỏ dependency KG khỏi baseline deploy

Không cần dùng:

- `Neo4jClient`
- `MySQLClient`
- `SkillSearchService`
- `CourseRecommendationService`

cho baseline deployment.

### 10.2 Giữ `EmbeddingService`

EmbeddingService là thành phần cốt lõi cho baseline.

### 10.3 Precompute course embeddings

Đây là tối ưu quan trọng nhất để deploy baseline ổn định và nhanh.

## 11. Checklist triển khai

- [ ] Model fine-tuned có sẵn
- [ ] Course catalog có sẵn
- [ ] Course embeddings cache được build
- [ ] API load model 1 lần khi start
- [ ] API load course vectors 1 lần khi start
- [ ] Query encode và cosine ranking chạy ổn
- [ ] Response top-k đúng
- [ ] Latency chấp nhận được
- [ ] Có rollback về model gốc nếu cần

## 12. Gợi ý production tối thiểu

Nếu bạn muốn release nhanh, production baseline nên gồm:

- 1 FastAPI service
- 1 embedding model local
- 1 file course embedding cache
- 1 file course metadata

Tức là **không cần DB ngoài** nếu mục tiêu là chỉ serve recommendation baseline giống thí nghiệm.

## 13. Kết luận

Luồng deploy bạn muốn là một **baseline recommendation system độc lập**, không phải GraphRAG/KG system. Đúng kiến trúc là:

`text input -> fine-tuned embedding model -> cosine over precomputed course vectors -> top-k course`

Đây là luồng dễ deploy nhất, bám sát thí nghiệm baseline của bạn, và phù hợp nhất nếu bạn ưu tiên chất lượng đã benchmark tốt hơn designed flow.
