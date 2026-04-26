# Metrics Improvement Execution Plan

## Objective
Improve designed-flow recommendation metrics (especially hit@10 and recall@10) by fixing semantic skill matching quality before heavy retraining.

## Current Baseline Snapshot (from latest files)
- Baseline (embedding) @10:
  - hit_rate: 0.96
  - recall: 0.8045
- Designed current hard config @10:
  - hit_rate: 0.79
  - recall: 0.5442
  - config: top1_gate=ON, top1_min=0.87, margin=0.05, raw_limit=10
- Designed best known config (historical ablation) @10:
  - hit_rate: 0.83
  - recall: 0.5595
  - config: top1_gate=OFF, margin=0.08

## Success Targets (incremental)
- Stage 1 target: designed hit@10 >= 0.83 (recover known best).
- Stage 2 target: designed hit@10 >= 0.86 and recall@10 >= 0.60.
- Stage 3 target: reduce fail pairs with rank=-1 by >= 40% from current.

## Guardrail Metrics to Track Every Iteration
- hit@1, hit@3, hit@5, hit@10
- recall@10
- fail pair count
- GT rank=-1 count (out-of-candidate-pool)
- cross-domain mismatch count in gap->canonical top1

## Workstream A: Fast Config Optimization (no code change)
### A1. Recover known best config on current index
- Set:
  - SKILL_SEARCH_RAW_LIMIT=10
  - SKILL_SEARCH_SCORE_MARGIN=0.08
  - DESIGNED_DISABLE_TOP1_GATE=1
  - DESIGNED_ENABLE_ENRICH=1
- Run full benchmark and export pairwise workbook.
- Exit criteria:
  - if hit@10 improves by >= +0.03, lock this as temporary serving config.

### A2. Candidate breadth sweep
- Try RAW_LIMIT in {10, 15, 20} with same gate-off/margin=0.08.
- Keep config that maximizes hit@10 then recall@10.

### A3. Margin sweep around winner
- Try margin in {0.06, 0.08, 0.10}.
- Keep best by hit@10, break ties with recall@10.

## Workstream B: Semantic Query Improvements (small code change)
### B1. Query normalization for gap text
- Normalize gap query to capability form:
  - current: includes "missing/lacks/insufficient"
  - target: neutral capability phrasing
- Keep role/jd keywords as context, but lower dominance of deficit wording.

### B2. Domain anchors and alias expansion
- Add alias map for high-impact technical terms:
  - AWS, EC2, ECS/EKS, BigQuery, Redshift, Kubernetes, Docker, SQL, MongoDB, Python, Java
- Inject aliases to query text at search time.

### B3. Generic-label penalty in selection
- Penalize top1 canonical labels that are cross-domain/generic when gap has technical signals.

## Workstream C: GT Quality for Evaluation Reliability
### C1. Maintain two evaluation tracks
- raw GT metrics (backward comparable)
- cleaned GT metrics (semantic reliability)

### C2. Error bucket dashboard
- semantic drift
- gate drop
- low-score mismatch
- generic-label capture

## Iteration Execution Protocol
For each iteration:
1. Run benchmark.
2. Save outputs with unique suffix.
3. Parse metrics and compare to previous best.
4. Parse rank_explain_failures for rank=-1 counts.
5. Update this file's progress log.
6. Continue only if measurable gain or new hypothesis.

## Progress Log
- 2026-04-19: Plan created.
- 2026-04-19: Completed A1 (`DESIGNED_DISABLE_TOP1_GATE=1`, `margin=0.08`, `raw_limit=10`, suffix=`a1_gateoff_margin008`).
  - Baseline @10: hit_rate=0.96, recall=0.8045.
  - Designed @10: hit_rate=0.82, recall=0.56.
  - Delta vs current hard config (0.79 / 0.5442): hit_rate +0.03, recall +0.0158.
  - Stage 1 target (>=0.83 hit@10) not reached yet, but A1 meets temporary gain threshold (+0.03).
  - Artifacts:
    - `data/processed/course_recommendation_metrics/designed_flow_groundtruth_gaps_metrics_a1_gateoff_margin008.json`
    - `data/processed/course_recommendation_metrics/designed_flow_groundtruth_gaps_comparison_a1_gateoff_margin008.xlsx`
- 2026-04-19: Fixed Elasticsearch Python client mismatch in `vietcv_data_factory` (`9.3.0` -> `8.11.0`).
  - Verified designed-flow benchmark runs successfully again with `--disable-top1-gate`.
  - Validation run (`suffix=esfix_test`, `batch_size=1`, `raw_limit=10`, `margin=0.08`) produced:
    - Designed @10: hit_rate=0.82, recall=0.56.
    - Designed @5: hit_rate=0.76, recall=0.485917.
    - Designed @3: hit_rate=0.71, recall=0.398667.
  - Artifacts:
    - `data/processed/course_recommendation_metrics/designed_flow_groundtruth_gaps_metrics_esfix_test.json`
    - `data/processed/course_recommendation_metrics/designed_flow_groundtruth_gaps_comparison_esfix_test.xlsx`
- 2026-04-19: Added gap normalization + technical alias hints (`suffix=lexical_alias_v1`, `batch_size=1`, `raw_limit=10`, `margin=0.08`, gate off, enrich on).
  - Designed @1: hit_rate=0.46, precision=0.46, MRR=0.46, MAP=0.46.
  - Designed @3: hit_rate=0.72, recall=0.412, MRR=0.58, MAP=0.361667.
  - Designed @5: hit_rate=0.80, recall=0.507583, MRR=0.5985, MAP=0.374714.
  - Designed @10: hit_rate=0.86, recall=0.603333, MRR=0.607052, MAP=0.39878.
  - Compared with `esfix_test`, all major metrics improved; compared with baseline, designed is still below on every metric.
  - Artifacts:
    - `data/processed/course_recommendation_metrics/designed_flow_groundtruth_gaps_metrics_lexical_alias_v1.json`
    - `data/processed/course_recommendation_metrics/designed_flow_groundtruth_gaps_comparison_lexical_alias_v1.xlsx`
- 2026-04-19: Margin sweep on alias-normalized query (`suffixes=alias_margin_005`, `alias_margin_008`, `alias_margin_010`, `batch_size=1`, `raw_limit=10`, gate off, enrich on).
  - Best balanced designed config among tested runs: `margin=0.05`.
  - `alias_margin_005` metrics:
    - @1: hit_rate=0.47, MRR=0.47, MAP=0.47.
    - @3: hit_rate=0.73, recall=0.433, MRR=0.551471, MAP=0.344363.
    - @5: hit_rate=0.80, recall=0.526225, MRR=0.568382, MAP=0.368995.
    - @10: hit_rate=0.86, recall=0.598333, MRR=0.614552, MAP=0.393947.
  - `alias_margin_010` metrics:
    - @1: hit_rate=0.46, MRR=0.46, MAP=0.46.
    - @3: hit_rate=0.72, recall=0.412, MRR=0.581667, MAP=0.361111.
    - @5: hit_rate=0.80, recall=0.510083, MRR=0.600167, MAP=0.375714.
    - @10: hit_rate=0.86, recall=0.606667, MRR=0.608718, MAP=0.399319.
  - Conclusion: `margin=0.05` gives the best overall designed balance so far, but baseline still wins on all metrics.
- Next: Move from breadth-only tuning to top-rank optimization on the alias-normalized query (focus on @1/@3/MRR/MAP while preserving @10).

## Command Templates
### Full benchmark
EMBEDDING_MODEL_PATH=/app/models/qwen_embedding_finetuned EMBEDDING_DEVICE=cuda EMBEDDING_BATCH_SIZE=1 SKILL_SEARCH_RAW_LIMIT=<N> SKILL_TOP1_MIN_SCORE=0.87 SKILL_SEARCH_SCORE_MARGIN=<M> DESIGNED_ENABLE_ENRICH=1 DESIGNED_DISABLE_TOP1_GATE=<0|1> DESIGNED_OUTPUT_SUFFIX=<suffix> bash scripts/run_serving_benchmarks.sh

### Designed debug rank explain
docker exec -e PYTHONPATH=/app/src -e EMBEDDING_MODEL_PATH=/app/models/qwen_embedding_finetuned -e EMBEDDING_DEVICE=cuda -e EMBEDDING_BATCH_SIZE=1 -e SKILL_SEARCH_RAW_LIMIT=<N> -e SKILL_TOP1_MIN_SCORE=0.87 -e SKILL_SEARCH_SCORE_MARGIN=<M> -e DESIGNED_ENABLE_ENRICH=1 vietcv_data_factory python /app/src/service_api/scripts/evaluate_serving_designed_flow_groundtruth_gaps.py --enable-enrich --debug-rank-explain --output-suffix <suffix>
