#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TS="${TS:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="logs/dataset_analysis"
mkdir -p "$LOG_DIR"
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-cuda}"

echo "[1/2] Running embedding baseline benchmark..."
docker exec -e PYTHONPATH=/app/src -e EMBEDDING_DEVICE="$EMBEDDING_DEVICE" vietcv_data_factory python src/service_api/scripts/evaluate_serving_embedding_baseline.py \
  2>&1 | tee "$LOG_DIR/serving_baseline_${TS}.log"

echo "[2/2] Running designed-flow benchmark..."
docker exec -e PYTHONPATH=/app/src -e EMBEDDING_DEVICE="$EMBEDDING_DEVICE" vietcv_data_factory python src/service_api/scripts/evaluate_serving_designed_flow.py \
  2>&1 | tee "$LOG_DIR/serving_designed_${TS}.log"

echo "[3/3] Exporting detailed pairwise comparison workbook..."
docker exec -e PYTHONPATH=/app/src vietcv_data_factory python src/service_api/scripts/export_serving_pairwise_report.py \
  2>&1 | tee "$LOG_DIR/pairwise_report_${TS}.log"

echo "Done."
echo "Baseline log:  $LOG_DIR/serving_baseline_${TS}.log"
echo "Designed log:   $LOG_DIR/serving_designed_${TS}.log"
echo "Pairwise log:   $LOG_DIR/pairwise_report_${TS}.log"
echo "Baseline metrics:  data/processed/course_recommendation_metrics/embedding_baseline_metrics.json"
echo "Designed metrics:   data/processed/course_recommendation_metrics/designed_flow_metrics.json"
echo "Pairwise workbook:  data/processed/course_recommendation_metrics/baseline_vs_designed_pairwise.xlsx"