from __future__ import annotations

from pathlib import Path

import pandas as pd


BEFORE = {
    1: {"accuracy": 0.6415, "mrr": 0.6415, "precision": 0.6415, "recall": 0.6415, "ndcg": 0.6415, "map": 0.6415},
    3: {"accuracy": 0.7986, "mrr": 0.7119, "precision": 0.2662, "recall": 0.7986, "ndcg": 0.7342, "map": 0.7119},
    5: {"accuracy": 0.8487, "mrr": 0.7234, "precision": 0.1697, "recall": 0.8487, "ndcg": 0.7549, "map": 0.7234},
    10: {"accuracy": 0.8896, "mrr": 0.7290, "precision": 0.0890, "recall": 0.8896, "ndcg": 0.7683, "map": 0.7290},
    20: {"accuracy": 0.9200, "mrr": 0.7311, "precision": 0.0460, "recall": 0.9200, "ndcg": 0.7759, "map": 0.7311},
}

AFTER = {
    1: {"accuracy": 0.7272, "mrr": 0.7272, "precision": 0.7272, "recall": 0.7272, "ndcg": 0.7272, "map": 0.7272},
    3: {"accuracy": 0.8936, "mrr": 0.8023, "precision": 0.2979, "recall": 0.8936, "ndcg": 0.8258, "map": 0.8023},
    5: {"accuracy": 0.9341, "mrr": 0.8116, "precision": 0.1868, "recall": 0.9341, "ndcg": 0.8426, "map": 0.8116},
    10: {"accuracy": 0.9673, "mrr": 0.8162, "precision": 0.0967, "recall": 0.9673, "ndcg": 0.8535, "map": 0.8162},
    20: {"accuracy": 0.9839, "mrr": 0.8175, "precision": 0.0492, "recall": 0.9839, "ndcg": 0.8578, "map": 0.8175},
}


def main() -> None:
    metrics = ["accuracy", "mrr", "precision", "recall", "ndcg", "map"]
    long_rows = []
    summary_rows = []

    for k in [1, 3, 5, 10, 20]:
        summary_row = {"k": f"@{k}"}
        for metric in metrics:
            before = BEFORE[k][metric]
            after = AFTER[k][metric]
            delta = after - before
            pct_change = (delta / before * 100.0) if before else 0.0

            long_rows.append(
                {
                    "k": f"@{k}",
                    "metric": metric,
                    "before_train": before,
                    "after_train": after,
                    "delta": delta,
                    "pct_change_%": pct_change,
                }
            )

            summary_row[f"{metric}_before"] = before
            summary_row[f"{metric}_after"] = after
            summary_row[f"{metric}_delta"] = delta

        summary_rows.append(summary_row)

    out_dir = Path("data/processed/mapping_metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "embedding_metrics_before_vs_after_train.xlsx"

    with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="summary", index=False)
        pd.DataFrame(long_rows).to_excel(writer, sheet_name="long_format", index=False)

    print(out_file)


if __name__ == "__main__":
    main()