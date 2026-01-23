from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List

import pymysql
from elasticsearch import Elasticsearch

from src.settings import Settings, get_settings

LOGGER = logging.getLogger("data_factory.verify")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify ES + MySQL indexing health")
    parser.add_argument("--config", type=str, help="Path to settings YAML file")
    parser.add_argument(
        "--output",
        type=str,
        help="Path to output txt (default: data/processed/embeddings/verify_report)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5,
        help="Number of sample rows/docs to print",
    )
    return parser.parse_args()


def load_settings(config_path: str | None) -> Settings:
    if config_path:
        return Settings.load(Path(config_path))
    return get_settings()


def check_mysql(settings: Settings, sample_size: int = 5) -> Dict[str, Any]:
    cfg = settings.mysql
    if not cfg.enabled:
        return {"enabled": False}

    conn = pymysql.connect(
        host=cfg.host,
        port=cfg.port,
        user=cfg.username,
        password=cfg.password,
        database=cfg.database,
        charset=cfg.charset,
        connect_timeout=cfg.connect_timeout,
    )
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {cfg.esco_table}")
            count = cur.fetchone()[0]
            cur.execute(
                f"SELECT skill_id, preferred_label, description, skill_type, alternative_labels, broader_skill_ids, created_at "
                f"FROM {cfg.esco_table} LIMIT {int(sample_size)}"
            )
            columns = [col[0] for col in cur.description]
            sample_rows = cur.fetchall()
    finally:
        conn.close()

    return {
        "enabled": True,
        "count": count,
        "columns": columns,
        "sample": sample_rows,
    }


def check_es(settings: Settings) -> Dict[str, Any]:
    cfg = settings.elasticsearch
    client = Elasticsearch(cfg.hosts)

    exists = client.indices.exists(index=cfg.index)
    if not exists:
        return {"exists": False}

    count = client.count(index=cfg.index)["count"]
    sample_hits: List[dict] = []
    resp = client.search(index=cfg.index, size=5, query={"match_all": {}})
    for hit in resp.get("hits", {}).get("hits", []):
        sample_hits.append({
            "id": hit.get("_id"),
            "vector_len": len(hit.get("_source", {}).get("vector", []) or []),
        })

    return {
        "exists": True,
        "count": count,
        "sample": sample_hits,
    }


def write_report(path: Path, mysql_info: Dict[str, Any], es_info: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("=== Verify Report ===")

    lines.append("[MySQL]")
    if not mysql_info.get("enabled", False):
        lines.append("  status: disabled")
    else:
        lines.append(f"  count: {mysql_info.get('count')}")
        cols = mysql_info.get("columns", [])
        lines.append("  columns: " + ", ".join(cols))
        lines.append("  sample rows:")
        for row in mysql_info.get("sample", []):
            rendered = " | ".join(str(item) if item is not None else "NULL" for item in row)
            lines.append(f"    - {rendered}")

    lines.append("")
    lines.append("[Elasticsearch]")
    if not es_info.get("exists", False):
        lines.append("  status: missing index")
    else:
        lines.append(f"  count: {es_info.get('count')}")
        lines.append("  sample (id, vector_len):")
        for item in es_info.get("sample", []):
            lines.append(f"    - {item['id']} | {item['vector_len']}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("Wrote verify report to %s", path)


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)

    output_path = (
        Path(args.output)
        if args.output
        else Path(settings.paths.processed_embeddings_dir) / "verify_report.txt"
    )

    mysql_info = check_mysql(settings, sample_size=args.sample_size)
    es_info = check_es(settings)

    write_report(output_path, mysql_info, es_info)


if __name__ == "__main__":
    main()
