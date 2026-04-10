#!/usr/bin/env python3
"""
Verify False Negatives using LLM (Gemini).

Loads the pre-computed false-negative Excel (or recomputes from dataset),
takes the top 100 samples where pos_score is lowest relative to neg_score,
calls Gemini to judge each case, and outputs an Excel with two extra columns:
  - false_negative  : True / False
  - comment         : LLM explanation
Usage:
    python testing/verify_false_negatives_llm.py
    python testing/verify_false_negatives_llm.py --top 50
    python testing/verify_false_negatives_llm.py --input logs/dataset_analysis/false_negatives_thresh0p00.xlsx
    python testing/verify_false_negatives_llm.py --dataset data/processed/training_dataset/train_dataset_80.json --recompute
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from tqdm.asyncio import tqdm as async_tqdm

load_dotenv()

# ============================================================
# Config
# ============================================================
GEMINI_API_KEY       = os.getenv("GEMINI_API_KEY", "")
LLM_MODEL            = "gemini-2.5-pro"
MAX_CONCURRENT       = 4
MAX_RETRIES          = 3
DEFAULT_TOP          = 300
DEFAULT_INPUT_XLSX   = "logs/dataset_analysis/false_negatives_thresh0p00.xlsx"
DEFAULT_DATASET_JSON = "data/processed/training_dataset/train_dataset_80.json"
OUTPUT_DIR           = Path("logs/dataset_analysis")

# JSON schema for structured LLM output
ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "is_false_negative": {
            "type": "boolean",
            "description": (
                "True if the 'negative' sample is actually a valid/relevant match "
                "for the query (i.e. it was incorrectly labelled as negative). "
                "False if the negative is truly irrelevant."
            ),
        },
        "comment": {
            "type": "string",
            "description": (
                "Brief explanation (1-3 sentences) of the decision, "
                "highlighting semantic overlap or lack thereof."
            ),
        },
    },
    "required": ["is_false_negative", "comment"],
}

# ============================================================
# CLI
# ============================================================
parser = argparse.ArgumentParser(description="Verify False Negatives with LLM")
parser.add_argument("--input",     default=DEFAULT_INPUT_XLSX,
                    help="Pre-computed false-negatives Excel (default: %(default)s)")
parser.add_argument("--dataset",   default=DEFAULT_DATASET_JSON,
                    help="Training dataset JSON (used if --recompute or Excel not found)")
parser.add_argument("--recompute", action="store_true",
                    help="Ignore existing Excel and recompute scores from the JSON dataset")
parser.add_argument("--top",       type=int, default=DEFAULT_TOP,
                    help="Number of worst samples to verify (default: %(default)s)")
parser.add_argument("--model",     default=LLM_MODEL,
                    help="Gemini model name (default: %(default)s)")
parser.add_argument("--device",    default=None,
                    help="PyTorch device for embedding (only used with --recompute)")
args = parser.parse_args()

INPUT_XLSX   = Path(args.input)
DATASET_PATH = Path(args.dataset)
TOP_N        = args.top
LLM_MODEL    = args.model

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("🔬  LLM False-Negative Verification")
print("=" * 70)
print(f"  Input Excel : {INPUT_XLSX}")
print(f"  Top N       : {TOP_N}")
print(f"  LLM Model   : {LLM_MODEL}")
print()

if not GEMINI_API_KEY:
    print("❌  GEMINI_API_KEY not set. Exiting.", file=sys.stderr)
    sys.exit(1)

# ============================================================
# 1. Build DataFrame with margin / pos_score / neg_score
# ============================================================

def load_from_excel(path: Path) -> pd.DataFrame:
    """Load pre-computed false-negative data from existing Excel."""
    print(f"📂  Loading from Excel: {path}")
    df = pd.read_excel(path, sheet_name="False Negatives", engine="openpyxl")
    print(f"✓  Loaded {len(df):,} rows from Excel\n")
    return df


def compute_from_dataset(json_path: Path, device=None) -> pd.DataFrame:
    """Encode the full dataset and return a DataFrame sorted by margin."""
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm

    MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
    BATCH_SIZE = 32

    print(f"📂  Loading dataset: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"✓  Loaded {len(data):,} samples\n")

    print(f"🤖  Loading embedding model: {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME, trust_remote_code=True, device=device)
    actual_device = str(next(model.parameters()).device)
    print(f"✓  Model on {actual_device}\n")

    def encode(texts, label):
        emb = model.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True,
            batch_size=BATCH_SIZE, show_progress_bar=True,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return emb.astype(np.float32)

    queries   = [d["query"]    for d in data]
    positives = [d["positive"] for d in data]
    negatives = [d["negative"] for d in data]

    q_emb = encode(queries,   "queries")
    p_emb = encode(positives, "positives")
    n_emb = encode(negatives, "negatives")

    pos_scores = np.einsum("ij,ij->i", q_emb, p_emb).astype(np.float32)
    neg_scores = np.einsum("ij,ij->i", q_emb, n_emb).astype(np.float32)
    margins    = pos_scores - neg_scores

    rows = []
    for idx, item in enumerate(data):
        meta = item.get("metadata", {})
        rows.append({
            "index"      : idx,
            "margin"     : round(float(margins[idx]), 6),
            "pos_score"  : round(float(pos_scores[idx]), 6),
            "neg_score"  : round(float(neg_scores[idx]), 6),
            "query"      : item.get("query", ""),
            "positive"   : item.get("positive", ""),
            "negative"   : item.get("negative", ""),
            "skill_name" : meta.get("skill_name", ""),
            "skill_type" : meta.get("skill_type", ""),
            "skill_uri"  : meta.get("skill_uri", ""),
            "generated_at": meta.get("generated_at", ""),
            "model"      : meta.get("model", ""),
        })

    df = pd.DataFrame(rows)
    # Only keep false negatives (pos < neg)
    df = df[df["margin"] < 0].copy()
    df.sort_values("margin", ascending=True, inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"✓  {len(df):,} false-negative samples identified\n")
    return df


# Decide data source
if not args.recompute and INPUT_XLSX.exists():
    df_all = load_from_excel(INPUT_XLSX)
else:
    if not DATASET_PATH.exists():
        print(f"❌  Dataset not found: {DATASET_PATH}", file=sys.stderr)
        sys.exit(1)
    df_all = compute_from_dataset(DATASET_PATH, device=args.device)

# Sort by margin ascending and pick top N
df_all.sort_values("margin", ascending=True, inplace=True)
df_top = df_all.head(TOP_N).copy().reset_index(drop=True)

print(f"📊  Top {len(df_top)} worst samples selected  "
      f"(margin range: {df_top['margin'].min():.4f} → {df_top['margin'].max():.4f})\n")

# ============================================================
# 2. Async LLM analysis
# ============================================================

client = genai.Client(api_key=GEMINI_API_KEY)


def build_prompt(row: dict) -> str:
    return f"""You are evaluating a triplet from a semantic-similarity training dataset used to train a course-recommendation system that matches job skills to course descriptions.

Each triplet has:
  - **Query**: a skill or competency description that a learner wants to acquire.
  - **Positive**: a course/document description labelled as RELEVANT to the query.
  - **Negative**: a course/document description labelled as IRRELEVANT to the query.

A **false negative** occurs when the "negative" sample is actually semantically relevant to the query — meaning it was incorrectly labelled as irrelevant.

---
**Query:**
{row['query']}

**Positive (labelled RELEVANT, cosine score = {row['pos_score']:.4f}):**
{row['positive']}

**Negative (labelled IRRELEVANT, cosine score = {row['neg_score']:.4f}):**
{row['negative']}
---

Note: the embedding model scored the negative ({row['neg_score']:.4f}) *higher* than the positive ({row['pos_score']:.4f}) for this query, which is suspicious.

Decide:
1. Is the "negative" actually a valid/relevant match for the query? If yes → **is_false_negative = true**.
2. Provide a concise comment (1-3 sentences) explaining your reasoning.
"""


async def analyze_row(row: dict, idx: int, semaphore: asyncio.Semaphore) -> dict:
    async with semaphore:
        prompt = build_prompt(row)
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ANALYSIS_SCHEMA,
        )
        for attempt in range(MAX_RETRIES):
            try:
                response = await asyncio.to_thread(
                    client.models.generate_content,
                    model=LLM_MODEL,
                    contents=prompt,
                    config=config,
                )
                result = json.loads(response.text)
                return {
                    "success": True,
                    "idx": idx,
                    "is_false_negative": result.get("is_false_negative", None),
                    "comment": result.get("comment", ""),
                }
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    print(f"\n  ❌ Failed row {idx}: {e}")
                    return {
                        "success": False,
                        "idx": idx,
                        "is_false_negative": None,
                        "comment": f"[Error: {e}]",
                    }


async def run_analysis(rows: list[dict]) -> list[dict]:
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    tasks = [analyze_row(row, i, semaphore) for i, row in enumerate(rows)]
    results = [None] * len(tasks)

    print(f"🚀  Calling LLM for {len(tasks)} samples  "
          f"(concurrency={MAX_CONCURRENT}, model={LLM_MODEL})\n")

    with __import__("tqdm").tqdm(total=len(tasks), desc="Analyzing", unit="sample") as pbar:
        for coro in asyncio.as_completed(tasks):
            res = await coro
            results[res["idx"]] = res
            pbar.update(1)
            pbar.set_postfix(
                fn=sum(1 for r in results if r and r.get("is_false_negative") is True),
                ok=sum(1 for r in results if r and r.get("success") is True),
            )

    return results


# ============================================================
# 3. Execute
# ============================================================
start = time.time()
records = df_top.to_dict("records")
results = asyncio.run(run_analysis(records))

# Attach results back to DataFrame
df_top["false_negative"] = [
    r["is_false_negative"] if r else None for r in results
]
df_top["comment"] = [
    r["comment"] if r else "" for r in results
]

elapsed = time.time() - start

# Stats
total      = len(df_top)
successful = sum(1 for r in results if r and r.get("success"))
fn_true    = int(df_top["false_negative"].sum())
fn_false   = total - fn_true - int(df_top["false_negative"].isna().sum())

print(f"\n✅  Done in {elapsed:.1f}s")
print(f"   Successful LLM calls : {successful}/{total}")
print(f"   Confirmed FN (true)  : {fn_true}")
print(f"   Confirmed non-FN     : {fn_false}")

# ============================================================
# 4. Save Excel
# ============================================================
output_path = OUTPUT_DIR / f"verified_false_negatives_top{TOP_N}.xlsx"
print(f"\n💾  Saving to {output_path} …")

from openpyxl.styles import Alignment, Font, PatternFill

with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    # ── Sheet 1: verified samples ──
    df_top.to_excel(writer, sheet_name="Verified Samples", index=False)
    ws = writer.sheets["Verified Samples"]

    # Column widths
    col_config = {
        "A": ("index",          8),
        "B": ("margin",        10),
        "C": ("pos_score",     10),
        "D": ("neg_score",     10),
        "E": ("query",         55),
        "F": ("positive",      55),
        "G": ("negative",      55),
        "H": ("skill_name",    30),
        "I": ("skill_type",    20),
        "J": ("skill_uri",     50),
        "K": ("generated_at",  22),
        "L": ("model",         20),
        "M": ("false_negative",16),
        "N": ("comment",       70),
    }
    for col, (_, width) in col_config.items():
        ws.column_dimensions[col].width = width

    wrap_align   = Alignment(wrap_text=True, vertical="top")
    center_align = Alignment(horizontal="center", vertical="top")
    top_align    = Alignment(vertical="top")

    header_fill = PatternFill("solid", fgColor="667EEA")
    header_font = Font(bold=True, color="FFFFFF")
    for cell in ws[1]:
        cell.fill   = header_fill
        cell.font   = header_font
        cell.alignment = center_align

    ws.freeze_panes = "A2"

    # Row fills: red = confirmed FN, green = not FN, grey = error
    fn_fill  = PatternFill("solid", fgColor="FFCDD2")   # red
    ok_fill  = PatternFill("solid", fgColor="C8E6C9")   # green
    err_fill = PatternFill("solid", fgColor="F5F5F5")   # light grey

    text_cols = {4, 5, 6, 13}   # 0-indexed: query, positive, negative, comment

    for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
        fn_val = row[12].value   # column M (0-indexed: 12) = false_negative
        if fn_val is True:
            fill = fn_fill
        elif fn_val is False:
            fill = ok_fill
        else:
            fill = err_fill

        for i, cell in enumerate(row):
            cell.fill      = fill
            cell.alignment = wrap_align if i in text_cols else top_align

    # ── Sheet 2: Summary ──
    summary_rows = {
        "Output file"           : str(output_path),
        "LLM model"             : LLM_MODEL,
        "Samples verified"      : total,
        "Successful LLM calls"  : successful,
        "Confirmed false-negative (true)"  : fn_true,
        "Confirmed non-FN (false)"         : fn_false,
        "Errors / no answer"    : total - successful,
        "FN confirmation rate"  : f"{fn_true / max(successful, 1) * 100:.1f}%",
        "Elapsed (s)"           : round(elapsed, 1),
    }
    df_summary = pd.DataFrame(
        list(summary_rows.items()), columns=["Metric", "Value"]
    )
    df_summary.to_excel(writer, sheet_name="Summary", index=False)
    ws2 = writer.sheets["Summary"]
    ws2.column_dimensions["A"].width = 38
    ws2.column_dimensions["B"].width = 55
    for cell in ws2[1]:
        cell.fill = header_fill
        cell.font = header_font

print(f"✅  Saved: {output_path}")
