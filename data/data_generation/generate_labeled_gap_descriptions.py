import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

from dotenv import load_dotenv
from google import genai
from google.genai import types
from tqdm import tqdm

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
LABELS_PATH = BASE_DIR / "data" / "processed" / "training_dataset" / "human_labeled_recommendations.json"
SCENARIOS_PATH = BASE_DIR / "data" / "processed" / "course_recommendations" / "course_recommendations.json"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "course_recommendations"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "labeled_skill_gap_descriptions.json"
INDEXED_OUTPUT_FILE = OUTPUT_DIR / "labeled_skill_gap_descriptions_indexed.json"
CHECKPOINT_FILE = OUTPUT_DIR / "labeled_skill_gap_descriptions_checkpoint.json"
STATS_FILE = OUTPUT_DIR / "labeled_skill_gap_descriptions_stats.json"

# Generation config
LLM_MODEL = "gemini-3-flash-preview"
MAX_CONCURRENT_REQUESTS = 10
CHECKPOINT_INTERVAL = 10
MAX_RETRIES = 3
MAX_DESC_WORDS = 20

DESC_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "skill": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["skill", "description"],
            },
        }
    },
    "required": ["items"],
}


class LabeledGapDescriptionGenerator:
    def __init__(self, model_name: str = LLM_MODEL) -> None:
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not found in environment")
        self.model_name = model_name
        self.client = genai.Client(api_key=GEMINI_API_KEY)

    @staticmethod
    def _split_skills(value: object) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            items = value
        else:
            items = str(value).replace("\n", ";").split(";")

        cleaned: List[str] = []
        seen: Set[str] = set()
        for item in items:
            text = str(item).strip().strip(" ,\t")
            key = text.lower()
            if text and key not in seen:
                seen.add(key)
                cleaned.append(text)
        return cleaned

    @staticmethod
    def _safe_text(value: object) -> str:
        return str(value or "").strip()

    @staticmethod
    def _extract_labeled_pair_ids(labels: List[dict]) -> Set[int]:
        pair_ids: Set[int] = set()
        for item in labels:
            pair_id = item.get("pair_id")
            if isinstance(pair_id, int):
                pair_ids.add(pair_id)
            elif isinstance(pair_id, str) and pair_id.isdigit():
                pair_ids.add(int(pair_id))
        return pair_ids

    def _build_prompt(self, scenario: dict, gaps: List[str]) -> str:
        jd_title = self._safe_text(scenario.get("jd_title"))
        jd_info = scenario.get("jd_info", {}) if isinstance(scenario.get("jd_info"), dict) else {}
        jd_desc = self._safe_text(jd_info.get("description"))[:1500]
        jd_keywords = self._split_skills(jd_info.get("keywords"))
        jd_tech = self._split_skills(jd_info.get("technical_skills"))

        return (
            "You are writing concise skill-gap descriptions for course recommendation.\n"
            "Use the JD context to disambiguate each skill.\n\n"
            f"JD title: {jd_title}\n"
            f"JD keywords: {'; '.join(jd_keywords[:12])}\n"
            f"JD technical skills: {'; '.join(jd_tech[:20])}\n"
            f"JD description excerpt: {jd_desc}\n\n"
            "Missing technical skills (must keep exact skill names):\n"
            f"{json.dumps(gaps, ensure_ascii=False)}\n\n"
            "Task:\n"
            "- For each skill, write one short description (max 20 words).\n"
            "- Description should explain what competency is missing in this JD context.\n"
            "- Do not invent skills outside the list.\n"
            "- Return JSON matching schema: {\"items\": [{\"skill\": ..., \"description\": ...}]}\n"
        )

    @staticmethod
    def _normalize_response(gaps: List[str], response_obj: dict) -> List[Dict[str, str]]:
        items = response_obj.get("items", []) if isinstance(response_obj, dict) else []

        by_key: Dict[str, str] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            skill = str(item.get("skill") or "").strip()
            desc = str(item.get("description") or "").strip()
            if skill and desc:
                by_key[skill.lower()] = desc

        normalized: List[Dict[str, str]] = []
        for gap in gaps:
            desc = by_key.get(gap.lower(), "")
            if not desc:
                desc = f"Missing practical capability in {gap} for this job context."
            normalized.append({"skill": gap, "description": desc})
        return normalized

    async def _generate_one(self, scenario: dict, semaphore: asyncio.Semaphore) -> dict:
        async with semaphore:
            pair_id = scenario.get("pair_id")
            jd_id = scenario.get("jd_id")
            jd_title = self._safe_text(scenario.get("jd_title"))
            skill_gaps = scenario.get("skill_gaps", {}) if isinstance(scenario.get("skill_gaps"), dict) else {}
            gaps = self._split_skills(skill_gaps.get("missing_technical_skills"))

            if not gaps:
                return {
                    "success": True,
                    "pair_id": pair_id,
                    "jd_id": jd_id,
                    "jd_title": jd_title,
                    "gap_descriptions": [],
                    "generated_at": datetime.now().isoformat(),
                    "model": self.model_name,
                }

            prompt = self._build_prompt(scenario, gaps)

            for attempt in range(MAX_RETRIES):
                try:
                    config = types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=DESC_SCHEMA,
                    )
                    resp = await asyncio.to_thread(
                        self.client.models.generate_content,
                        model=self.model_name,
                        contents=prompt,
                        config=config,
                    )
                    obj = json.loads(resp.text)
                    descriptions = self._normalize_response(gaps, obj)
                    return {
                        "success": True,
                        "pair_id": pair_id,
                        "jd_id": jd_id,
                        "jd_title": jd_title,
                        "gap_descriptions": descriptions,
                        "generated_at": datetime.now().isoformat(),
                        "model": self.model_name,
                    }
                except Exception as exc:
                    if attempt < MAX_RETRIES - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    return {
                        "success": False,
                        "pair_id": pair_id,
                        "jd_id": jd_id,
                        "jd_title": jd_title,
                        "error": str(exc),
                        "generated_at": datetime.now().isoformat(),
                        "model": self.model_name,
                    }

    async def run(self) -> None:
        labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
        scenarios = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))

        labeled_pair_ids = self._extract_labeled_pair_ids(labels)
        labeled_scenarios = [
            s for s in scenarios
            if s.get("pair_id") is not None and int(s.get("pair_id")) in labeled_pair_ids
        ]

        print("=" * 60)
        print("Generate short descriptions for labeled missing technical skills")
        print("=" * 60)
        print(f"Labeled pairs: {len(labeled_pair_ids)}")
        print(f"Scenarios to process: {len(labeled_scenarios)}")

        results: List[dict] = []
        processed_pair_ids: Set[int] = set()

        if OUTPUT_FILE.exists():
            try:
                existing = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
                if isinstance(existing, list):
                    results = existing
                    processed_pair_ids = {
                        int(item["pair_id"]) for item in existing
                        if isinstance(item, dict) and str(item.get("pair_id", "")).isdigit()
                    }
                    print(f"Loaded existing output: {len(processed_pair_ids)} processed pairs")
            except Exception as exc:
                print(f"Could not read existing output: {exc}")

        if not processed_pair_ids and CHECKPOINT_FILE.exists():
            try:
                checkpoint_data = json.loads(CHECKPOINT_FILE.read_text(encoding="utf-8"))
                if isinstance(checkpoint_data, list):
                    results = checkpoint_data
                    processed_pair_ids = {
                        int(item["pair_id"]) for item in checkpoint_data
                        if isinstance(item, dict) and str(item.get("pair_id", "")).isdigit()
                    }
                    print(f"Loaded checkpoint: {len(processed_pair_ids)} processed pairs")
            except Exception as exc:
                print(f"Could not read checkpoint: {exc}")

        to_process = [
            s for s in labeled_scenarios
            if int(s.get("pair_id")) not in processed_pair_ids
        ]
        print(f"Remaining pairs: {len(to_process)}")

        if not to_process:
            print("Nothing to do.")
            return

        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        tasks = [self._generate_one(s, semaphore) for s in to_process]

        checkpoint_counter = 0
        with tqdm(total=len(tasks), desc="Generating gap descriptions", unit="pair") as pbar:
            for coro in asyncio.as_completed(tasks):
                item = await coro
                results.append(item)
                checkpoint_counter += 1
                pbar.update(1)
                pbar.set_postfix({"status": "OK" if item.get("success") else "FAIL"})

                if checkpoint_counter % CHECKPOINT_INTERVAL == 0:
                    CHECKPOINT_FILE.write_text(
                        json.dumps(results, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )

        # Final save
        OUTPUT_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        CHECKPOINT_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

        by_pair_id: Dict[str, Dict[str, str]] = {}
        by_jd_id: Dict[str, Dict[str, str]] = {}
        for item in results:
            if not isinstance(item, dict) or not item.get("success"):
                continue
            desc_items = item.get("gap_descriptions", [])
            if not isinstance(desc_items, list):
                continue

            skill_map: Dict[str, str] = {}
            for rec in desc_items:
                if not isinstance(rec, dict):
                    continue
                key = str(rec.get("skill") or "").strip().lower()
                val = str(rec.get("description") or "").strip()
                if key and val:
                    skill_map[key] = val

            if not skill_map:
                continue

            pair_id = item.get("pair_id")
            jd_id = item.get("jd_id")
            if pair_id is not None:
                by_pair_id[str(pair_id)] = skill_map
            if jd_id is not None and str(jd_id).strip():
                by_jd_id[str(jd_id)] = skill_map

        indexed = {
            "created_at": datetime.now().isoformat(),
            "model": self.model_name,
            "by_pair_id": by_pair_id,
            "by_jd_id": by_jd_id,
        }
        INDEXED_OUTPUT_FILE.write_text(json.dumps(indexed, ensure_ascii=False, indent=2), encoding="utf-8")

        success_count = sum(1 for r in results if r.get("success"))
        fail_count = len(results) - success_count
        empty_gap_count = sum(
            1
            for r in results
            if r.get("success") and isinstance(r.get("gap_descriptions"), list) and not r.get("gap_descriptions")
        )

        stats = {
            "created_at": datetime.now().isoformat(),
            "model": self.model_name,
            "total_records": len(results),
            "success_records": success_count,
            "failed_records": fail_count,
            "empty_gap_records": empty_gap_count,
            "labels_path": str(LABELS_PATH),
            "scenarios_path": str(SCENARIOS_PATH),
            "output_file": str(OUTPUT_FILE),
            "indexed_output_file": str(INDEXED_OUTPUT_FILE),
            "checkpoint_file": str(CHECKPOINT_FILE),
        }
        STATS_FILE.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

        print("\nDone")
        print(f"Output: {OUTPUT_FILE}")
        print(f"Indexed output: {INDEXED_OUTPUT_FILE}")
        print(f"Stats: {STATS_FILE}")
        print(f"Success: {success_count}, Fail: {fail_count}")


def main() -> None:
    if not GEMINI_API_KEY:
        print("GEMINI_API_KEY is not set")
        return

    generator = LabeledGapDescriptionGenerator()
    try:
        asyncio.run(generator.run())
    except Exception as exc:
        print(f"Error: {exc}")
        raise


if __name__ == "__main__":
    main()
