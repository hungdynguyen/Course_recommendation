from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class QueueFile:
    filename: str
    size_bytes: int
    uploaded_at: str
    courses_count: int


class AdminIngestDemoService:
    """Minimal admin ingest flow for demo purposes."""

    def __init__(self) -> None:
        self._repo_root = Path(__file__).resolve().parents[3]
        self._staging_root = self._repo_root / "data" / "staging"
        self._pending_dir = self._staging_root / "pending"
        self._processing_dir = self._staging_root / "processing"
        self._archive_dir = self._staging_root / "archived"
        self._target_catalog_dir = self._repo_root / "data" / "Data_Courses_Filtered" / "demo_uploads"

        self._state_lock = threading.Lock()
        self._active_run_id: Optional[str] = None
        self._run_state: Dict[str, Dict] = {}

        for d in [self._pending_dir, self._processing_dir, self._archive_dir, self._target_catalog_dir]:
            d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Queue operations
    # ------------------------------------------------------------------

    def save_uploaded_json_files(self, files: List[tuple[str, bytes]]) -> Dict:
        saved = []
        for original_name, content in files:
            safe_name = Path(original_name).name
            if not safe_name.lower().endswith(".json"):
                raise ValueError(f"Only .json is supported for demo upload: {safe_name}")

            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
            out_name = f"{timestamp}_{safe_name}"
            out_path = self._pending_dir / out_name
            out_path.write_bytes(content)
            saved.append(out_name)

        return {"saved_files": saved, "saved_count": len(saved)}

    def list_pending_queue(self) -> Dict:
        files: List[QueueFile] = []
        total_courses = 0

        for path in sorted(self._pending_dir.glob("*.json")):
            stat = path.stat()
            courses_count = self._count_courses(path)
            total_courses += courses_count
            files.append(
                QueueFile(
                    filename=path.name,
                    size_bytes=stat.st_size,
                    uploaded_at=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
                    courses_count=courses_count,
                )
            )

        return {
            "pending_files": [asdict(f) for f in files],
            "total_pending_files": len(files),
            "total_pending_courses": total_courses,
        }

    # ------------------------------------------------------------------
    # Pipeline run operations
    # ------------------------------------------------------------------

    def start_pipeline(self, clear_existing: bool = False) -> Dict:
        with self._state_lock:
            if self._active_run_id:
                state = self._run_state[self._active_run_id]
                if state.get("status") == "running":
                    return {
                        "run_id": self._active_run_id,
                        "status": "already_running",
                        "message": "A pipeline run is already in progress",
                    }

            run_id = datetime.now(timezone.utc).strftime("pipeline_%Y%m%d_%H%M%S")
            self._active_run_id = run_id
            self._run_state[run_id] = {
                "run_id": run_id,
                "status": "running",
                "phase": "preparing",
                "progress_percent": 5,
                "started_at": datetime.now(timezone.utc).isoformat(),
                "finished_at": None,
                "message": "Preparing uploaded files",
                "log_tail": "",
                "clear_existing": clear_existing,
            }

        worker = threading.Thread(target=self._run_pipeline_worker, args=(run_id, clear_existing), daemon=True)
        worker.start()

        return {
            "run_id": run_id,
            "status": "started",
            "message": "Pipeline started",
        }

    def get_pipeline_status(self, run_id: Optional[str] = None) -> Dict:
        with self._state_lock:
            target_id = run_id or self._active_run_id
            if not target_id or target_id not in self._run_state:
                return {
                    "run_id": None,
                    "status": "idle",
                    "message": "No pipeline run found",
                }
            return dict(self._run_state[target_id])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_pipeline_worker(self, run_id: str, clear_existing: bool) -> None:
        processing_files: List[Path] = []
        cmd: List[str] = []
        try:
            self._update_state(run_id, phase="moving_files", progress_percent=15, message="Moving files to processing")
            for source in sorted(self._pending_dir.glob("*.json")):
                dest = self._processing_dir / source.name
                source.replace(dest)
                processing_files.append(dest)

            if not processing_files:
                self._update_state(
                    run_id,
                    status="failed",
                    phase="moving_files",
                    progress_percent=100,
                    message="No pending files to ingest",
                    finished=True,
                )
                return

            self._update_state(run_id, phase="copy_to_catalog", progress_percent=30, message="Copying files to catalog")
            for src in processing_files:
                target = self._target_catalog_dir / src.name
                shutil.copy2(src, target)

            self._update_state(run_id, phase="ingest", progress_percent=60, message="Running data_factory pipeline")
            cmd = [
                sys.executable,
                "src/data_factory/scripts/build_graph.py",
                "--config",
                "src/data_factory/config/settings.yaml",
            ]
            if not clear_existing:
                cmd.append("--no-clear")

            completed = subprocess.run(
                cmd,
                cwd=str(self._repo_root),
                capture_output=True,
                text=True,
                check=False,
            )

            combined_log = (completed.stdout or "") + "\n" + (completed.stderr or "")
            log_tail = "\n".join(combined_log.strip().splitlines()[-20:])

            if completed.returncode != 0:
                self._archive_files(processing_files, success=False)
                self._update_state(
                    run_id,
                    status="failed",
                    phase="ingest",
                    progress_percent=100,
                    message=f"Pipeline failed (exit {completed.returncode})",
                    log_tail=log_tail,
                    finished=True,
                )
                return

            self._archive_files(processing_files, success=True)
            self._update_state(
                run_id,
                status="completed",
                phase="done",
                progress_percent=100,
                message="Ingest completed",
                log_tail=log_tail,
                finished=True,
            )

        except Exception as exc:
            if processing_files:
                self._archive_files(processing_files, success=False)
            self._update_state(
                run_id,
                status="failed",
                phase="error",
                progress_percent=100,
                message=f"Error: {exc}",
                log_tail="",
                finished=True,
            )

    def _archive_files(self, files: List[Path], success: bool) -> None:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        bucket = "success" if success else "failed"
        target_dir = self._archive_dir / stamp / bucket
        target_dir.mkdir(parents=True, exist_ok=True)
        for path in files:
            if path.exists():
                path.replace(target_dir / path.name)

    def _count_courses(self, path: Path) -> int:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return 0

        if isinstance(payload, list):
            return len(payload)
        if isinstance(payload, dict):
            if isinstance(payload.get("courses"), list):
                return len(payload["courses"])
            # Common single-course shape
            if payload.get("course_id") or payload.get("title"):
                return 1
        return 0

    def _update_state(
        self,
        run_id: str,
        *,
        status: Optional[str] = None,
        phase: Optional[str] = None,
        progress_percent: Optional[int] = None,
        message: Optional[str] = None,
        log_tail: Optional[str] = None,
        finished: bool = False,
    ) -> None:
        with self._state_lock:
            state = self._run_state.get(run_id)
            if not state:
                return
            if status is not None:
                state["status"] = status
            if phase is not None:
                state["phase"] = phase
            if progress_percent is not None:
                state["progress_percent"] = progress_percent
            if message is not None:
                state["message"] = message
            if log_tail is not None:
                state["log_tail"] = log_tail
            if finished:
                state["finished_at"] = datetime.now(timezone.utc).isoformat()
                self._active_run_id = None
