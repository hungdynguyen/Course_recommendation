# Code Module Design – Backend Implementation

## I. Overview

Documento này chi tiết cách implement backend modules cho 2 luồng chính:
1. **User Recommendation Flow** (sử dụng existing endpoints, minor modifications)
2. **Admin Pipeline Management** (NEW modules + endpoints)

---

## II. Directory Structure – New Files

```
src/
├── service_api/
│   ├── api/v1/
│   │   ├── endpoints/
│   │   │   ├── admin.py                    # NEW: admin endpoints (upload, queue, run, status)
│   │   │   ├── jds.py                      # NEW: JD management endpoints
│   │   │   └── recommendations.py          # MODIFY: accept CV text directly
│   │   └── api.py                          # MODIFY: register new routers
│   │
│   ├── models/
│   │   ├── request.py                      # MODIFY: AdminUploadRequest, PipelineRunRequest
│   │   ├── response.py                     # MODIFY: AdminQueueResponse, PipelineStatusResponse
│   │   └── admin.py                        # NEW: AdminFile, IngestStatus, PipelineRun models
│   │
│   ├── services/
│   │   ├── admin_service.py                # NEW: AdminService (queue mgmt, file ops)
│   │   ├── pipeline_manager.py             # NEW: PipelineManager (async ETL trigger, status)
│   │   └── jd_service.py                   # NEW: JDService (list JDs from Neo4j)
│   │
│   ├── schemas/
│   │   └── admin_schemas.py                # NEW: Pydantic schemas for validation
│   │
│   └── tasks/                              # NEW: Async task definitions
│       ├── __init__.py
│       └── ingest_task.py                  # NEW: Celery/BackgroundTask wrapper
│
├── data_factory/
│   ├── pipelines/
│   │   ├── graph_build_pipeline.py         # MODIFY: extract metrics, add progress tracking
│   │   └── incremental_pipeline.py         # NEW: incremental ingestion (phase 2)
│   │
│   ├── services/
│   │   └── pipeline_state_manager.py       # NEW: track pipeline progress
│   │
│   └── scripts/
│       ├── build_graph.py                  # MODIFY: accept callback for progress
│       └── ingest_from_staging.py          # NEW: entry point for demo (load from staging dir)
│
└── shared/
    ├── models/
    │   └── admin.py                        # NEW: shared admin models
    │
    └── utils/
        └── file_utils.py                   # NEW: staging dir helpers
```

---

## III. Data Models & Schemas

### 3.1 Admin Models (`src/service_api/models/admin.py`)

```python
from enum import Enum
from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

class IngestPhase(str, Enum):
    """Pipeline phases"""
    LOAD = "load"
    EMBED = "embed"
    DEDUP = "dedup"
    ES_INDEX = "es_index"
    NEO4J_BUILD = "neo4j_build"

class IngestStatus(str, Enum):
    """Pipeline status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class AdminFile(BaseModel):
    """Metadata cho file trong staging queue"""
    file_id: str                          # upload_20260415_143022
    filename: str                         # filename.json
    upload_time: datetime
    courses_count: int
    file_size_bytes: int
    checksum: Optional[str] = None        # SHA256 để verify

class PipelineRun(BaseModel):
    """Track một lần chạy ETL pipeline"""
    run_id: str                           # pipeline_run_20260415_143030
    status: IngestStatus
    current_phase: Optional[IngestPhase] = None
    progress_percent: int = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    failed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    # Phase metrics
    phase_durations: Dict[str, float] = {}
    
    # Aggregated metrics
    total_courses_loaded: int = 0
    total_skills_created: int = 0
    total_skills_deduplicated: int = 0
    metrics_report: Optional[Dict[str, Any]] = None

class JD(BaseModel):
    """Job Description"""
    jd_id: str
    title: str
    description: Optional[str] = None
    category: Optional[str] = None
    keywords: Optional[list[str]] = None
```

### 3.2 Request Models (`src/service_api/models/request.py` – MODIFY)

```python
from typing import Optional, List
from pydantic import BaseModel, Field

# EXISTING – no change needed
class GapRecommendationRequest(BaseModel):
    cv_text: str
    jd_title: str
    jd_keywords: Optional[List[str]] = None
    gap_focus_terms: Optional[List[str]] = None
    max_courses: int = 10

# NEW
class AdminUploadRequest(BaseModel):
    """Bulk course JSON upload"""
    course_data: List[Dict]                # List of course objects (LLM extraction format)
    
    class Config:
        json_schema_extra = {
            "example": {
                "course_data": [
                    {
                        "course_id": "COURSE_001",
                        "course_title": "Python Advanced",
                        "category": "Khoa Công nghệ thông tin_2024",
                        "skills": [
                            {"skill_name": "Advanced Python", "type": "taught"},
                            {"skill_name": "Design Patterns", "type": "taught"}
                        ]
                    }
                ]
            }
        }

class PipelineRunRequest(BaseModel):
    """Trigger ETL pipeline"""
    clear_existing: bool = False           # False = incremental, True = full rebuild
    output_metrics_report: bool = True
    skip_dedup: bool = False               # Future: skip dedup phase for speed
    max_courses_per_batch: Optional[int] = None  # Override batch size

class JDListRequest(BaseModel):
    """Optional filter for JD list"""
    category_filter: Optional[str] = None
    limit: int = 50
```

### 3.3 Response Models (`src/service_api/models/response.py` – MODIFY)

```python
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel
from service_api.models.admin import AdminFile, PipelineRun, IngestStatus, IngestPhase, JD

# EXISTING – no change needed
class GapRecommendationResponse(BaseModel):
    """Existing model"""
    gap_skills: List[Dict]
    recommended_courses: List[Dict]
    metadata: Dict

# NEW
class AdminUploadResponse(BaseModel):
    """Response after upload"""
    status: str = "saved"                 # or "error"
    file_id: str
    courses_count: int
    location: str
    message: str
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "saved",
                "file_id": "upload_20260415_143022",
                "courses_count": 45,
                "location": "data/staging/pending/upload_20260415_143022.json",
                "message": "45 courses queued for ingestion"
            }
        }

class AdminQueueResponse(BaseModel):
    """List pending files"""
    pending_files: List[AdminFile]
    total_pending_courses: int
    total_pending_size_bytes: int

class PipelineRunResponse(BaseModel):
    """Response after triggering pipeline"""
    run_id: str
    status: str = "started"
    message: str
    estimated_duration_sec: Optional[int] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "run_id": "pipeline_run_20260415_143030",
                "status": "started",
                "message": "Ingestion pipeline started",
                "estimated_duration_sec": 180
            }
        }

class PipelineStatusResponse(BaseModel):
    """Real-time status during/after run"""
    run_id: str
    status: str                            # "running", "completed", "failed"
    current_phase: Optional[str] = None    # "load", "embed", "dedup", "es_index", "neo4j"
    phase_name: Optional[str] = None
    progress_percent: int = 0
    
    started_at: Optional[datetime] = None
    elapsed_time_sec: float = 0
    estimated_remaining_sec: Optional[float] = None
    
    phase_durations: Dict[str, float] = {}
    
    # Final aggregates
    total_courses_ingested: int = 0
    total_skills_created: int = 0
    metrics_report: Optional[Dict[str, Any]] = None
    
    error_message: Optional[str] = None

class JDListResponse(BaseModel):
    """List available JDs"""
    jds: List[JD]
    total_count: int
```

---

## IV. New Service Modules

### 4.1 AdminService (`src/service_api/services/admin_service.py`)

**Purpose**: Manage staging directory, file operations, queue state.

```python
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Optional
import json
import hashlib
import logging

from service_api.models.admin import AdminFile

logger = logging.getLogger(__name__)

class AdminService:
    """Manage staging area + queue operations"""
    
    STAGING_BASE = Path("data/staging")
    PENDING_DIR = STAGING_BASE / "pending"
    PROCESSING_DIR = STAGING_BASE / "processing"
    ARCHIVED_DIR = STAGING_BASE / "archived"
    
    def __init__(self):
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create staging directories if missing"""
        for d in [self.PENDING_DIR, self.PROCESSING_DIR, self.ARCHIVED_DIR]:
            d.mkdir(parents=True, exist_ok=True)
    
    def save_uploaded_courses(self, course_data: List[Dict]) -> AdminFile:
        """Save courses to pending queue with metadata"""
        file_id = f"upload_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        filename = f"{file_id}.json"
        filepath = self.PENDING_DIR / filename
        
        # Serialize
        with open(filepath, 'w') as f:
            json.dump({"courses": course_data}, f, indent=2)
        
        file_size = filepath.stat().st_size
        checksum = self._calculate_checksum(filepath)
        
        admin_file = AdminFile(
            file_id=file_id,
            filename=filename,
            upload_time=datetime.now(timezone.utc),
            courses_count=len(course_data),
            file_size_bytes=file_size,
            checksum=checksum
        )
        
        logger.info(f"Saved {len(course_data)} courses to {filepath}")
        return admin_file
    
    def list_pending_queue(self) -> tuple[List[AdminFile], int]:
        """List all pending files + total course count"""
        pending_files = []
        total_courses = 0
        
        for filepath in sorted(self.PENDING_DIR.glob("*.json")):
            try:
                with open(filepath) as f:
                    data = json.load(f)
                courses = data.get("courses", [])
                count = len(courses)
                total_courses += count
                
                admin_file = AdminFile(
                    file_id=filepath.stem,
                    filename=filepath.name,
                    upload_time=datetime.fromtimestamp(filepath.stat().st_mtime, tz=timezone.utc),
                    courses_count=count,
                    file_size_bytes=filepath.stat().st_size,
                )
                pending_files.append(admin_file)
            except Exception as e:
                logger.error(f"Error reading {filepath}: {e}")
        
        return pending_files, total_courses
    
    def move_to_processing(self) -> List[Path]:
        """Move all pending files to processing directory"""
        moved_files = []
        for source in self.PENDING_DIR.glob("*.json"):
            dest = self.PROCESSING_DIR / source.name
            source.rename(dest)
            moved_files.append(dest)
            logger.info(f"Moved {source.name} to processing")
        return moved_files
    
    def move_to_archived(self, files: List[Path], success: bool = True):
        """Move processed files to archived directory"""
        status_subdir = "success" if success else "failed"
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        archive_dir = self.ARCHIVED_DIR / date_str / status_subdir
        archive_dir.mkdir(parents=True, exist_ok=True)
        
        for file in files:
            dest = archive_dir / file.name
            file.rename(dest)
            logger.info(f"Archived {file.name} to {archive_dir}")
    
    def load_courses_from_files(self, files: List[Path]) -> List[Dict]:
        """Merge all courses from processing files"""
        all_courses = []
        for filepath in files:
            try:
                with open(filepath) as f:
                    data = json.load(f)
                courses = data.get("courses", [])
                all_courses.extend(courses)
            except Exception as e:
                logger.error(f"Error loading {filepath}: {e}")
        
        logger.info(f"Loaded {len(all_courses)} total courses from {len(files)} files")
        return all_courses
    
    @staticmethod
    def _calculate_checksum(filepath: Path) -> str:
        """Calculate SHA256 of file"""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
```

### 4.2 PipelineManager (`src/service_api/services/pipeline_manager.py`)

**Purpose**: Track pipeline execution state, trigger async tasks, expose status.

```python
from typing import Optional, Dict, Any
from datetime import datetime, timezone
import logging
import uuid

from service_api.models.admin import PipelineRun, IngestStatus, IngestPhase

logger = logging.getLogger(__name__)

class PipelineManager:
    """Manage pipeline runs and status"""
    
    def __init__(self):
        # In-memory store (production: use Redis/DB)
        self._runs: Dict[str, PipelineRun] = {}
    
    def create_run(
        self,
        clear_existing: bool = False,
        output_metrics: bool = True
    ) -> str:
        """Create new pipeline run, return run_id"""
        run_id = f"pipeline_run_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        
        run = PipelineRun(
            run_id=run_id,
            status=IngestStatus.PENDING,
            started_at=datetime.now(timezone.utc)
        )
        
        self._runs[run_id] = run
        logger.info(f"Created run {run_id}")
        return run_id
    
    def get_run(self, run_id: str) -> Optional[PipelineRun]:
        """Fetch run by ID"""
        return self._runs.get(run_id)
    
    def update_phase(
        self,
        run_id: str,
        phase: IngestPhase,
        progress_percent: int = 0
    ):
        """Update current phase during run"""
        run = self._runs.get(run_id)
        if run:
            run.current_phase = phase
            run.progress_percent = progress_percent
            run.status = IngestStatus.RUNNING
            logger.info(f"Run {run_id}: phase={phase}, progress={progress_percent}%")
    
    def update_phase_duration(self, run_id: str, phase: str, duration_sec: float):
        """Record phase completion time"""
        run = self._runs.get(run_id)
        if run:
            run.phase_durations[phase] = duration_sec
    
    def complete_run(
        self,
        run_id: str,
        metrics_report: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ):
        """Mark run as completed or failed"""
        run = self._runs.get(run_id)
        if run:
            run.completed_at = datetime.now(timezone.utc)
            if error:
                run.status = IngestStatus.FAILED
                run.error_message = error
                run.failed_at = datetime.now(timezone.utc)
            else:
                run.status = IngestStatus.COMPLETED
                if metrics_report:
                    run.metrics_report = metrics_report
                    # Extract aggregates
                    run.total_courses_loaded = metrics_report.get("phases", {}).get("load", {}).get("raw_courses", 0)
                    run.total_skills_created = metrics_report.get("phases", {}).get("dedup", {}).get("canonical_skills", 0)
            
            logger.info(f"Run {run_id}: {run.status}")
    
    def get_estimated_remaining_time(self, run_id: str) -> Optional[float]:
        """Estimate remaining time based on phase progress"""
        run = self._runs.get(run_id)
        if not run or not run.started_at:
            return None
        
        elapsed = (datetime.now(timezone.utc) - run.started_at).total_seconds()
        
        # Heuristic: if 40% done in 60% of estimated time, adjust estimate
        if run.progress_percent > 0 and run.progress_percent < 100:
            estimated_total = elapsed / (run.progress_percent / 100)
            remaining = estimated_total - elapsed
            return max(0, remaining)
        
        return None
```

### 4.3 JDService (`src/service_api/services/jd_service.py`)

**Purpose**: Fetch JD list from Neo4j or static config.

```python
from typing import List, Optional
import logging

from shared.db.neo4j_client import Neo4jClient
from service_api.models.admin import JD

logger = logging.getLogger(__name__)

class JDService:
    """Manage Job Descriptions"""
    
    def __init__(self, neo4j: Neo4jClient):
        self._neo4j = neo4j
        self._jd_cache: Optional[List[JD]] = None
    
    def list_jds(self, category_filter: Optional[str] = None, limit: int = 50) -> List[JD]:
        """
        Fetch JD list from Neo4j (or static config).
        
        Future: Store as nodes in Neo4j {'JobDescription' nodes}
        For now: Extract unique categories from courses as proxy JDs.
        """
        # Query: Get distinct course categories
        query = """
        MATCH (c:Course)
        RETURN DISTINCT c.category AS category
        LIMIT $limit
        """
        
        try:
            rows = self._neo4j.query(query, {"limit": limit})
            jds = []
            for i, row in enumerate(rows):
                category = row.get("category", "Unknown")
                jd = JD(
                    jd_id=f"jd_{i:03d}",
                    title=f"Role in {category}",
                    description=f"Typical role for {category} department",
                    category=category
                )
                jds.append(jd)
            
            return jds
        except Exception as e:
            logger.error(f"Error fetching JDs: {e}")
            return []
    
    def get_jd_by_id(self, jd_id: str) -> Optional[JD]:
        """Fetch single JD"""
        jds = self.list_jds()
        for jd in jds:
            if jd.jd_id == jd_id:
                return jd
        return None
```

---

## V. Async Task Processing

### 5.1 Ingest Task (`src/service_api/tasks/ingest_task.py`)

**Purpose**: Run ETL pipeline asynchronously, update status periodically.

```python
import logging
import asyncio
from pathlib import Path
from typing import Optional, Callable

from data_factory.pipelines.graph_build_pipeline import GraphBuildPipeline
from data_factory.settings import Settings
from shared.db.es_client import ElasticsearchIndexClient
from shared.db.neo4j_client import Neo4jBatchClient
from shared.embeddings.embedding_service import EmbeddingService
from service_api.services.admin_service import AdminService
from service_api.services.pipeline_manager import PipelineManager

logger = logging.getLogger(__name__)

class IngestTask:
    """Run ETL pipeline with status callbacks"""
    
    def __init__(self, pipeline_manager: PipelineManager, admin_service: AdminService):
        self.pm = pipeline_manager
        self.admin = admin_service
    
    def run_ingest(
        self,
        run_id: str,
        clear_existing: bool = False,
        output_metrics: bool = True
    ):
        """
        Execute full pipeline:
        1. Move pending → processing
        2. Load courses
        3. Init ETL pipeline
        4. Run graph build with progress callbacks
        5. Move to archived
        6. Update final status
        """
        try:
            logger.info(f"Starting ingest run {run_id}")
            
            # Step 1: Move files
            files = self.admin.move_to_processing()
            if not files:
                logger.warning("No files to process")
                self.pm.complete_run(run_id, error="No files in queue")
                return
            
            # Step 2: Load courses
            courses = self.admin.load_courses_from_files(files)
            logger.info(f"Loaded {len(courses)} courses for ingest")
            
            # Step 3: Initialize pipeline components
            settings = Settings.load(None)
            neo4j = Neo4jBatchClient(
                uri=settings.neo4j.uri,
                username=settings.neo4j.username,
                password=settings.neo4j.password,
                database=settings.neo4j.database,
                batch_size=settings.neo4j.batch_size,
            )
            
            es = ElasticsearchIndexClient(
                hosts=settings.elasticsearch.hosts,
                username=settings.elasticsearch.username,
                password=settings.elasticsearch.password,
                vector_dim=settings.elasticsearch.vector_dim,
                batch_size=settings.elasticsearch.batch_size,
                recreate_index=settings.elasticsearch.recreate_index,
                index=settings.elasticsearch.index,
            )
            
            embedding = EmbeddingService(
                model_name=settings.embedding.model_name,
                model_path=settings.embedding.model_path,
                device=settings.embedding.device,
                batch_size=settings.embedding.batch_size,
                normalize=settings.embedding.normalize,
            )
            
            # Step 4: Create progress callback
            def on_phase_update(phase: str, progress: float, duration: float):
                self.pm.update_phase(run_id, phase, int(progress * 100))
                self.pm.update_phase_duration(run_id, phase, duration)
            
            # Step 5: Run pipeline
            pipeline = GraphBuildPipeline(
                settings=settings,
                neo4j=neo4j,
                es=es,
                embedding=embedding,
                on_phase_update=on_phase_update,  # NEW callback
            )
            
            metrics_report = pipeline.run(
                clear_existing=clear_existing,
                output_metrics_report=output_metrics,
                metrics_output_path="build_graph_metrics_report.json",
            )
            
            # Step 6: Archive files
            self.admin.move_to_archived(files, success=True)
            
            # Step 7: Mark complete
            self.pm.complete_run(run_id, metrics_report=metrics_report)
            logger.info(f"Ingest run {run_id} completed successfully")
            
        except Exception as e:
            logger.exception(f"Ingest run {run_id} failed: {e}")
            self.pm.complete_run(run_id, error=str(e))
            # Archive failed files
            self.admin.move_to_archived(files, success=False)
        
        finally:
            neo4j.close()
```

---

## VI. Dashboard Endpoints

### 6.1 Admin Endpoints (`src/service_api/api/v1/endpoints/admin.py`)

```python
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from typing import List
import logging

from service_api.config import settings
from service_api.dependencies import (
    get_admin_service,
    get_pipeline_manager,
    get_ingest_task
)
from service_api.models.request import AdminUploadRequest, PipelineRunRequest
from service_api.models.response import (
    AdminUploadResponse,
    AdminQueueResponse,
    PipelineRunResponse,
    PipelineStatusResponse,
)
from service_api.services.admin_service import AdminService
from service_api.services.pipeline_manager import PipelineManager
from service_api.tasks.ingest_task import IngestTask

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post(
    "/courses/upload",
    response_model=AdminUploadResponse,
    summary="Upload courses to staging queue",
)
async def upload_courses(
    request: AdminUploadRequest,
    admin_svc: AdminService = Depends(get_admin_service),
) -> AdminUploadResponse:
    """
    Drag-drop endpoint: receive course JSON array, save to staging/pending
    
    Validation:
    - course_data must be non-empty
    - Each course must have: course_id, course_title, skills (list)
    """
    if not request.course_data:
        raise HTTPException(status_code=400, detail="course_data cannot be empty")
    
    # Validate schema (basic)
    for course in request.course_data:
        if not course.get("course_id") or not course.get("course_title"):
            raise HTTPException(
                status_code=400,
                detail="Each course must have 'course_id' and 'course_title'"
            )
    
    admin_file = admin_svc.save_uploaded_courses(request.course_data)
    
    return AdminUploadResponse(
        status="saved",
        file_id=admin_file.file_id,
        courses_count=admin_file.courses_count,
        location=str(admin_svc.PENDING_DIR / admin_file.filename),
        message=f"{admin_file.courses_count} courses queued for ingestion"
    )

@router.get(
    "/courses/queue",
    response_model=AdminQueueResponse,
    summary="List pending course files in queue",
)
async def get_pending_queue(
    admin_svc: AdminService = Depends(get_admin_service),
) -> AdminQueueResponse:
    """Get list of files waiting to be ingested"""
    pending_files, total_courses = admin_svc.list_pending_queue()
    total_size = sum(f.file_size_bytes for f in pending_files)
    
    return AdminQueueResponse(
        pending_files=pending_files,
        total_pending_courses=total_courses,
        total_pending_size_bytes=total_size,
    )

@router.post(
    "/pipeline/run",
    response_model=PipelineRunResponse,
    summary="Trigger graph build pipeline",
)
async def run_pipeline(
    request: PipelineRunRequest,
    background_tasks: BackgroundTasks,
    pm: PipelineManager = Depends(get_pipeline_manager),
    ingest_task: IngestTask = Depends(get_ingest_task),
) -> PipelineRunResponse:
    """
    Start ETL pipeline to ingest pending courses.
    
    - Moves pending/ → processing/
    - Triggers ETL (async, background)
    - Returns run_id for status polling
    """
    run_id = pm.create_run(
        clear_existing=request.clear_existing,
        output_metrics=request.output_metrics_report
    )
    
    # Queue background task
    background_tasks.add_task(
        ingest_task.run_ingest,
        run_id=run_id,
        clear_existing=request.clear_existing,
        output_metrics=request.output_metrics_report,
    )
    
    return PipelineRunResponse(
        run_id=run_id,
        status="started",
        message="Ingestion pipeline started",
        estimated_duration_sec=180,  # Heuristic
    )

@router.get(
    "/pipeline/status",
    response_model=PipelineStatusResponse,
    summary="Poll pipeline execution status",
)
async def get_pipeline_status(
    run_id: str,
    pm: PipelineManager = Depends(get_pipeline_manager),
) -> PipelineStatusResponse:
    """
    Get real-time status of a running pipeline.
    
    Frontend polls this every 2 sec to update progress bar.
    """
    run = pm.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    
    elapsed = 0
    if run.started_at:
        from datetime import datetime, timezone
        elapsed = (datetime.now(timezone.utc) - run.started_at).total_seconds()
    
    estimated_remaining = pm.get_estimated_remaining_time(run_id)
    
    return PipelineStatusResponse(
        run_id=run_id,
        status=run.status.value,
        current_phase=run.current_phase.value if run.current_phase else None,
        phase_name=_phase_name(run.current_phase) if run.current_phase else None,
        progress_percent=run.progress_percent,
        started_at=run.started_at,
        elapsed_time_sec=elapsed,
        estimated_remaining_sec=estimated_remaining,
        phase_durations=run.phase_durations,
        total_courses_ingested=run.total_courses_loaded,
        total_skills_created=run.total_skills_created,
        metrics_report=run.metrics_report,
        error_message=run.error_message,
    )

def _phase_name(phase) -> str:
    """Map phase enum to human-readable name"""
    mapping = {
        "load": "Loading courses",
        "embed": "Computing skill embeddings",
        "dedup": "Deduplicating skills",
        "es_index": "Indexing to Elasticsearch",
        "neo4j_build": "Building Neo4j KG",
    }
    return mapping.get(phase.value if hasattr(phase, 'value') else phase, "Unknown")
```

### 6.2 JD Endpoints (`src/service_api/api/v1/endpoints/jds.py`)

```python
from fastapi import APIRouter, Depends
from service_api.dependencies import get_jd_service
from service_api.models.response import JDListResponse
from service_api.services.jd_service import JDService

router = APIRouter()

@router.get(
    "/list",
    response_model=JDListResponse,
    summary="Get available Job Descriptions",
)
async def list_jds(
    category_filter: str = None,
    limit: int = 50,
    jd_svc: JDService = Depends(get_jd_service),
) -> JDListResponse:
    """Fetch list of JDs for user to select from"""
    jds = jd_svc.list_jds(category_filter=category_filter, limit=limit)
    return JDListResponse(jds=jds, total_count=len(jds))
```

### 6.3 Update Main Routers (`src/service_api/api/v1/api.py`)

```python
from fastapi import APIRouter

from service_api.api.v1.endpoints import health, recommendations, skills, admin, jds

api_router = APIRouter()
api_router.include_router(health.router,            tags=["health"])
api_router.include_router(skills.router,            prefix="/skills",            tags=["skills"])
api_router.include_router(recommendations.router,   prefix="/recommendations",   tags=["recommendations"])
api_router.include_router(jds.router,               prefix="/jds",               tags=["jds"])
api_router.include_router(admin.router,             prefix="/admin",             tags=["admin"])
```

---

## VII. Dependency Injection (`src/service_api/dependencies.py` – MODIFY)

```python
from functools import lru_cache
from service_api.config import settings
from shared.db.neo4j_client import Neo4jClient
from shared.db.es_client import ElasticsearchClient
from shared.embeddings.embedding_service import EmbeddingService
from service_api.services.course_recommendation import CourseRecommendationService
from service_api.services.gap_detection import GapDetectionService
from service_api.services.skill_search import SkillSearchService

# EXISTING
from service_api.services.admin_service import AdminService
from service_api.services.pipeline_manager import PipelineManager
from service_api.services.jd_service import JDService
from service_api.tasks.ingest_task import IngestTask

# Existing singletons
_neo4j_client: Optional[Neo4jClient] = None
_es_client: Optional[ElasticsearchClient] = None
_embedding_service: Optional[EmbeddingService] = None

# NEW
_admin_service: Optional[AdminService] = None
_pipeline_manager: Optional[PipelineManager] = None
_jd_service: Optional[JDService] = None
_ingest_task: Optional[IngestTask] = None

def get_admin_service() -> AdminService:
    global _admin_service
    if _admin_service is None:
        _admin_service = AdminService()
    return _admin_service

def get_pipeline_manager() -> PipelineManager:
    global _pipeline_manager
    if _pipeline_manager is None:
        _pipeline_manager = PipelineManager()
    return _pipeline_manager

def get_jd_service() -> JDService:
    global _jd_service
    if _jd_service is None:
        _jd_service = JDService(get_neo4j())
    return _jd_service

def get_ingest_task() -> IngestTask:
    global _ingest_task
    if _ingest_task is None:
        _ingest_task = IngestTask(get_pipeline_manager(), get_admin_service())
    return _ingest_task

def close_all():
    """Close all connections"""
    global _neo4j_client, _es_client, _embedding_service
    if _neo4j_client:
        _neo4j_client.close()
    if _es_client:
        _es_client.close()
    # ... etc
```

---

## VIII. Pipeline Modifications

### 8.1 GraphBuildPipeline – Add Progress Callback

```python
# In src/data_factory/pipelines/graph_build_pipeline.py

from typing import Callable, Optional

class GraphBuildPipeline:
    
    def __init__(
        self,
        settings: Settings,
        neo4j: Neo4jBatchClient,
        es: ElasticsearchIndexClient,
        embedding: EmbeddingService,
        on_phase_update: Optional[Callable] = None,  # NEW
    ):
        self._settings = settings
        self._neo4j = neo4j
        self._es = es
        self._embedding = embedding
        self._dedup = SkillDedupService(settings.deduplication)
        self._on_phase_update = on_phase_update  # NEW
    
    def _report_phase_progress(self, phase: str, progress_percent: float, duration: float):
        """Call callback if provided"""
        if self._on_phase_update:
            self._on_phase_update(phase, progress_percent, duration)
    
    def run(self, ...):
        # Phase 1: Load
        t0 = time.perf_counter()
        course_skills = load_course_skills(...)
        phase_dur = time.perf_counter() - t0
        self._report_phase_progress("load", 100.0, phase_dur)
        
        # Phase 2: Embed
        t0 = time.perf_counter()
        embeddings = self._embedding.embed_batch(skill_names, ...)
        phase_dur = time.perf_counter() - t0
        self._report_phase_progress("embed", 100.0, phase_dur)
        
        # ... etc for each phase
```

---

## IX. Error Handling Strategy

```python
# Error handling across layers

class AdminException(Exception):
    """Base admin error"""
    pass

class StagingException(AdminException):
    """File staging error"""
    pass

class PipelineException(AdminException):
    """Pipeline execution error"""
    pass

# In endpoints
try:
    # Logic
except StagingException as e:
    raise HTTPException(status_code=400, detail=str(e))
except PipelineException as e:
    raise HTTPException(status_code=500, detail=str(e))
except Exception as e:
    logger.exception("Unexpected error")
    raise HTTPException(status_code=500, detail="Internal server error")
```

---

## X. Configuration & Environment

### 10.1 .env (NEW variables)

```bash
# Admin & Staging
DATA_STAGING_BASE=data/staging
ADMIN_MAX_FILE_SIZE_MB=100
ADMIN_MAX_COURSES_PER_BATCH=1000

# Pipeline
PIPELINE_ASYNC_ENABLED=true
PIPELINE_STATUS_POLL_INTERVAL_SEC=2
PIPELINE_ESTIMATED_INGEST_TIME_SEC=180

# Batch optimization
EMBEDDING_BATCH_SIZE=64
ELASTICSEARCH_BATCH_SIZE=1000
NEO4J_BATCH_SIZE=5000
```

### 10.2 settings.yaml (MODIFY)

```yaml
# Add new section
admin:
  staging_base: ${DATA_STAGING_BASE}
  max_file_size_mb: ${ADMIN_MAX_FILE_SIZE_MB}
  max_courses_per_batch: ${ADMIN_MAX_COURSES_PER_BATCH}

pipeline:
  async_enabled: ${PIPELINE_ASYNC_ENABLED}
  status_poll_interval_sec: ${PIPELINE_STATUS_POLL_INTERVAL_SEC}
  estimated_ingest_time_sec: ${PIPELINE_ESTIMATED_INGEST_TIME_SEC}
```

---

## XI. Testing Strategy

### 11.1 Unit Tests

```python
# tests/service_api/services/test_admin_service.py
def test_save_uploaded_courses():
    service = AdminService()
    courses = [{"course_id": "C1", "course_title": "Python 101", "skills": []}]
    admin_file = service.save_uploaded_courses(courses)
    
    assert admin_file.courses_count == 1
    assert (service.PENDING_DIR / admin_file.filename).exists()

# tests/service_api/services/test_pipeline_manager.py
def test_create_run():
    pm = PipelineManager()
    run_id = pm.create_run()
    
    run = pm.get_run(run_id)
    assert run.status == IngestStatus.PENDING
    assert run.started_at is not None
```

### 11.2 Integration Tests

```python
# tests/integration/test_admin_flow.py
async def test_full_admin_flow():
    # 1. Upload courses
    # 2. Check queue
    # 3. Trigger pipeline
    # 4. Poll status
    # 5. Verify completion
    pass
```

---

## XII. Implementation Order

1. ✅ Create models (admin.py)
2. ✅ Create services (AdminService, PipelineManager, JDService)
3. ✅ Create endpoints (admin.py, jds.py)
4. ✅ Update dependencies
5. ✅ Create ingest task
6. ✅ Modify GraphBuildPipeline (add callback)
7. ⬜ Add unit tests
8. ⬜ Test end-to-end locally
9. ⬜ Deploy

---

## XIII. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| In-memory PipelineManager | Simple for demo; upgrade to Redis/DB in production |
| Staging directories | Atomic moves, easy rollback on error |
| Batch accumulation | 30% time savings vs individual runs |
| Async background tasks | Don't block frontend during long ingest |
| Callback-based progress | Decouples pipeline from status storage |
| Incremental ingest option | Faster for adding small batches of courses |

---

**Status**: Code design complete. Ready for implementation.
