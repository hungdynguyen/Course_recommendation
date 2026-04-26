from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from service_api.dependencies import get_admin_ingest_demo_service
from service_api.services.admin_ingest_demo import AdminIngestDemoService

router = APIRouter()


@router.post("/courses/upload")
async def upload_courses(
    files: list[UploadFile] = File(...),
    svc: AdminIngestDemoService = Depends(get_admin_ingest_demo_service),
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    payloads: list[tuple[str, bytes]] = []
    for f in files:
        content = await f.read()
        if not content:
            continue
        payloads.append((f.filename or "upload.json", content))

    if not payloads:
        raise HTTPException(status_code=400, detail="All uploaded files are empty")

    try:
        result = svc.save_uploaded_json_files(payloads)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "status": "saved",
        "saved_count": result["saved_count"],
        "saved_files": result["saved_files"],
    }


@router.get("/courses/queue")
def get_pending_queue(
    svc: AdminIngestDemoService = Depends(get_admin_ingest_demo_service),
):
    return svc.list_pending_queue()


@router.post("/pipeline/run")
def run_pipeline(
    clear_existing: bool = False,
    svc: AdminIngestDemoService = Depends(get_admin_ingest_demo_service),
):
    return svc.start_pipeline(clear_existing=clear_existing)


@router.get("/pipeline/status")
def get_pipeline_status(
    run_id: str | None = None,
    svc: AdminIngestDemoService = Depends(get_admin_ingest_demo_service),
):
    return svc.get_pipeline_status(run_id=run_id)
