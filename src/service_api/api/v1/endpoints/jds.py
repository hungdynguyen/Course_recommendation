from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/list")
def list_jds():
    """Minimal static JD list for demo UI."""
    return {
        "jds": [
            {
                "jd_id": "jd_backend",
                "title": "Backend Developer",
                "jd_skills": ["python", "sql", "api", "docker", "git"],
            },
            {
                "jd_id": "jd_data",
                "title": "Data Analyst",
                "jd_skills": ["python", "sql", "statistics", "power bi", "excel"],
            },
            {
                "jd_id": "jd_ai",
                "title": "AI Engineer",
                "jd_skills": ["python", "machine learning", "deep learning", "pytorch", "nlp"],
            },
        ]
    }
