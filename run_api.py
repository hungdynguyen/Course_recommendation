# Run FastAPI server
# Usage: uvicorn src.service_api.main:app --host 0.0.0.0 --port 8000 --reload

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "src.service_api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
