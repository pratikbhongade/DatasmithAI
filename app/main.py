from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router as chat_router
from app.core.config import settings
from app.core.logger import get_logger
import os

logger = get_logger(__name__)

app = FastAPI(title=settings.PROJECT_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router, prefix=settings.API_V1_STR)

ui_path = os.path.join(os.path.dirname(__file__), "ui")
if not os.path.exists(ui_path):
    os.makedirs(ui_path)
    
app.mount("/static", StaticFiles(directory=ui_path), name="static")

@app.get("/")
async def serve_ui():
    index_file = os.path.join(ui_path, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "UI not built yet."}

@app.get("/health")
async def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting DataSmith AI server...")
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
