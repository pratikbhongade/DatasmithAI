from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, WebSocket, WebSocketDisconnect, BackgroundTasks
from typing import Optional
from app.services.extractor import ExtractorRouter
from app.agents.orchestrator import AgentOrchestrator
from app.models.schemas import AgentResponse
from app.api.websocket import manager
from app.core.logger import get_logger
from app.core.config import settings

from app.core.memory import memory_manager

logger = get_logger(__name__)
router = APIRouter()

# lazy init - otherwise uvicorn reloader locks the qdrant db file
_extractor = None
_orchestrator = None

def get_extractor() -> ExtractorRouter:
    global _extractor
    if _extractor is None:
        _extractor = ExtractorRouter()
    return _extractor

def get_orchestrator() -> Optional[AgentOrchestrator]:
    global _orchestrator
    if _orchestrator is None:
        try:
            if settings.GROQ_API_KEY and settings.GROQ_API_KEY != "your_api_key_here":
                _orchestrator = AgentOrchestrator()
            else:
                logger.warning("GROQ_API_KEY is not set. Orchestrator will fail if called.")
        except Exception as e:
            logger.error(f"Failed to initialize Orchestrator: {e}")
    return _orchestrator

@router.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await manager.connect(websocket, session_id)
    try:
        while True:
            # We don't expect messages from client after connection, but we keep it open
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(session_id)

@router.post("/chat")
async def chat_endpoint(
    bg_tasks: BackgroundTasks,
    session_id: str = Form(...),
    text_prompt: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    try:
        if not text_prompt and not file:
            raise HTTPException(status_code=400, detail="Must provide either text or a file.")

        file_bytes = None
        file_type = None
        file_name = None

        # print("in chat endpoint")
        if file:
            file_bytes = await file.read()
            file_type = file.content_type
            file_name = file.filename

        logger.info(f"Extracting content. File: {file_name}, Type: {file_type}")
        extractor = get_extractor()
        ext_res = extractor.extract(
            file_bytes=file_bytes, 
            file_type=file_type, 
            file_name=file_name,
            text_prompt=text_prompt
        )

        if ext_res.source_type != "text" and len(ext_res.content) > 0:
            # got real content from file or YouTube - save it and don't inject old context
            memory_manager.save_context(session_id, ext_res.content, ext_res.source_type)
        else:
            mem = memory_manager.get_context(session_id)
            if mem and text_prompt:
                ext_res.content = f"[User Prompt: {text_prompt}]\n\n--- Previous Document Context ---\n{mem['content']}"
                ext_res.source_type = mem['type']

        orchestrator = get_orchestrator()
        if not orchestrator:
            raise HTTPException(status_code=500, detail="Agent orchestrator is not initialized. Please check API keys.")

        # kick it off in the background so we can stream results over ws
        bg_tasks.add_task(
            orchestrator.process_request,
            ext_data=ext_res,
            session_id=session_id,
            raw_prompt=text_prompt if text_prompt else ""
        )

        return {"status": "processing"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))
