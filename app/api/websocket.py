from fastapi import WebSocket
from typing import Dict
from app.core.logger import get_logger

logger = get_logger(__name__)

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        self.active_connections[session_id] = websocket
        logger.info(f"WebSocket connected: {session_id}")

    def disconnect(self, session_id: str):
        if session_id in self.active_connections:
            del self.active_connections[session_id]
            logger.info(f"WebSocket disconnected: {session_id}")

    async def send_log(self, session_id: str, step_name: str, details: str):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_json({
                "type": "log",
                "step_name": step_name,
                "details": details
            })

    async def send_token(self, session_id: str, token: str):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_json({
                "type": "token",
                "token": token
            })

    async def send_result(self, session_id: str, cost: float, extracted_text: str = None, needs_clarification: bool = False, clarification_question: str = None):
        if session_id in self.active_connections:
            payload = {
                "type": "result",
                "cost": cost,
                "needs_clarification": needs_clarification
            }
            if extracted_text:
                payload["extracted_text"] = extracted_text
            if clarification_question:
                payload["clarification_question"] = clarification_question
            await self.active_connections[session_id].send_json(payload)

    async def send_error(self, session_id: str, error_msg: str):
        if session_id in self.active_connections:
            await self.active_connections[session_id].send_json({
                "type": "error",
                "details": error_msg
            })

manager = ConnectionManager()
