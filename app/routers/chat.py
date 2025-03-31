import json
import logging
from fastapi import (
    APIRouter,
    WebSocket,
    WebSocketDisconnect,
)
from app.chat.chat_connection_manager import ChatConnectionManager
from app.middlewares.check_service import is_service_active

logger = logging.getLogger("uvicorn.error")
router = APIRouter(tags=["chat"])
manager = ChatConnectionManager()


@router.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    if not is_service_active():
        await websocket.close(code=1001, reason="Service is inactive")
        return

    await manager.connect(websocket)
    try:
        while True:
            raw_message = await websocket.receive_text()
            message = json.loads(raw_message)
            await manager.process_message(message, websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
