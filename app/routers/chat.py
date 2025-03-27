from fastapi import (
    APIRouter,
    WebSocket,
    WebSocketDisconnect,
)
from typing import Any
import json
import logging


class Message:
    def __init__(self, text: str, sender: str) -> None:
        self.text = text
        self.sender = sender

    def to_dict(self) -> dict:
        return {"text": self.text, "sender": self.sender}


class Chat:
    def __init__(self, article_id: int) -> None:
        self.article_id = article_id
        self.__messages: list[Message] = []

    def add_message(self, message: str, sender: str) -> None:
        self.__messages.append(Message(message, sender))

    @property
    def messages(self) -> list[Message]:
        return self.__messages


class ChatConnectionManager:
    def __init__(self) -> None:
        self.active_connections: dict[WebSocket, dict[int, Chat]] = {}

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections[websocket] = {}

    async def send_message(self, message: dict[str, Any], websocket: WebSocket):
        await websocket.send_text(json.dumps(message))

    async def process_message(self, message: dict[str, Any], websocket: WebSocket):
        article_id = message["article_id"]
        msg = message["message"]
        logger.info(f"Received message: {article_id} - {msg}")

        if article_id not in self.active_connections[websocket]:
            chat = Chat(article_id)
            chat.add_message(msg, "user")
            self.active_connections[websocket][article_id] = chat
        else:
            self.active_connections[websocket][article_id].add_message(msg, "user")

        await self.send_message({"message": f"({article_id}): {msg}"}, websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            del self.active_connections[websocket]


logger = logging.getLogger("uvicorn.error")
router = APIRouter(tags=["chat"])
manager = ChatConnectionManager()


@router.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            raw_message = await websocket.receive_text()
            message = json.loads(raw_message)
            await manager.process_message(message, websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
