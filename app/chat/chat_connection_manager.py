import json
import logging
from typing import Any
from fastapi import WebSocket
from app.chat.answer_engine import answer_question
from app.chat.models.chat import Chat

logger = logging.getLogger("uvicorn.error")


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

        chat = self.active_connections[websocket][article_id]
        answer = answer_question(msg, chat.df_id)
        chat.add_message(answer, "assistant")
        await self.send_message({"message": answer}, websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            del self.active_connections[websocket]
