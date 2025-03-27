from fastapi import (
    APIRouter,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from app.config.templates import templates
from app.engine.retriever import search
from app.engine.rewriter import rewrite
import json
import logging

logger = logging.getLogger("uvicorn.error")
router = APIRouter(tags=["home"])


@router.get("/")
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


active_connections = []


@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            query = await websocket.receive_text()
            await send_data(query, websocket)
    except WebSocketDisconnect:
        active_connections.remove(websocket)


async def send_data(query: str, websocket: WebSocket):
    logger.info(f"Received query: {query}")
    new_query = rewrite(query)
    logger.info(f"New query: {new_query}")

    articles = search(new_query, top_k=10)
    data = {
        "new_query": new_query,
        "articles": [
            {
                "id": article.id,
                "title": article.title,
                "page_url": article.page_url,
                "origin": article.issue.origin,
                "published": article.published.strftime("%d-%m-%Y"),
            }
            for article in articles
        ],
    }

    await websocket.send_text(json.dumps(data))
