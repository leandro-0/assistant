from fastapi import (
    APIRouter,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from app.database.models import Article
from app.database.connection import Session
from app.config.templates import templates
import json

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
    db = Session()
    articles = db.query(Article).limit(5).all()

    articles_data = [
        {
            "title": article.title,
            "page_url": article.page_url,
            "origin": article.issue.origin,
        }
        for article in articles
    ]

    await websocket.send_text(json.dumps(articles_data))
    db.close()
