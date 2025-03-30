from fastapi import FastAPI
from app.routers import chat
from app.routers import home
from dotenv import load_dotenv
from app.core.lifespan import lifespan

load_dotenv()

app = FastAPI(lifespan=lifespan)

app.include_router(home.router)
app.include_router(chat.router)
