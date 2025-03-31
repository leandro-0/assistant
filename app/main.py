from fastapi import FastAPI
from app.middlewares.check_service import CheckServiceMiddleware
from app.routers import chat
from app.routers import home
from dotenv import load_dotenv
from app.core.lifespan import lifespan

load_dotenv()

app = FastAPI(lifespan=lifespan)

app.add_middleware(CheckServiceMiddleware)

app.include_router(home.router)
app.include_router(chat.router)
