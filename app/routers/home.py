from fastapi import APIRouter, Depends, HTTPException, Request

from app.config.templates import templates


router = APIRouter(tags=["home"])


@router.get("/")
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})
