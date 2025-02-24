from fastapi import APIRouter, Depends, HTTPException


router = APIRouter(tags=["home"])


@router.get("/")
async def home():
    return {}
