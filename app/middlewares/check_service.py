import os
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

logger = logging.getLogger("uvicorn.error")
STATUS_FILE = "active"


def is_service_active():
    """Read the status file and returns True if the service is meant to be active"""
    if not os.path.exists(STATUS_FILE):
        return False

    with open(STATUS_FILE, "r") as f:
        status = f.read().strip().lower()

    return status == "true"


class CheckServiceMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        """Intercepts all requests and blocks if the service is inactive."""
        if not is_service_active():
            logger.warning("Service is inactive, returning 503")
            return JSONResponse(
                status_code=503, content={"detail": "Service temporarily unavailable"}
            )

        return await call_next(request)
