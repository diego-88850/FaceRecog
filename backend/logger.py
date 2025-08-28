import json
import logging
import time
from typing import Callable
from fastapi import Request

logger = logging.getLogger("faceauth")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_formatter = logging.Formatter("%(message)s")
_handler.setFormatter(_formatter)
logger.addHandler(_handler)

async def log_middleware(request: Request, call_next: Callable):
    t0 = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - t0) * 1000
    record = {
        "route": request.url.path,
        "status": response.status_code,
        "latency_ms": round(latency_ms, 2),
        "method": request.method,
    }
    logger.info(json.dumps(record))
    return response