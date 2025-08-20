import logging, json, time
from typing import Callable
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from .config import settings

# -------- JSON logger ----------
class JsonRequestLogger(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = {
            "level": record.levelname,
            "msg": record.getMessage(),
            "logger": record.name,
        }
        if hasattr(record, "extras"):
            base.update(record.extras)  # type: ignore[attr-defined]
        return json.dumps(base)

handler = logging.StreamHandler()
handler.setFormatter(JsonRequestLogger())
logger = logging.getLogger("app")
logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))
logger.handlers = [handler]
logger.propagate = False

app = FastAPI(title="FaceAuth Backend", version="0.0.1")

# -------- CORS ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- access log middleware ----------
@app.middleware("http")
async def access_logger(request: Request, call_next: Callable):
    t0 = time.perf_counter()
    resp: Response
    try:
        resp = await call_next(request)
        status = resp.status_code
    except Exception:
        status = 500
        raise
    finally:
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        logger.info(
            "req",
            extra={
                "extras": {
                    "route": request.url.path,
                    "method": request.method,
                    "status": status,
                    "latency_ms": latency_ms,
                }
            },
        )
    return resp

@app.get("/healthz")
def healthz():
    return {"ok": True}
