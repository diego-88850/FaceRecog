from __future__ import annotations
from pydantic import BaseModel

class VerifyResponse(BaseModel):
    decision: str
    who: str | None
    confidence: float | None
    latency_ms: float