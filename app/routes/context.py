from fastapi import APIRouter
from fastapi.responses import JSONResponse

from ..models import VALID_SCOPES, ContextPush
from ..store import store
from ..timeutil import iso_z, utcnow

router = APIRouter()


@router.post("/v1/context")
def push_context(body: ContextPush):
    if body.scope not in VALID_SCOPES:
        return JSONResponse(status_code=400, content={
            "accepted": False, "reason": "invalid_scope",
            "details": f"scope must be one of {list(VALID_SCOPES)}, got '{body.scope}'"})
    if not body.context_id:
        return JSONResponse(status_code=400, content={
            "accepted": False, "reason": "invalid_context_id", "details": "context_id is required"})
    accepted, current = store.put(body.scope, body.context_id, body.version, body.payload, body.delivered_at)
    if not accepted:
        return JSONResponse(status_code=409, content={
            "accepted": False, "reason": "stale_version", "current_version": current})
    return {"accepted": True, "ack_id": f"ack_{body.context_id}_v{body.version}", "stored_at": iso_z(utcnow())}
