import time

from fastapi import APIRouter

from ..conversation import conversations
from ..store import store

router = APIRouter()
START = time.time()

METADATA = {
    "team_name": "Jayant Pant",
    "team_members": ["Jayant Pant"],
    "model": "deterministic-template-composer (no LLM in compose path)",
    "approach": "Trigger-kind dispatch + weighted single-anchor signal selection over merchant state; "
                "all body values read through a provenance-tracking facts layer; rule-based reply intent router",
    "contact_email": "jayantpant_23ee116@dtu.ac.in",
    "version": "0.2.0",
    "submitted_at": "2026-09-27T00:00:00Z",
}


@router.get("/v1/healthz")
def healthz():
    return {"status": "ok", "uptime_seconds": int(time.time() - START), "contexts_loaded": store.counts()}


@router.get("/v1/metadata")
def metadata():
    return METADATA


@router.post("/v1/teardown")
def teardown():
    store.clear()
    conversations.reset()
    return {"status": "ok", "wiped": True}
