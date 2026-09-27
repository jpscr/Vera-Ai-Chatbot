import threading
from dataclasses import dataclass
from typing import Any, Optional

from .models import VALID_SCOPES


@dataclass(frozen=True)
class StoredContext:
    version: int
    payload: dict[str, Any]
    delivered_at: Optional[str]


class ContextStore:
    """In-memory versioned store keyed by (scope, context_id).

    A push is accepted only when its version is strictly greater than the stored one;
    equal or lower versions are rejected as stale (the judge expects 409 on a re-push).
    """

    def __init__(self) -> None:
        self._data: dict[tuple[str, str], StoredContext] = {}
        self._lock = threading.Lock()

    def put(self, scope: str, context_id: str, version: int, payload: dict[str, Any],
            delivered_at: Optional[str]) -> tuple[bool, int]:
        key = (scope, context_id)
        with self._lock:
            current = self._data.get(key)
            if current is not None and current.version >= version:
                return False, current.version
            self._data[key] = StoredContext(version, payload, delivered_at)
            return True, version

    def get(self, scope: str, context_id: Optional[str]) -> Optional[dict[str, Any]]:
        if not context_id:
            return None
        entry = self._data.get((scope, context_id))
        return entry.payload if entry else None

    def counts(self) -> dict[str, int]:
        counts = {scope: 0 for scope in VALID_SCOPES}
        for scope, _ in list(self._data.keys()):
            counts[scope] = counts.get(scope, 0) + 1
        return counts

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


store = ContextStore()
