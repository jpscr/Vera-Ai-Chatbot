import re
import threading
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

OPT_OUT_DAYS = 30
DEFAULT_SUPPRESSION_DAYS = 7


@dataclass
class Turn:
    role: str
    text: str
    ts: Optional[datetime]
    turn_number: Optional[int] = None
    intent: Optional[str] = None


@dataclass
class Conversation:
    conversation_id: str
    merchant_id: Optional[str]
    customer_id: Optional[str]
    trigger_id: Optional[str]
    suppression_key: Optional[str]
    send_as: str
    topic: str
    # Ordered actions the bot has offered; accepting advances step_index.
    next_steps: list[dict[str, str]] = field(default_factory=list)
    step_index: int = 0
    detail: str = ""
    options: list[str] = field(default_factory=list)
    last_cta: str = ""
    status: str = "open"  # open | waiting | completed | ended
    turns: list[Turn] = field(default_factory=list)
    bot_bodies: list[str] = field(default_factory=list)
    auto_reply_count: int = 0
    hostile_count: int = 0
    unclear_count: int = 0
    wait_until: Optional[datetime] = None
    created_at: Optional[datetime] = None
    initiated_by_bot: bool = True

    @property
    def current_step(self) -> Optional[dict[str, str]]:
        if self.step_index < len(self.next_steps):
            return self.next_steps[self.step_index]
        return None

    def already_sent(self, body: str) -> bool:
        norm = normalize(body)
        return any(normalize(b) == norm for b in self.bot_bodies)


@dataclass
class MerchantState:
    inbound_fingerprints: Counter = field(default_factory=Counter)
    auto_reply_total: int = 0
    opted_out_until: Optional[datetime] = None
    cooldown_until: Optional[datetime] = None


@dataclass
class SuppressionClaim:
    conversation_id: str
    merchant_id: Optional[str]
    expires_at: Optional[datetime]


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9ऀ-ॿ]+", " ", (text or "").lower()).strip()


class ConversationManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._convs: dict[str, Conversation] = {}
        self._merchants: dict[str, MerchantState] = {}
        self._suppression: dict[str, SuppressionClaim] = {}

    # ---- conversations -------------------------------------------------
    def get(self, conversation_id: str) -> Optional[Conversation]:
        return self._convs.get(conversation_id)

    def exists(self, conversation_id: str) -> bool:
        return conversation_id in self._convs

    def open(self, conv: Conversation) -> Conversation:
        with self._lock:
            self._convs[conv.conversation_id] = conv
            return conv

    def get_or_adopt(self, conversation_id: str, merchant_id: Optional[str],
                     customer_id: Optional[str], now: Optional[datetime]) -> Conversation:
        """Replies can arrive for conversations this process never opened (e.g. replay tests)."""
        with self._lock:
            conv = self._convs.get(conversation_id)
            if conv is None:
                conv = Conversation(
                    conversation_id=conversation_id, merchant_id=merchant_id,
                    customer_id=customer_id, trigger_id=None, suppression_key=None,
                    send_as="merchant_on_behalf" if customer_id else "vera",
                    topic="", created_at=now, initiated_by_bot=False)
                self._convs[conversation_id] = conv
            elif merchant_id and not conv.merchant_id:
                conv.merchant_id = merchant_id
            return conv

    def record_inbound(self, conv: Conversation, text: str, ts: Optional[datetime],
                       turn_number: Optional[int], intent: str) -> None:
        with self._lock:
            conv.turns.append(Turn("inbound", text, ts, turn_number, intent))

    def record_outbound(self, conv: Conversation, body: str, ts: Optional[datetime]) -> None:
        with self._lock:
            conv.turns.append(Turn("bot", body, ts))
            conv.bot_bodies.append(body)

    def open_conversations_for(self, merchant_id: str) -> list[Conversation]:
        return [c for c in self._convs.values()
                if c.merchant_id == merchant_id and c.status in ("open", "waiting")]

    # ---- merchant-level state -----------------------------------------
    def merchant(self, merchant_id: Optional[str]) -> MerchantState:
        key = merchant_id or "_unknown"
        with self._lock:
            return self._merchants.setdefault(key, MerchantState())

    def note_inbound_fingerprint(self, merchant_id: Optional[str], text: str) -> int:
        """Counts identical inbound texts per merchant across conversations; returns the new count."""
        state = self.merchant(merchant_id)
        with self._lock:
            fp = normalize(text)
            state.inbound_fingerprints[fp] += 1
            return state.inbound_fingerprints[fp]

    def opt_out(self, merchant_id: Optional[str], now: Optional[datetime]) -> None:
        if not merchant_id or now is None:
            return
        with self._lock:
            self.merchant(merchant_id).opted_out_until = now + timedelta(days=OPT_OUT_DAYS)
            for conv in self.open_conversations_for(merchant_id):
                conv.status = "ended"

    def set_cooldown(self, merchant_id: Optional[str], until: Optional[datetime]) -> None:
        if not merchant_id or until is None:
            return
        with self._lock:
            state = self.merchant(merchant_id)
            if state.cooldown_until is None or until > state.cooldown_until:
                state.cooldown_until = until

    def merchant_block_reason(self, merchant_id: str, now: datetime) -> Optional[str]:
        state = self._merchants.get(merchant_id)
        if state is None:
            return None
        if state.opted_out_until and now < state.opted_out_until:
            return "merchant_opted_out"
        if state.cooldown_until and now < state.cooldown_until:
            return "merchant_cooldown"
        return None

    # ---- suppression ledger -------------------------------------------
    def suppression_holder(self, key: Optional[str], now: Optional[datetime]) -> Optional[SuppressionClaim]:
        if not key:
            return None
        claim = self._suppression.get(key)
        if claim is None:
            return None
        if claim.expires_at is not None and now is not None and now >= claim.expires_at:
            return None
        return claim

    def claim_suppression(self, key: Optional[str], conversation_id: str,
                          merchant_id: Optional[str], expires_at: Optional[datetime],
                          now: Optional[datetime]) -> bool:
        if not key:
            return True
        with self._lock:
            holder = self.suppression_holder(key, now)
            if holder is not None and holder.conversation_id != conversation_id:
                return False
            if expires_at is None and now is not None:
                expires_at = now + timedelta(days=DEFAULT_SUPPRESSION_DAYS)
            self._suppression[key] = SuppressionClaim(conversation_id, merchant_id, expires_at)
            return True

    def reset(self) -> None:
        with self._lock:
            self._convs.clear()
            self._merchants.clear()
            self._suppression.clear()

    def snapshot(self, conversation_id: str) -> Optional[dict[str, Any]]:
        conv = self._convs.get(conversation_id)
        if conv is None:
            return None
        return {"status": conv.status, "step_index": conv.step_index,
                "turns": [(t.role, t.intent, t.text) for t in conv.turns]}


conversations = ConversationManager()
