import logging

from fastapi import APIRouter

from ..composer import compose_full
from ..conversation import Conversation, conversations
from ..models import TickRequest
from ..store import store
from ..timeutil import parse_iso, utcnow

router = APIRouter()
log = logging.getLogger("vera.tick")

MAX_ACTIONS = 20
MAX_UNANSWERED = 3


def _unanswered(merchant_id: str) -> int:
    return sum(1 for c in conversations.open_conversations_for(merchant_id)
               if c.initiated_by_bot and not any(t.role == "inbound" for t in c.turns))


@router.post("/v1/tick")
def tick(body: TickRequest):
    now = parse_iso(body.now) or utcnow()
    candidates = []
    for order, trigger_id in enumerate(dict.fromkeys(body.available_triggers)):
        trg = store.get("trigger", trigger_id)
        if not trg:
            continue
        # The judge's available_triggers list is authoritative for "active now"; expires_at only
        # sizes the suppression window (dataset expiries predate wall-clock `now` in local sims).
        expires = parse_iso(trg.get("expires_at"))
        if expires and expires <= now:
            expires = None
        merchant_id = trg.get("merchant_id") or (trg.get("payload") or {}).get("merchant_id")
        merchant = store.get("merchant", merchant_id)
        if not merchant:
            continue
        category = store.get("category", merchant.get("category_slug"))
        if not category:
            continue
        customer_id = trg.get("customer_id") or (trg.get("payload") or {}).get("customer_id")
        customer = store.get("customer", customer_id) if customer_id else None
        candidates.append((-(trg.get("urgency") or 0), order, trigger_id, trg, merchant_id, merchant,
                           category, customer_id, customer, expires))
    candidates.sort(key=lambda c: (c[0], c[1]))

    actions, used_recipients = [], set()
    for _, _, trigger_id, trg, merchant_id, merchant, category, customer_id, customer, expires in candidates:
        if len(actions) >= MAX_ACTIONS:
            break
        recipient = (merchant_id, customer_id if customer else None)
        if recipient in used_recipients:
            continue
        if conversations.merchant_block_reason(merchant_id, now):
            continue
        if customer is None and _unanswered(merchant_id) >= MAX_UNANSWERED:
            continue
        if customer is not None and (customer.get("preferences") or {}).get("reminder_opt_in") is False:
            continue
        conversation_id = f"conv_{merchant_id}_{trigger_id}"
        if conversations.exists(conversation_id):
            continue
        suppression_key = trg.get("suppression_key")
        if conversations.suppression_holder(suppression_key, now) is not None:
            continue
        try:
            msg = compose_full(category, merchant, trg, customer)
        except Exception:
            log.exception("compose failed for %s", trigger_id)
            continue
        if not msg.body:
            continue
        if not conversations.claim_suppression(msg.suppression_key, conversation_id, merchant_id, expires, now):
            continue
        conv = conversations.open(Conversation(
            conversation_id=conversation_id, merchant_id=merchant_id,
            customer_id=customer_id if customer else None, trigger_id=trigger_id,
            suppression_key=msg.suppression_key, send_as=msg.send_as, topic=msg.topic,
            next_steps=msg.steps, detail=msg.detail, options=msg.options, last_cta=msg.cta, created_at=now))
        conversations.record_outbound(conv, msg.body, now)
        used_recipients.add(recipient)
        actions.append({
            "conversation_id": conversation_id, "merchant_id": merchant_id,
            "customer_id": customer_id if customer else None, "send_as": msg.send_as,
            "trigger_id": trigger_id, "template_name": msg.template_name,
            "template_params": msg.template_params, "body": msg.body, "cta": msg.cta,
            "suppression_key": msg.suppression_key, "rationale": msg.rationale,
        })
    return {"actions": actions}
