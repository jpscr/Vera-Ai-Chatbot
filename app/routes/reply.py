"""POST /v1/reply — decide send / wait / end for an inbound merchant or customer message.

Policy (from the testing brief's replay scenarios + api-call-examples):
  auto_reply   1st: one short owner-flag nudge; 2nd: wait 24h; 3rd+: end (counted per merchant,
               across conversations, since the same canned text can arrive on any thread)
  opt_out      end + suppress the merchant for 30 days
  hostile      1st: one-line apology with a STOP path; 2nd: end + suppress
  decline      end, no re-pitch
  defer        wait for the time the merchant asked for
  accept/ack   execute the pending step immediately — never another qualifying question
  answer       (reply to an open-ended ask) treated as accept, echoing their answer
  info         restate the grounded detail + re-offer the same single step
  off_topic    polite decline + redirect to the pending step (brief §2.7); 2nd time: wait
  thanks       end if work was delivered, else wait
  unclear      wait rather than guess
"""
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter

from ..composer import default_steps
from ..conversation import Conversation, conversations
from ..facts import is_hinglish_text
from ..intent import SLOT_PICK, classify, defer_seconds
from ..models import ReplyRequest
from ..store import store
from ..timeutil import parse_iso, utcnow

router = APIRouter()

AUTO_REPLY_WAIT = 86400
DECLINE_COOLDOWN = timedelta(days=3)
AUTO_REPLY_COOLDOWN = timedelta(days=3)


@dataclass
class Decision:
    action: str
    rationale: str
    body: str = ""
    cta: str = "none"
    wait_seconds: int = 0
    opt_out: bool = False
    complete: bool = False
    cooldown: Optional[timedelta] = None


@dataclass
class Ctx:
    conv: Conversation
    message: str
    intent: str
    hinglish: bool
    merchant: Optional[dict]
    now: datetime

    @property
    def name(self) -> str:
        return ((self.merchant or {}).get("identity") or {}).get("name") or "your business"

    @property
    def step(self) -> Optional[dict]:
        return self.conv.current_step

    def pick(self, en: str, hi: str) -> str:
        return hi if self.hinglish else en


def _next_offer(c: Ctx) -> tuple[str, str]:
    nxt = c.conv.current_step
    if not nxt:
        return "", "none"
    return (c.pick(f"\n\nNext: {nxt['label']} — reply YES and I'll take care of it.",
                   f"\n\nAgla step: {nxt['label']} — YES bolein toh woh bhi kar deti hoon."), "binary_yes_no")


# ------------------------------------------------------------------ intent handlers
def on_auto_reply(c: Ctx) -> Decision:
    state = conversations.merchant(c.conv.merchant_id)
    state.auto_reply_total += 1
    c.conv.auto_reply_count += 1
    n = max(state.auto_reply_total, c.conv.auto_reply_count)
    if n == 1:
        label = c.step["label"] if c.step else "the next step"
        body = c.pick(f"Looks like an auto-reply 🙂 When the owner sees this, just reply YES and I'll take it from there ({label}).",
                      f"Lagta hai yeh auto-reply hai 🙂 Owner dekhein toh bas YES reply kar dein — main aage ka kaam kar dungi ({label}).")
        return Decision("send", "Detected WhatsApp Business auto-reply (canned phrasing). One explicit prompt to flag it for the owner; no re-pitch.",
                        body=body, cta="binary_yes_no")
    if n == 2:
        return Decision("wait", "Same auto-reply again — owner isn't at the phone. Backing off 24h instead of burning turns.",
                        wait_seconds=AUTO_REPLY_WAIT)
    return Decision("end", f"Auto-reply {n}x with no human reply; zero engagement signal. Closing and cooling this merchant off for 3 days.",
                    cooldown=AUTO_REPLY_COOLDOWN)


def on_opt_out(c: Ctx) -> Decision:
    return Decision("end", "Merchant explicitly opted out. Closing conversation and suppressing all proactive sends to this merchant for 30 days.",
                    opt_out=True)


def on_hostile(c: Ctx) -> Decision:
    c.conv.hostile_count += 1
    if c.conv.hostile_count >= 2:
        return Decision("end", "Second hostile message; exiting without further engagement and suppressing this merchant for 30 days.",
                        opt_out=True)
    body = c.pick(f"Apologies for the bother. Reply STOP and I won't message again — otherwise I'll only reach out when there's something specific for {c.name}.",
                  f"Maaf kijiye, pareshan karne ka irada nahi tha. STOP reply karein toh main dobara message nahi karungi — warna sirf {c.name} ke kaam ki baat hone par hi likhungi.")
    return Decision("send", "Frustration without an explicit opt-out: one-line apology + clear STOP path, no pitch.", body=body, cta="none")


def on_decline(c: Ctx) -> Decision:
    return Decision("end", "Merchant declined the offer. Ending gracefully with no re-pitch; 3-day cooldown on proactive sends.",
                    cooldown=DECLINE_COOLDOWN)


def on_defer(c: Ctx) -> Decision:
    secs = defer_seconds(c.message)
    return Decision("wait", f"Merchant asked for time; backing off {secs // 60} minutes before any follow-up.", wait_seconds=secs)


def on_accept(c: Ctx, echo: Optional[str] = None) -> Decision:
    conv = c.conv
    if conv.options and conv.customer_id:
        if len(conv.options) == 1:
            return _book(c, conv.options[0])
        body = c.pick(f"Great! Reply 1 for {conv.options[0]} or 2 for {conv.options[1]} and it's booked.",
                      f"Badhiya! {conv.options[0]} ke liye 1, {conv.options[1]} ke liye 2 reply karein — turant book ho jayega.")
        return Decision("send", "Customer said yes but two slots were offered; asking only which slot (booking needs it).",
                        body=body, cta="multi_choice_slot")
    step = c.step
    if step is None:
        return Decision("end", "Merchant acknowledged; every offered step is already delivered. Closing the loop.", complete=True)
    conv.step_index += 1
    lead = step["done"]
    if echo:
        lead = f"Noted: \"{echo}\". " + lead.replace("Got it — ", "")
    tail, cta = _next_offer(c)
    prefix = c.pick("", "Bilkul! ")
    return Decision("send", f"Merchant committed (intent={c.intent}); switched straight to action: '{step['label']}'. "
                            + ("Offering one follow-on step as a single YES." if tail else "No further ask."),
                    body=prefix + lead + tail, cta=cta if tail else "none", complete=not tail)


def _book(c: Ctx, option: str) -> Decision:
    body = c.pick(f"Booked ✅ {option}. We'll send a reminder the day before — see you then!",
                  f"Book ho gaya ✅ {option}. Ek din pehle reminder bhej denge — milte hain!")
    return Decision("send", f"Customer picked a slot; confirmed '{option}' and closed the booking flow.", body=body, complete=True)


def on_slot_pick(c: Ctx) -> Decision:
    idx = int(re.match(SLOT_PICK, c.message.strip().lower()).group(1)) - 1
    return _book(c, c.conv.options[idx])


def on_answer(c: Ctx) -> Decision:
    snippet = c.message.strip().replace("\n", " ")
    snippet = snippet if len(snippet) <= 60 else snippet[:57].rsplit(" ", 1)[0] + "…"
    return on_accept(c, echo=snippet)


def on_info(c: Ctx) -> Decision:
    detail = (c.step or {}).get("detail") or c.conv.detail
    if not detail or not c.step:
        return on_unclear(c)
    detail = detail[0].lower() + detail[1:] if detail[:2] != detail[:2].upper() else detail
    body = c.pick(f"Sure — {detail}\n\nShall I go ahead ({c.step['label']})? Reply YES.",
                  f"Zaroor — {detail}\n\nShuru karun ({c.step['label']})? YES reply karein.")
    return Decision("send", "Merchant asked for details; restated only grounded facts already in context and re-offered the same single step.",
                    body=body, cta="binary_yes_no")


def on_off_topic(c: Ctx) -> Decision:
    c.conv.unclear_count += 1
    if c.conv.unclear_count >= 2:
        return Decision("wait", "Repeated out-of-scope asks; pausing rather than looping the redirect.", wait_seconds=3600)
    text = c.message.lower()
    who = ("your CA" if any(k in text for k in ("gst", "tax", "itr", "ca ", "accountant")) else
           "your bank or advisor" if any(k in text for k in ("loan", "insurance", "stock", "crypto")) else
           "a lawyer" if "legal" in text or "lawyer" in text else "someone better placed")
    redirect = (c.pick(f" Coming back to where we were ({c.step['label']}) — shall I go ahead? Reply YES.",
                       f" Wapas apne kaam par ({c.step['label']}) — shuru karun? YES reply karein.") if c.step else "")
    body = c.pick(f"That one's outside what I can help with — best handled by {who}.",
                  f"Yeh mere scope se bahar hai — iske liye {who} sahi rahenge.") + redirect
    return Decision("send", "Out-of-scope request politely declined; redirected to the original thread without losing it.",
                    body=body, cta="binary_yes_no" if redirect else "none")


def on_thanks(c: Ctx) -> Decision:
    if c.conv.step_index > 0 or c.conv.status == "completed":
        return Decision("end", "Merchant closed the loop with thanks after delivery; ending gracefully.", complete=True)
    return Decision("wait", "Bare thanks with the offer still pending — ambiguous, so waiting rather than re-pitching.", wait_seconds=3600)


def on_ack(c: Ctx) -> Decision:
    return on_accept(c) if c.step or (c.conv.options and c.conv.customer_id) else on_thanks(c)


def on_unclear(c: Ctx) -> Decision:
    c.conv.unclear_count += 1
    secs = 1800 if c.conv.unclear_count == 1 else 14400
    return Decision("wait", "Couldn't map the message to a clear intent; waiting rather than guessing.", wait_seconds=secs)


HANDLERS = {"auto_reply": on_auto_reply, "opt_out": on_opt_out, "hostile": on_hostile, "decline": on_decline,
            "defer": on_defer, "accept": on_accept, "slot_pick": on_slot_pick, "answer": on_answer, "info": on_info,
            "off_topic": on_off_topic, "thanks": on_thanks, "ack": on_ack, "unclear": on_unclear}


# ------------------------------------------------------------------ route
def _finalize(conv: Conversation, d: Decision, now: datetime) -> dict:
    if d.action == "send":
        if not d.body.strip() or conv.already_sent(d.body):
            d = Decision("wait", d.rationale + " Suppressed: would repeat a body already sent in this conversation.", wait_seconds=3600)
        else:
            holder = conversations.suppression_holder(conv.suppression_key, now)
            if holder is not None and holder.conversation_id != conv.conversation_id:
                d = Decision("wait", f"Suppression key '{conv.suppression_key}' is held by {holder.conversation_id}; not sending a colliding message.",
                             wait_seconds=3600)
    if d.action == "send":
        conversations.record_outbound(conv, d.body, now)
        conv.last_cta = d.cta
        conv.status = "completed" if d.complete else "open"
        conv.wait_until = None
        return {"action": "send", "body": d.body, "cta": d.cta, "rationale": d.rationale}
    if d.action == "wait":
        conv.status = "waiting"
        conv.wait_until = now + timedelta(seconds=d.wait_seconds)
        conversations.set_cooldown(conv.merchant_id, conv.wait_until)
        return {"action": "wait", "wait_seconds": d.wait_seconds, "rationale": d.rationale}
    conv.status = "ended"
    if d.opt_out:
        conversations.opt_out(conv.merchant_id, now)
    elif d.cooldown:
        conversations.set_cooldown(conv.merchant_id, now + d.cooldown)
    return {"action": "end", "rationale": d.rationale}


@router.post("/v1/reply")
def reply(req: ReplyRequest):
    now = parse_iso(req.received_at) or utcnow()
    conv = conversations.get_or_adopt(req.conversation_id, req.merchant_id, req.customer_id, now)
    merchant = store.get("merchant", conv.merchant_id)
    category = store.get("category", (merchant or {}).get("category_slug"))
    if not conv.next_steps and not conv.customer_id:
        conv.next_steps = default_steps(category, merchant)

    if conv.status == "ended":
        conversations.record_inbound(conv, req.message, now, req.turn_number, "after_end")
        return {"action": "end", "rationale": "Conversation was already closed; not sending anything further on this thread."}

    repeat = conversations.note_inbound_fingerprint(conv.merchant_id, req.message)
    intent = classify(req.message, repeat_count=repeat, expects_answer=conv.last_cta == "open_ended",
                      options=conv.options if conv.customer_id else None)
    conversations.record_inbound(conv, req.message, now, req.turn_number, intent)
    if intent != "auto_reply":
        conversations.merchant(conv.merchant_id).auto_reply_total = 0

    ctx = Ctx(conv=conv, message=req.message, intent=intent, hinglish=is_hinglish_text(req.message),
              merchant=merchant, now=now)
    return _finalize(conv, HANDLERS[intent](ctx), now)
