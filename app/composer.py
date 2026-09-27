"""Deterministic, template-based composer: compose(category, merchant, trigger, customer?).

Pipeline (pure Python, no LLM, no randomness, no wall-clock reads):
  1. Signal selection — the trigger kind is the dominant "why now" signal. Each kind declares
     weights over merchant anchors (perf gaps, live offers, cohorts, reviews...). The single
     highest weight x magnitude anchor is used as supporting evidence; the rest are dropped and
     named in the rationale.
  2. Grounding — every value in the body is read via `Facts`, which records the source path.
  3. Voice — salutation, vocabulary, code-mix and taboos come from the category JSON.
  4. Assembly — hook (why now) + one support line + one CTA as the last sentence.
"""
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .facts import (CUSTOMER_EMOJI, Anchor, Facts, first_sentence, fmt_date, fmt_int, fmt_pct,
                    humanize, price_in, strip_trailing_punct)
from .timeutil import parse_iso

SINGULAR = {"dentists": "dental clinic", "salons": "salon", "gyms": "gym", "pharmacies": "pharmacy",
            "restaurants": "restaurant"}
ITEM_NOUN = {"dentists": "treatment", "salons": "service", "gyms": "class or program",
             "pharmacies": "product", "restaurants": "dish"}
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


@dataclass
class Draft:
    hook: str
    ask: str
    cta: str = "binary_yes_no"
    support: str = ""
    signal: str = ""
    anchor: Optional[Anchor] = None
    dropped: list[Anchor] = field(default_factory=list)
    judgment: str = ""
    steps: list[dict[str, str]] = field(default_factory=list)
    options: list[str] = field(default_factory=list)
    customer_facing: bool = False
    greeting: str = ""


@dataclass
class ComposedMessage:
    body: str
    cta: str
    send_as: str
    suppression_key: str
    rationale: str
    template_name: str
    template_params: list[str]
    topic: str
    detail: str
    steps: list[dict[str, str]]
    options: list[str]

    def public(self) -> dict[str, Any]:
        return {"body": self.body, "cta": self.cta, "send_as": self.send_as,
                "suppression_key": self.suppression_key, "rationale": self.rationale}


# ================================================================ small text helpers
def sentence(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    return text if text[-1] in ".?!:\"" else text + "."


def iso_dates_to_human(text: str) -> str:
    return re.sub(r"\b(\d{4}-\d{2}-\d{2})\b", lambda m: fmt_date(m.group(1), with_year=True), text or "")


def fmt_time(iso: str) -> str:
    dt = parse_iso(iso)
    if dt is None:
        return ""
    hour = dt.hour % 12 or 12
    suffix = "am" if dt.hour < 12 else "pm"
    return f"{hour}{suffix}" if dt.minute == 0 else f"{hour}:{dt.minute:02d}{suffix}"


def fmt_day_time(iso: str) -> str:
    dt = parse_iso(iso)
    if dt is None:
        return str(iso)
    return f"{DAYS[dt.weekday()]} {dt.day} {dt.strftime('%b')}, {fmt_time(iso)}"


def window_words(window: Optional[str]) -> str:
    m = re.fullmatch(r"(\d+)d", window or "")
    return f"{m.group(1)} days" if m else (humanize(window) if window else "week")


def list_words(items: list[str]) -> str:
    items = [i for i in items if i]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def offer_keywords(title: str) -> list[str]:
    head = re.split(r"[@(]", title)[0]
    return [w for w in re.findall(r"[A-Za-z]{4,}", head)]


def day_restricted_out(title: str, weekday: int) -> bool:
    m = re.search(r"\b(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s*-\s*(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\b", title)
    if not m:
        return False
    a, b = DAYS.index(m.group(1)), DAYS.index(m.group(2))
    return not (a <= weekday <= b)


# ================================================================ anchor -> single action
def anchor_action(f: Facts, a: Anchor) -> tuple[str, str, dict[str, str]]:
    """Maps the chosen supporting fact to one support line, one ask, and the step a YES triggers."""
    t = a.type
    if t == "no_offer":
        cat = f.catalog_offer()
        title = cat["title"] if cat else "a service + price offer"
        return (sentence(a.text),
                f"Want me to put '{title}' live on your listing today?",
                {"label": f"set up '{title}' on your listing",
                 "done": f"Setting up '{title}' on your listing now — I'll send the preview here for your CONFIRM before it goes live.",
                 "detail": sentence(a.text)})
    if t in ("ctr_gap", "calls_gap", "views_down", "calls_down"):
        return (sentence(f"For context, {a.text}"),
                "Want me to rewrite your Google description and draft a fresh post this week?",
                {"label": "refresh your Google description + a new post",
                 "done": "Drafting the new description and a fresh Google post now — sending both here for your CONFIRM before anything goes live.",
                 "detail": sentence(a.text)})
    if t == "review_neg":
        return (sentence(a.text),
                "Want me to draft calm public replies to those reviews?",
                {"label": "draft replies to those reviews",
                 "done": "Drafting the review replies now — polite, no excuses, one line on what's changed. You'll get them here to CONFIRM before posting.",
                 "detail": sentence(a.text)})
    if t == "review_pos":
        return (sentence(f"{a.text} — that's worth putting front and centre"),
                "Want me to turn that into a Google post this week?",
                {"label": "a Google post built on that review theme",
                 "done": "Drafting the Google post around what your reviewers already say — sending it here for your CONFIRM.",
                 "detail": sentence(a.text)})
    if t == "unverified":
        return (sentence(f"Also, {a.text}"),
                "Want me to start the Google verification for you now?",
                {"label": "start Google verification",
                 "done": "Starting the verification request now — Google will reach you by phone call or postcard; I'll tell you exactly what to do when it arrives.",
                 "detail": sentence(a.text)})
    if t == "sub_expired":
        return (sentence(f"Also, {a.text}"),
                "Want me to restart your plan today?",
                {"label": "restart your plan",
                 "done": "Starting the plan renewal now — I'll send the payment confirmation here for your CONFIRM.",
                 "detail": sentence(a.text)})
    if t == "sub_ending":
        return (sentence(f"Also, {a.text}"),
                "Want me to renew it now so there's no gap?",
                {"label": "renew your plan",
                 "done": "Processing the renewal now — I'll send the payment confirmation here for your CONFIRM.",
                 "detail": sentence(a.text)})
    if t == "live_offer":
        return (sentence(a.text),
                "Want me to push it as a Google post today?",
                {"label": "a Google post for your live offer",
                 "done": "Drafting the Google post for your live offer now — sending it here for your CONFIRM.",
                 "detail": sentence(a.text)})
    if t == "cohort":
        return (sentence(f"You have {a.text}"),
                "Want me to draft a WhatsApp you can send them?",
                {"label": f"a WhatsApp for your {f.audience}",
                 "done": f"Drafting the WhatsApp for your {f.audience} now — you'll see it here to CONFIRM before it goes out.",
                 "detail": sentence(f"You have {a.text}")})
    return (sentence(a.text), "Want me to draft a Google post about it?",
            {"label": "a Google post", "done": "Drafting the Google post now — sending it here for your CONFIRM.",
             "detail": sentence(a.text)})


def post_step(topic: str) -> dict[str, str]:
    return {"label": f"a Google post about {topic}",
            "done": f"Drafting the Google post about {topic} — sending it here for your CONFIRM before it goes live."}


# ================================================================ merchant-facing handlers
def h_research_digest(f: Facts) -> Draft:
    item = f.digest_item(f.tp.get("top_item_id")) or f.digest_of_kind(["research", "trend", "tech"])
    if not item:
        return h_default(f)
    f.use("digest.id", item["id"])
    title = iso_dates_to_human(item["title"])
    src = item.get("source", "")
    lead = first_sentence(item.get("summary", ""))
    if item.get("trial_n"):
        lead = strip_trailing_punct(lead) + f" (n={fmt_int(f.use('digest.trial_n', item['trial_n']))})."
    caveat = ""
    parts = re.split(r"(?<=[.!?])\s+", item.get("summary", ""))
    if len(parts) > 1 and len(parts[1]) < 70:
        caveat = parts[1]
    hook = f"{src} just landed — {title}. {sentence(lead)} {caveat}".strip()

    seg = (item.get("patient_segment") or "").rstrip("s")
    anchor, dropped, support = None, [], ""
    cohort_key = next((k for k in f.aggregate if seg and seg.split("_")[0] in k), None)
    if cohort_key:
        n = f.use(f"customer_aggregate.{cohort_key}", f.aggregate[cohort_key])
        support = f"Directly relevant to your {fmt_int(n)} {humanize(seg).replace('high risk', 'high-risk')} {f.audience}."
        anchor = Anchor("cohort", 1.0, support, f"customer_aggregate.{cohort_key}={n}")
    elif item.get("actionable"):
        support = sentence(f"Practical takeaway: {item['actionable']}")
    reader = "patient" if f.slug == "dentists" else "customer"
    ask = f"Want me to pull the key findings and draft a {reader}-friendly WhatsApp you can share?"
    share = f"Quick update from {f.merchant_name}: new research ({src}) — {title}. Reply here if you'd like us to review what it means for you."
    return Draft(
        hook=hook, support=support, ask=ask, anchor=anchor, dropped=dropped,
        signal=f"research_digest '{item['id']}' ({src})",
        judgment="Cohort match between digest segment and merchant aggregate makes it relevant." if cohort_key else "",
        steps=[{"label": f"a {reader}-friendly WhatsApp on this",
                "done": f"Here's the draft — edit freely:\n\n\"{share}\"\n\nReply CONFIRM and I'll queue it for your {f.audience} list.",
                "detail": f"{sentence(lead)} Source: {src}."},
               post_step("this research")])


def h_regulation(f: Facts) -> Draft:
    item = f.digest_item(f.tp.get("top_item_id")) or f.digest_of_kind(["compliance"])
    if not item:
        return h_default(f)
    f.use("digest.id", item["id"])
    deadline = f.tp.get("deadline_iso")
    deadline_txt = fmt_date(f.use("trigger.deadline_iso", deadline), with_year=True) if deadline else ""
    summary = " ".join(re.split(r"(?<=[.!?])\s+", item.get("summary", ""))[:2])
    hook = f"compliance heads-up ({iso_dates_to_human(item.get('source', 'regulator'))}): {iso_dates_to_human(item['title'])}. {summary}"
    support = sentence(f"What to do: {item['actionable']}") if item.get("actionable") else ""
    by = f" before {deadline_txt}" if deadline_txt else ""
    ask = f"Want me to turn this into a 1-page checklist for your team so you're covered{by}?"
    return Draft(hook=hook, support=support, ask=ask,
                 signal=f"regulation_change '{item['id']}' (deadline {deadline_txt or 'n/a'})",
                 judgment="Compliance deadline = loss-aversion lever; no merchant anchor needed.",
                 steps=[{"label": "a compliance checklist",
                         "done": f"Drafting the checklist now: (1) {item.get('actionable', 'review the circular')}; (2) record it in your SOP file; (3) brief your team. Sending it here for your CONFIRM.",
                         "detail": summary},
                        {"label": "a reminder a week before the deadline",
                         "done": f"Done — I'll remind you a week before{by}."}])


def h_cde(f: Facts) -> Draft:
    item = f.digest_item(f.tp.get("digest_item_id")) or f.digest_of_kind(["cde"])
    if not item:
        return h_default(f)
    f.use("digest.id", item["id"])
    when = fmt_day_time(item["date"]) if item.get("date") else ""
    credits = f.tp.get("credits") or item.get("credits")
    bits = [b for b in [when, f"{credits} CDE credits" if credits else ""] if b]
    fee = item.get("actionable") if price_in(item.get("actionable", "")) else humanize(f.tp.get("fee", ""))
    hook = f"CDE slot worth blocking: \"{item['title']}\"" + (f" — {', '.join(bits)}" if bits else "") + "."
    support = " ".join(s for s in [item.get("summary", ""), sentence(fee) if fee else ""] if s)
    return Draft(hook=hook, support=support,
                 ask="Want me to send you the registration details and a calendar reminder?",
                 signal=f"cde_opportunity '{item['id']}'",
                 steps=[{"label": "registration details + a calendar reminder",
                         "done": f"Sending the registration details now and adding a reminder for {when or 'the session'}.",
                         "detail": support}])


def h_competitor(f: Facts) -> Draft:
    tp = f.tp
    kind_word = SINGULAR.get(f.slug, "business")
    if tp.get("competitor_name"):
        name = f.use("trigger.competitor_name", tp["competitor_name"])
        dist = tp.get("distance_km")
        opened = fmt_date(tp["opened_date"]) if tp.get("opened_date") else ""
        hook = f"heads-up — {name} opened {dist} km from you" + (f" on {opened}" if opened else "") + "."
        their = tp.get("their_offer")
        if their:
            hook += f" They're listing '{their}'."
            mine = f.active_offer_matching(offer_keywords(their))
            p_their, p_mine = price_in(their), (price_in(mine) if mine else None)
            if mine and p_their and p_mine:
                diff = int(p_mine[1:].replace(",", "")) - int(p_their[1:].replace(",", ""))
                if diff > 0:
                    hook += f" That's ₹{diff:,} under your '{mine}'."
    else:
        hook = f"heads-up — a new {kind_word} listing near {f.locality or 'you'} was just flagged on Google."
        name = None
    weights = {"review_pos": 1.0, "ctr_lead": 0.8, "live_offer": 0.5, "cohort": 0.4, "no_offer": 0.3}
    anchor, dropped = f.best_anchor(weights)
    judgment = "Don't race to the bottom on price; lead with the merchant's established edge."
    counter = "I wouldn't match on price —" if tp.get("their_offer") else "Best counter isn't a discount —"
    if anchor and anchor.type == "review_pos":
        support = sentence(f"{counter} {anchor.text}, which a new {kind_word} can't copy yet")
        ask = "Want me to draft a Google post built around that this week?"
        step = {"label": "a Google post on your strongest review theme",
                "done": "Drafting the Google post around what your reviewers already say — sending it here for your CONFIRM.",
                "detail": support}
    elif f.identity.get("established_year"):
        yr = f.use("identity.established_year", f.identity["established_year"])
        support = f"{counter} you've been in {f.locality or 'the area'} since {yr}; that trust is the edge."
        ask = "Want me to draft a Google post that leads with that this week?"
        step = {"label": "a Google post on your track record",
                "done": f"Drafting a Google post leading with your {yr}-since track record — sending it here for your CONFIRM.",
                "detail": support}
        if anchor:
            dropped = [anchor] + dropped
            anchor = None
    else:
        support, ask, step = anchor_action(f, anchor) if anchor else ("", "Want me to draft a Google post that sets you apart?", post_step("what sets you apart"))
    return Draft(hook=hook, support=support, ask=ask, anchor=anchor, dropped=dropped,
                 signal=f"competitor_opened ({name or 'unnamed listing'})", judgment=judgment, steps=[step])


def h_perf_dip(f: Facts) -> Draft:
    tp = f.tp
    exclude = []
    if tp.get("metric") and tp.get("delta_pct") is not None:
        metric, d = tp["metric"], f.use("trigger.delta_pct", tp["delta_pct"])
        hook = f"your {humanize(metric)} dropped {fmt_pct(d)} over the last {window_words(tp.get('window'))}"
        base = tp.get("vs_baseline")
        if base:
            now_val = round(base * (1 + d))
            hook += f" — about {now_val} vs your usual {fmt_int(f.use('trigger.vs_baseline', base))}"
        hook += "."
        exclude = [f"{metric}_down"]
    else:
        downs = [(k, v) for k, v in f.delta.items() if v is not None and v < 0]
        if downs:
            k, v = min(downs, key=lambda kv: kv[1])
            metric = k.replace("_pct", "")
            hook = f"your {metric} are down {fmt_pct(f.use(f'performance.delta_7d.{k}', v))} this week."
            exclude = [f"{metric}_down"]
        else:
            calls, avg = f.perf.get("calls"), f.peer.get("avg_calls_30d")
            hook = (f"your listing got {fmt_int(f.use('performance.calls', calls))} calls in the last 30 days vs ~{fmt_int(avg)} for {f.peer_scope()}."
                    if calls is not None and avg else "your listing activity has slowed this week.")
            exclude = ["calls_gap"]
    weights = {"no_offer": 1.0, "sub_expired": 1.0, "unverified": 0.9, "ctr_gap": 0.9, "review_neg": 0.85,
               "calls_gap": 0.6, "sub_ending": 0.6, "live_offer": 0.35, "views_down": 0.4, "calls_down": 0.4}
    anchor, dropped = f.best_anchor(weights, exclude=exclude)
    support, ask, step = anchor_action(f, anchor) if anchor else ("", "Want me to draft a fresh Google post to restart momentum?", post_step("your services"))
    return Draft(hook=hook, support=support, ask=ask, anchor=anchor, dropped=dropped,
                 signal="perf_dip" + (f" ({tp.get('metric')} {fmt_pct(tp['delta_pct'], signed=True)})" if tp.get("delta_pct") is not None else ""),
                 judgment="One fix only — the highest-leverage gap, not a list.", steps=[step])


def h_perf_spike(f: Facts) -> Draft:
    tp = f.tp
    if tp.get("metric") and tp.get("delta_pct") is not None:
        metric, d = tp["metric"], f.use("trigger.delta_pct", tp["delta_pct"])
        hook = f"your {humanize(metric)} are up {fmt_pct(d)} over the last {window_words(tp.get('window'))}"
        if tp.get("vs_baseline"):
            hook += f" (usual ~{fmt_int(f.use('trigger.vs_baseline', tp['vs_baseline']))})"
        driver = tp.get("likely_driver")
        hook += (f" — looks driven by your {humanize(f.use('trigger.likely_driver', driver))}." if driver else ".")
    else:
        ups = [(k, v) for k, v in f.delta.items() if v is not None and v > 0]
        if ups:
            k, v = max(ups, key=lambda kv: kv[1])
            hook = f"your {k.replace('_pct', '')} are up {fmt_pct(f.use(f'performance.delta_7d.{k}', v))} this week — nice momentum."
        else:
            hook = f"your listing pulled {fmt_int(f.use('performance.views', f.perf.get('views', 0)))} views in the last 30 days."
        driver = None
    if driver:
        topic = humanize(driver)
        return Draft(hook=hook, ask=f"Want me to draft 2 more posts in the same style while it's working?",
                     signal=f"perf_spike driven by {topic}", judgment="Double down on the known driver.",
                     steps=[{"label": "2 more posts in the same style",
                             "done": f"Drafting 2 follow-up posts in the style of your {topic} — sending both here for your CONFIRM.",
                             "detail": hook}])
    weights = {"live_offer": 1.0, "no_offer": 0.8, "review_pos": 0.6, "ctr_lead": 0.4}
    anchor, dropped = f.best_anchor(weights)
    if anchor and anchor.type == "live_offer":
        support = sentence(f"{anchor.text} — good moment to pin it while traffic is up")
        ask = "Want me to post it as a Google update today?"
        step = {"label": "a Google post for your live offer",
                "done": "Drafting the Google post for your live offer now — sending it here for your CONFIRM.",
                "detail": support}
    else:
        support, ask, step = anchor_action(f, anchor) if anchor else ("", "Want me to draft a post to ride the momentum?", post_step("your momentum"))
    return Draft(hook=hook, support=support, ask=ask, anchor=anchor, dropped=dropped,
                 signal="perf_spike", steps=[step])


def h_seasonal_dip(f: Facts) -> Draft:
    tp = f.tp
    metric, d = tp.get("metric", "views"), tp.get("delta_pct")
    beat = f.seasonal_beat(["Apr-Jun", "lowest", "lull"])
    hook = f"your {metric} are down {fmt_pct(f.use('trigger.delta_pct', d))} this week" if d is not None else f"your {metric} have dipped"
    if beat:
        hook += f" — but that's the expected {beat['month_range']} pattern ({beat['note']}), not a problem with your listing."
    else:
        hook += " — this matches the seasonal pattern for this time of year."
    dig = f.digest_of_kind(["seasonal"], keywords=["acquisition", "resolution"])
    support = sentence(f"Playbook: {dig['actionable']}") if dig and dig.get("actionable") else ""
    members = f.aggregate.get("total_active_members")
    if members:
        ask = f"Want me to draft a retention challenge for your {fmt_int(f.use('customer_aggregate.total_active_members', members))} members to carry them through the dip?"
    else:
        ask = f"Want me to draft a retention message for your current {f.audience}?"
    return Draft(hook=hook, support=support, ask=ask, signal="seasonal_perf_dip (expected)",
                 judgment="Reframe to pre-empt anxiety; shift effort from acquisition to retention.",
                 steps=[{"label": "a retention challenge",
                         "done": "Drafting a 4-week attendance challenge now — WhatsApp announcement + a leaderboard post. Sending here for your CONFIRM.",
                         "detail": hook}])


def h_milestone(f: Facts) -> Draft:
    tp = f.tp
    if tp.get("value_now") is not None and tp.get("milestone_value"):
        now_v, goal = f.use("trigger.value_now", tp["value_now"]), f.use("trigger.milestone_value", tp["milestone_value"])
        noun = humanize(tp.get("metric", "")).replace(" count", "s")
        gap = goal - now_v
        hook = f"you're at {fmt_int(now_v)} {noun} — just {gap} away from {fmt_int(goal)}." if gap > 0 else f"you just crossed {fmt_int(goal)} {noun}."
        avg = f.peer.get("avg_review_count") if "review" in noun else None
        support = f"You're already ahead of the {fmt_int(avg)} average for {f.peer_scope()}." if avg and now_v > avg else ""
        ask = (f"Want me to draft a short review-request WhatsApp for your regulars to close the last {gap} this week?"
               if gap > 0 else "Want me to draft a thank-you post to mark it?")
        return Draft(hook=hook, support=support, ask=ask, signal=f"milestone_reached ({now_v}/{goal})",
                     steps=[{"label": "a review-request WhatsApp",
                             "done": f"Here's the draft: \"Thanks for choosing {f.merchant_name}! If you enjoyed your last visit, a quick Google review would mean a lot to us 🙏\" Reply CONFIRM and I'll queue it for your regulars.",
                             "detail": hook}])
    n = f.aggregate.get("total_unique_ytd")
    if n:
        hook = f"you've served {fmt_int(f.use('customer_aggregate.total_unique_ytd', n))} unique {f.audience} so far this year — worth marking."
    else:
        hook = f"your listing crossed {fmt_int(f.use('performance.views', f.perf.get('views', 0)))} views in the last 30 days."
    return Draft(hook=hook, ask="Want me to turn it into a thank-you Google post for your regulars?",
                 signal="milestone_reached (derived from merchant aggregate)",
                 steps=[post_step("your milestone")])


def h_dormant(f: Facts) -> Draft:
    tp = f.tp
    if tp.get("days_since_last_merchant_message"):
        days = f.use("trigger.days_since_last_merchant_message", tp["days_since_last_merchant_message"])
        hook = f"it's been {days} days since we last spoke"
        if tp.get("last_topic"):
            hook += f" (last time it was about {humanize(tp['last_topic'])})"
        hook += "."
    else:
        hook = "it's been a while since we caught up."
    weights = {"sub_expired": 1.0, "calls_down": 0.9, "views_down": 0.85, "no_offer": 0.8, "unverified": 0.8,
               "ctr_gap": 0.75, "review_neg": 0.7, "sub_ending": 0.9, "calls_gap": 0.6, "live_offer": 0.3}
    anchor, dropped = f.best_anchor(weights)
    if anchor:
        _, ask, step = anchor_action(f, anchor)
        support = sentence(f"One thing worth 2 minutes: {anchor.text}")
    else:
        support, ask, step = "", "Want me to send you a 3-line snapshot of your listing this week?", post_step("your listing")
    return Draft(hook=hook, support=support,
                 ask=ask, anchor=anchor, dropped=dropped, signal="dormant_with_vera",
                 judgment="Re-open with a single concrete gap, not a catch-up pitch.", steps=[step])


def h_renewal(f: Facts) -> Draft:
    tp, sub, perf = f.tp, f.subscription, f.perf
    days = tp.get("days_remaining", sub.get("days_remaining"))
    plan = tp.get("plan") or sub.get("plan") or "current"
    amount = tp.get("renewal_amount")
    hook = f"your {plan} plan renews in {f.use('days_remaining', days)} days" + (f" (₹{fmt_int(f.use('trigger.renewal_amount', amount))})" if amount else "") + "."
    support = ""
    if perf.get("views") is not None:
        support = (f"Last {perf.get('window_days', 30)} days on the plan: {fmt_int(f.use('performance.views', perf['views']))} profile views, "
                   f"{fmt_int(f.use('performance.calls', perf.get('calls', 0)))} calls, {fmt_int(perf.get('directions', 0))} direction requests.")
    return Draft(hook=hook, support=support, ask="Want me to process the renewal now so there's no gap?",
                 signal=f"renewal_due ({days}d left)", judgment="Show value delivered before asking for the renewal.",
                 steps=[{"label": "process your renewal",
                         "done": "Processing the renewal now — I'll send the payment confirmation here for your CONFIRM.",
                         "detail": support}])


def h_festival(f: Facts) -> Draft:
    tp = f.tp
    fest = tp.get("festival")
    beat = f.seasonal_beat(["festival", "diwali", "wedding"])
    relevant = f.slug in (tp.get("category_relevance") or [f.slug])
    offer_title, offer_src = None, ""
    if beat:
        for word in re.findall(r"[a-z]{5,}", beat["note"].lower()):
            active = f.active_offer_matching([word])
            if active:
                offer_title, offer_src = active, "active"
                break
            cat = f.catalog_offer(keywords=[word], prefer_types=())
            if cat and word in cat["title"].lower():
                offer_title, offer_src = cat["title"], "catalog"
                break
    if not offer_title:
        if f.active_offers:
            offer_title, offer_src = f.active_offers[0], "active"
        else:
            cat = f.catalog_offer()
            offer_title, offer_src = (cat["title"], "catalog") if cat else (None, "")
    if fest:
        days = tp.get("days_until")
        hook = f"{fest} is on {fmt_date(f.use('trigger.date', tp.get('date')), with_year=True)}" + (f" — {f.use('trigger.days_until', days)} days out." if days is not None else ".")
        if days is not None and days > 60:
            judgment = "Too early for a festival discount; the play is positioning before the seasonal peak."
            lead = "Too early for a promo, but the right time to position:"
        else:
            judgment = "Inside the promo window."
            lead = "Good window to set up your festival offer:"
        if not relevant:
            judgment = f"{fest} isn't a core {f.slug} moment; skip the discount, use the season beat instead."
            lead = f"Not a big moment for {f.slug}, so I'd skip a discount —"
    else:
        hook = "festival season is next on the calendar."
        lead = "Worth prepping now:"
        judgment = "Placeholder trigger: anchored on the category seasonal beat."
    support = f"{lead} {beat['month_range']} is the peak window ({beat['note']})." if beat else ""
    if offer_title and offer_src == "active":
        ask = f"Want me to pin your '{offer_title}' with a festival post so it's live before the rush?"
    elif offer_title:
        ask = f"Want me to set up '{offer_title}' on your listing now so it's indexed before the rush?"
    else:
        ask = "Want me to draft a festival post for your listing?"
    return Draft(hook=hook, support=support, ask=ask, signal=f"festival_upcoming ({fest or 'season'})",
                 judgment=judgment + (f" Offer from {offer_src}: {offer_title}." if offer_title else ""),
                 steps=[{"label": f"set up '{offer_title}'" if offer_title else "a festival post",
                         "done": (f"Setting up '{offer_title}' with a festival post now — sending the preview here for your CONFIRM."
                                  if offer_title else "Drafting the festival post now — sending it here for your CONFIRM."),
                         "detail": support}])


def h_ipl(f: Facts) -> Draft:
    tp = f.tp
    iso = tp.get("match_time_iso", "")
    dt = parse_iso(iso)
    hook = f"{tp.get('match', 'tonight’s IPL match')} at {tp.get('venue', f.identity.get('city', ''))} tonight, {fmt_time(iso)}."
    dig = f.digest_of_kind(["seasonal"], keywords=["IPL"])
    weeknight = tp.get("is_weeknight")
    covers = f.vocab("covers", "footfall")
    if dig:
        f.use("digest.id", dig["id"])
    if weeknight is False:
        support = (f"Worth knowing first: {first_sentence(dig['summary']).rstrip('.')} ({dig['source']}). " if dig else "") + \
                  f"So I'd skip a dine-in match promo tonight and push delivery instead."
        pick = None
        for title in f.active_offers:
            if dt is None or not day_restricted_out(title, dt.weekday()):
                pick = title
                break
        if pick:
            ask = f"Want me to draft a delivery-only post for '{pick}' for tonight's match?"
        else:
            cat = f.catalog_offer(keywords=["Delivery"])
            ask = (f"Want me to put '{cat['title']}' live for tonight and draft the post?" if cat
                   else "Want me to draft a delivery-only post for tonight?")
            skipped = [t for t in f.active_offers if dt is not None and day_restricted_out(t, dt.weekday())]
            if skipped:
                support += f" (Your '{skipped[0]}' doesn't run on {dt.strftime('%A')}s, so I've left it out.)"
        judgment = f"Weekend match: digest shows {covers} drop, so contrarian delivery-first call."
    else:
        support = (sentence(f"Weeknight matches have been the strong ones — {dig['actionable']}") if dig else "")
        offer = f.active_offer_matching(["match", "combo"]) or (f.catalog_offer(keywords=["Match-night"]) or {}).get("title")
        ask = f"Want me to push '{offer}' as a match-night post by 5pm?" if offer else "Want me to draft a match-night post by 5pm?"
        judgment = "Weeknight match: lean into the promo."
    return Draft(hook=hook, support=support, ask=ask, signal=f"ipl_match_today ({tp.get('match')})", judgment=judgment,
                 steps=[{"label": "tonight's match post",
                         "done": "Drafting tonight's post now — sending it here for your CONFIRM so it's live before the match.",
                         "detail": support}])


def h_review_theme(f: Facts) -> Draft:
    tp = f.tp
    if tp.get("theme"):
        occ = f.use("trigger.occurrences_30d", tp.get("occurrences_30d"))
        hook = f"{occ} reviews in the last 30 days mention {humanize(tp['theme'])}"
        hook += " — and it's rising" if tp.get("trend") == "rising" else ""
        hook += f". One says: \"{tp['common_quote']}\"." if tp.get("common_quote") else "."
        neg = True
    else:
        themes = sorted(f.review_themes, key=lambda t: (t.get("sentiment") != "neg", -(t.get("occurrences_30d") or 0)))
        if not themes:
            return h_default(f)
        t = themes[0]
        neg = t.get("sentiment") == "neg"
        hook = f"{f.use('review_themes.occurrences_30d', t.get('occurrences_30d'))} reviews in the last 30 days mention {humanize(t['theme'])}"
        hook += f" (\"{t['common_quote']}\")." if t.get("common_quote") else "."
    if neg:
        ask = "Want me to draft calm public replies + a one-line fix note for your team?"
        step = {"label": "review replies + a fix note",
                "done": "Drafting the replies now — polite, specific, no excuses — plus a one-line note for your team. Sending here for your CONFIRM.",
                "detail": hook}
    else:
        ask = "Want me to turn that praise into a Google post this week?"
        step = post_step("what reviewers are praising")
    return Draft(hook=hook, ask=ask, signal="review_theme_emerged", steps=[step])


def h_curious_ask(f: Facts) -> Draft:
    noun = ITEM_NOUN.get(f.slug, "service")
    own_words = [w for title in f.active_offers for w in offer_keywords(title)] + \
                [w for t in f.review_themes for w in str(t.get("theme", "")).split("_") if len(w) > 3]
    tr = f.trend(own_words)
    hook = f"quick one — which {noun} has been most asked-for at {f.merchant_name} this week?"
    support = ""
    if tr:
        support = f"Curious because '{tr['query']}' searches are up {fmt_pct(f.use('trend.delta_yoy', tr['delta_yoy']))} YoY — wondering if you're seeing it too."
    ask = "Reply with the name and I'll turn it into a Google post + a ready WhatsApp reply for price questions. 5 minutes, done."
    return Draft(hook=hook, support=support, ask=ask, cta="open_ended", signal="curious_ask_due",
                 judgment="Asking-the-merchant lever + effort externalization; open-ended by design.",
                 steps=[{"label": "a Google post + price-reply template",
                         "done": "Got it — drafting the Google post and a 4-line WhatsApp price reply around that now. Sending both here for your CONFIRM.",
                         "detail": support}])


def h_planning(f: Facts) -> Draft:
    tp = f.tp
    topic = humanize(tp.get("intent_topic", "the plan"))
    tokens = [w for w in re.findall(r"[a-z]{4,}", topic.lower()) if w not in ("package", "program", "summer")]
    lines = []
    offer = f.active_offer_matching(tokens)
    if offer:
        lines.append(f"• Base: your '{f.use('offers.title', offer)}'" + (f" — keep the {price_in(offer)} price point" if price_in(offer) else ""))
    last_vera = f.last_vera_body() or ""
    m = re.search(r"[Ss]uggest(?:ed)?\s+([^.?!]+)", last_vera)
    if m:
        lines.append(f"• Format (as we discussed): {m.group(1).strip()}")
    addon = f.catalog_offer(keywords=["Delivery", "Free"], prefer_types=())
    if addon and any(w in addon["title"].lower() for w in ("delivery", "free")):
        lines.append(f"• Hook: add '{addon['title']}' (a proven {f.slug} offer) so it's easy to say yes")
    tr = f.trend(tokens)
    if tr and any(t in tr["query"].lower() for t in tokens):
        lines.append(f"• Demand: '{tr['query']}' searches +{fmt_pct(f.use('trend.delta_yoy', tr['delta_yoy']))} YoY ({humanize(tr.get('segment_age', ''))})")
    beat = f.seasonal_beat(tokens + ["holiday", "school"])
    if beat and not tr:
        lines.append(f"• Timing: {beat['month_range']} — {beat['note']}")
    quote = tp.get("merchant_last_message")
    hook = f"on your \"{quote}\" — here's a first cut of the {topic}:" if quote else f"here's a first cut of the {topic}:"
    draft_block = "\n" + "\n".join(lines) + "\n" if lines else " "
    return Draft(hook=hook + draft_block, ask="Want me to publish this as a Google post and draft the WhatsApp announcement?",
                 signal=f"active_planning_intent ({topic})",
                 judgment="Merchant already said yes — deliver a draft, no qualifying questions.",
                 steps=[{"label": "publish the post + WhatsApp announcement",
                         "done": f"Drafting the Google post and WhatsApp announcement for the {topic} now — sending both here for your CONFIRM before anything goes live.",
                         "detail": "\n".join(lines)}])


def h_gbp_unverified(f: Facts) -> Draft:
    tp = f.tp
    uplift = tp.get("estimated_uplift_pct")
    hook = f"your Google profile for {f.merchant_name} is still unverified."
    views = f.perf.get("views")
    support = ""
    if views is not None:
        support = f"You're already getting {fmt_int(f.use('performance.views', views))} views a month without it"
        support += f" — verification is estimated to add ~{fmt_pct(f.use('trigger.estimated_uplift_pct', uplift))} on top." if uplift else "."
    path = humanize(tp.get("verification_path", "")).replace(" or ", " or ")
    ask = f"Want me to start the verification ({path}) now?" if path else "Want me to start the verification now?"
    return Draft(hook=hook, support=support, ask=ask, signal="gbp_unverified",
                 steps=[{"label": "start Google verification",
                         "done": f"Starting the verification request now ({path}) — I'll tell you exactly what to do when Google reaches you.",
                         "detail": support}])


def h_category_seasonal(f: Facts) -> Draft:
    tp = f.tp
    moves = []
    for t in tp.get("trends") or []:
        m = re.match(r"(.+?)_demand_([+-]\d+)", t)
        if m:
            moves.append(f"{m.group(1).replace('_', ' ')} {m.group(2)}%")
    season = humanize(tp.get("season", "this season")).replace(" 2026", "")
    hook = f"{season} demand shift is on: {list_words(moves)}." if moves else f"{season} demand shift is on."
    dig = f.digest_of_kind(["seasonal"], keywords=["summer", "monsoon"])
    support = (f"Suggested shelf move: {dig['actionable'][0].lower() + dig['actionable'][1:]} ({dig['source']})."
               if dig and dig.get("actionable") else "")
    return Draft(hook=hook, support=support,
                 ask="Want me to post a 'summer essentials in stock' update on your Google listing today?",
                 signal=f"category_seasonal ({season})",
                 steps=[{"label": "a seasonal-stock Google post",
                         "done": "Drafting the 'in stock now' Google post — sending it here for your CONFIRM.",
                         "detail": support}])


def h_supply_alert(f: Facts) -> Draft:
    tp = f.tp
    item = f.digest_item(tp.get("alert_id"))
    batches = ", ".join(tp.get("affected_batches") or [])
    hook = f"urgent — voluntary recall on {tp.get('molecule', 'a molecule')} batches {batches} ({tp.get('manufacturer', 'manufacturer')})."
    if item:
        f.use("digest.id", item["id"])
        m = re.search(r"flagged for ([\w-]+)", item.get("summary", ""))
        risk = next((s for s in re.split(r"(?<=[.!?])\s+", item.get("summary", "")) if "risk" in s.lower()), "")
        hook += f" Flagged for {m.group(1)}" + (f"; {risk[0].lower() + risk[1:]}" if risk else ".") if m else ""
        hook += f" Source: {item.get('source')}."
    n = f.aggregate.get("chronic_rx_count")
    support = f"Worth checking against your {fmt_int(f.use('customer_aggregate.chronic_rx_count', n))} chronic-Rx customers today." if n else ""
    return Draft(hook=hook, support=support,
                 ask="Want me to draft the customer WhatsApp + replacement-pickup note now?",
                 signal=f"supply_alert ({tp.get('molecule')}, urgency {f.trigger.get('urgency')})",
                 judgment="Urgent compliance; bounded-risk framing, no alarmism.",
                 steps=[{"label": "customer WhatsApp + replacement note",
                         "done": f"Here's the draft: \"Namaste, {f.merchant_name} here. A batch of {tp.get('molecule')} you may have received is under a voluntary recall ({batches}). Please bring it in — we'll replace it at no cost.\" Reply CONFIRM and I'll queue it for affected customers.",
                         "detail": hook}])


def h_winback(f: Facts) -> Draft:
    tp = f.tp
    d, dip, n = tp.get("days_since_expiry"), tp.get("perf_dip_pct"), tp.get("lapsed_customers_added_since_expiry")
    parts = []
    if dip is not None:
        parts.append(f"views are down {fmt_pct(f.use('trigger.perf_dip_pct', dip))}")
    if n:
        parts.append(f"{f.use('trigger.lapsed_customers_added_since_expiry', n)} more {f.audience} have lapsed")
    hook = f"since your plan lapsed {f.use('trigger.days_since_expiry', d)} days ago, " + (list_words(parts) if parts else "activity has slowed") + "."
    ask = f"Want me to restart the plan and send a win-back note to those {n}?" if n else "Want me to restart the plan today?"
    return Draft(hook=hook, ask=ask, signal="winback_eligible", judgment="Loss framing from trigger payload.",
                 steps=[{"label": "restart plan + win-back note",
                         "done": "Starting the plan restart and drafting the win-back note now — sending both here for your CONFIRM.",
                         "detail": hook}])


def h_default(f: Facts) -> Draft:
    weights = {"no_offer": 1.0, "ctr_gap": 0.9, "calls_down": 0.9, "review_neg": 0.8, "unverified": 0.8,
               "sub_ending": 0.8, "sub_expired": 0.9, "views_down": 0.7, "live_offer": 0.5, "cohort": 0.5, "review_pos": 0.5}
    anchor, dropped = f.best_anchor(weights)
    support, ask, step = anchor_action(f, anchor) if anchor else ("", "Want me to draft a Google post for you this week?", post_step("your business"))
    hook = f"quick update on {humanize(f.kind)}."
    return Draft(hook=hook, support=support.replace("For context, ", "").replace("Also, ", ""), ask=ask,
                 anchor=anchor, dropped=dropped, signal=f"{f.kind} (generic handler)", steps=[step])


# ================================================================ customer-facing handlers
def greeting(f: Facts) -> str:
    name = f.cust_name or "there"
    senior = f.cust_identity.get("senior_citizen") or (f.cust_identity.get("age_band", "")[:1] in ("6", "7", "8"))
    emoji = CUSTOMER_EMOJI.get(f.slug, "")
    if senior or (f.cust_prefs.get("channel") or "").endswith("via_son"):
        return f"Namaste — {f.merchant_name} ({f.locality}) {'yahan' if f.cust_hinglish else 'here'}." if f.locality else f"Namaste — {f.merchant_name} here."
    return f"Hi {name}, {f.merchant_name} here {emoji}".strip()


def c_recall(f: Facts) -> Draft:
    tp, hi = f.tp, f.cust_hinglish
    service = humanize(tp["service_due"]) if tp.get("service_due") else (humanize(f.last_service()) if f.last_service() else "next visit")
    last = tp.get("last_service_date") or f.cust_rel.get("last_visit")
    last_txt = fmt_date(f.use("last_visit", last), with_year=True) if last else ""
    slots = [s.get("label") for s in tp.get("available_slots") or [] if s.get("label")]
    offer = f.active_offer_matching(re.findall(r"[a-z]{5,}", service.lower()))
    if hi:
        hook = f"Aapka {service} due hai" + (f" — pichhla visit {last_txt} ko tha." if last_txt else ".")
        if slots:
            hook += f" Aapke liye {' ya '.join(slots[:2])} ka slot ready hai."
    else:
        hook = f"Your {service} is due" + (f" — your last one was on {last_txt}." if last_txt else ".")
        if slots:
            hook += f" We've kept {' or '.join(slots[:2])} open for you."
    support = f"Offer: {offer}." if offer else ""
    if len(slots) >= 2:
        ask = (f"{slots[0]} ke liye 1, {slots[1]} ke liye 2 reply karein — ya apna time batayein." if hi
               else f"Reply 1 for {slots[0]}, 2 for {slots[1]}, or tell us a time that suits you.")
        cta = "multi_choice_slot"
    else:
        pref = humanize(f.cust_prefs.get("preferred_slots", "")) if f.cust_prefs.get("preferred_slots") else ""
        ask = (f"YES reply karein, hum aapke liye {pref} slot book kar denge." if hi and pref else
               "YES reply karein, hum slot book kar denge." if hi else
               f"Reply YES and we'll book a {pref} slot for you." if pref else "Reply YES and we'll book a slot for you.")
        cta = "binary_yes_no"
    return Draft(hook=hook, support=support, ask=ask, cta=cta, customer_facing=True, options=slots[:2],
                 signal=f"recall_due ({service})",
                 judgment=f"Language pref={f.cust_identity.get('language_pref')}; preferred_slots={f.cust_prefs.get('preferred_slots', 'n/a')}.",
                 steps=[{"label": "book the slot", "done": "Booked ✅ We'll send a reminder the day before. See you soon!"}])


def c_appointment(f: Facts) -> Draft:
    hi = f.cust_hinglish
    stylist = f.cust_prefs.get("preferred_stylist")
    where = f" ({f.locality})" if f.locality else ""
    if hi:
        hook = f"Yaad dila rahe hain — kal {f.merchant_name}{where} mein aapka appointment hai" + (f", {stylist} ke saath." if stylist else ".")
        ask = "Confirm karne ke liye YES reply karein, ya time badalna ho toh batayein."
    else:
        hook = f"Quick reminder — your appointment at {f.merchant_name}{where} is tomorrow" + (f" with {stylist}." if stylist else ".")
        ask = "Reply YES to confirm, or tell us if you'd like to reschedule."
    svc = f.last_service()
    support = (f"Pichhli baar aapne {humanize(svc)} karwaya tha." if hi else f"Last time you came in for {humanize(svc)}.") if svc else ""
    return Draft(hook=hook, support=support, ask=ask, customer_facing=True, signal="appointment_tomorrow",
                 judgment="Placeholder payload: no time given, so none stated.",
                 steps=[{"label": "confirm the appointment", "done": "Confirmed ✅ See you tomorrow!"}])


def c_refill(f: Facts) -> Draft:
    tp, hi = f.tp, f.cust_hinglish
    mols = tp.get("molecule_list") or []
    runs_out = fmt_date(tp["stock_runs_out_iso"]) if tp.get("stock_runs_out_iso") else ""
    name = f.cust_name or ""
    senior_offer = f.active_offer_matching(["Senior"]) if f.cust_identity.get("senior_citizen") else None
    delivery_offer = f.active_offer_matching(["Delivery"]) if (tp.get("delivery_address_saved") or f.cust_prefs.get("delivery_address")) else None
    if mols:
        if hi:
            hook = f"{name} ki {len(mols)} regular medicines ({', '.join(mols)})" + (f" {runs_out} ko khatam hongi." if runs_out else " refill ke liye due hain.")
            hook += " Same dose, same brand ka pack ready rakh sakte hain."
        else:
            hook = f"{name}'s {len(mols)} regular medicines ({', '.join(mols)})" + (f" run out on {runs_out}." if runs_out else " are due for refill.")
            hook += " Same dose, same brand pack can be kept ready."
    else:
        svc = humanize(f.last_service()) if f.last_service() else ("refill" if f.slug == "pharmacies" else "follow-up visit")
        last = fmt_date(f.cust_rel["last_visit"], with_year=True) if f.cust_rel.get("last_visit") else ""
        hook = (f"Aapka {svc} due hai" + (f" — pichhli baar {last} ko aaye the." if last else ".")) if hi else \
               (f"Your {svc} is due" + (f" — your last visit was on {last}." if last else "."))
    perks = [p for p in [senior_offer, delivery_offer] if p]
    support = (f"Aapke liye: {' + '.join(perks)}." if hi else f"Applies: {' + '.join(perks)}.") if perks else ""
    if mols:
        ask = ("Dispatch ke liye CONFIRM reply karein, ya dose mein koi badlav ho toh batayein." if hi
               else "Reply CONFIRM to dispatch, or tell us if the dosage has changed.")
    else:
        ask = "YES reply karein, hum ready rakhenge." if hi else "Reply YES and we'll keep it ready for you."
    return Draft(hook=hook, support=support, ask=ask, cta="binary_confirm_cancel" if mols else "binary_yes_no",
                 customer_facing=True, signal="chronic_refill_due",
                 judgment="Only active merchant offers mentioned; no totals computed (no prices in context).",
                 steps=[{"label": "dispatch the refill", "done": "Done ✅ Dispatching to your saved address — we'll message when it's out for delivery." if mols else "Done ✅ We'll keep it ready for you."}])


def c_lapsed(f: Facts) -> Draft:
    tp, hi = f.tp, f.cust_hinglish
    days = tp.get("days_since_last_visit")
    last = f.cust_rel.get("last_visit")
    if days:
        weeks = round(days / 7)
        when = f"{weeks} hafte" if hi else f"about {weeks} weeks"
    else:
        when = None
    focus = humanize(tp["previous_focus"]) if tp.get("previous_focus") else None
    if hi:
        hook = (f"Aapko aaye {when} ho gaye" if when else f"Aapko {fmt_date(last, with_year=True)} ke baad nahi dekha" if last else "Kaafi time ho gaya") + " — koi baat nahi, hota hai."
        if focus:
            hook += f" Aapka {focus} goal abhi bhi hamare dimaag mein hai."
    else:
        hook = (f"It's been {when} since your last visit" if when else f"We haven't seen you since {fmt_date(last, with_year=True)}" if last else "It's been a while") + " — happens to everyone, no judgment."
        if focus:
            hook += f" Your {focus} goal is still on our radar."
    offer = next((t for t in f.active_offers if "free" in t.lower()), None) or (f.active_offers[0] if f.active_offers else None)
    support = (f"Wapas shuru karne ke liye abhi chal raha hai: {offer}." if hi else
               f"If it helps you ease back in, we're running {offer} right now.") if offer else ""
    ask = "Ek slot hold karein? YES reply karein — koi commitment nahi." if hi else "Want us to hold a slot for you this week? Reply YES — no commitment."
    return Draft(hook=hook, support=support, ask=ask, customer_facing=True, signal=f"{f.kind}",
                 judgment="No-shame winback; offer only from merchant's active list.",
                 steps=[{"label": "hold a slot", "done": "Done ✅ Slot held — we'll confirm the time with you shortly."}])


def c_trial(f: Facts) -> Draft:
    tp, hi = f.tp, f.cust_hinglish
    trial = fmt_date(tp["trial_date"]) if tp.get("trial_date") else None
    opts = [o.get("label") for o in tp.get("next_session_options") or [] if o.get("label")]
    if hi:
        hook = (f"{trial} ko trial ke liye shukriya!" if trial else "Trial ke liye shukriya!") + (f" Agla session: {' ya '.join(opts[:2])}." if opts else "")
        ask = (f"{opts[0]} ke liye 1, {opts[1]} ke liye 2 reply karein." if len(opts) >= 2 else "Book karne ke liye YES reply karein.")
    else:
        hook = (f"Thanks for trying a session with us on {trial}!" if trial else "Thanks for trying a session with us!") + (f" Next one's open: {' or '.join(opts[:2])}." if opts else "")
        ask = (f"Reply 1 for {opts[0]}, 2 for {opts[1]}." if len(opts) >= 2 else "Reply YES to lock it in.")
    offer = f.active_offers[0] if f.active_offers else None
    support = (f"Join karna ho toh: {offer}." if hi else f"If you'd like to continue: {offer}.") if offer else ""
    return Draft(hook=hook, support=support, ask=ask, cta="multi_choice_slot" if len(opts) >= 2 else "binary_yes_no",
                 customer_facing=True, options=opts[:2], signal="trial_followup",
                 steps=[{"label": "book the session", "done": "Booked ✅ See you there!"}])


def c_wedding(f: Facts) -> Draft:
    tp, hi = f.tp, f.cust_hinglish
    days = tp.get("days_to_wedding")
    raw = str(tp.get("next_step_window_open", "next step"))
    m = re.search(r"_?(\d+)day$", raw)
    program = (f"{m.group(1)}-day " if m else "") + humanize(re.sub(r"_?\d+day$", "", raw))
    trial = fmt_date(tp["trial_completed"]) if tp.get("trial_completed") else None
    wd = fmt_date(tp["wedding_date"], with_year=True) if tp.get("wedding_date") else None
    hook = (f"{days} days to your wedding" + (f" on {wd}" if wd else "") + "! 💍" if days is not None else "Your big day is getting closer! 💍")
    hook += (f" Since your bridal trial on {trial}, now's the right window to start the {program}." if trial else f" Now's the right window to start the {program}.")
    offer = f.active_offer_matching(["bridal", "skin", "facial"])
    support = f"{offer}." if offer else ""
    pref = f.cust_prefs.get("preferred_slots")
    pref_txt = re.sub(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
                      lambda d: d.group(1).capitalize(), humanize(pref)) if pref else ""
    ask = f"Want us to block your {pref_txt} slot for the first session? Reply YES." if pref else "Want us to block your first session? Reply YES."
    return Draft(hook=hook, support=support, ask=ask, customer_facing=True, signal="wedding_package_followup",
                 steps=[{"label": "block the first session", "done": "Done ✅ First session blocked — we'll confirm the exact time shortly."}])


def c_default(f: Facts) -> Draft:
    hi = f.cust_hinglish
    last = f.cust_rel.get("last_visit")
    hook = (f"Aapki pichhli visit {fmt_date(last, with_year=True)} ko thi — aapko dobara dekhna accha lagega." if hi else
            f"Your last visit was on {fmt_date(last, with_year=True)} — we'd love to see you again.") if last else \
           ("Aapko dobara dekhna accha lagega." if hi else "We'd love to see you again.")
    offer = f.active_offers[0] if f.active_offers else None
    support = f"{offer}." if offer else ""
    ask = "YES reply karein, hum slot book kar denge." if hi else "Reply YES and we'll book a slot for you."
    return Draft(hook=hook, support=support, ask=ask, customer_facing=True, signal=f"{f.kind} (generic customer handler)",
                 steps=[{"label": "book a slot", "done": "Done ✅ We'll confirm the time shortly."}])


def m_customer_missing(f: Facts) -> Draft:
    return Draft(hook=f"one of your {f.audience} has a {humanize(f.kind)} coming up.",
                 ask="Want me to send them a reminder on your behalf?",
                 signal=f"{f.kind} (customer context missing — routed to merchant)",
                 judgment="Customer-scoped trigger without CustomerContext; never invent a name.",
                 steps=[{"label": "send the reminder", "done": "Sending the reminder on your behalf now."}])


MERCHANT_HANDLERS: dict[str, Callable[[Facts], Draft]] = {
    "research_digest": h_research_digest, "regulation_change": h_regulation, "cde_opportunity": h_cde,
    "competitor_opened": h_competitor, "perf_dip": h_perf_dip, "perf_spike": h_perf_spike,
    "seasonal_perf_dip": h_seasonal_dip, "milestone_reached": h_milestone, "dormant_with_vera": h_dormant,
    "renewal_due": h_renewal, "festival_upcoming": h_festival, "ipl_match_today": h_ipl,
    "review_theme_emerged": h_review_theme, "curious_ask_due": h_curious_ask,
    "active_planning_intent": h_planning, "gbp_unverified": h_gbp_unverified,
    "category_seasonal": h_category_seasonal, "supply_alert": h_supply_alert, "winback_eligible": h_winback,
}
CUSTOMER_HANDLERS: dict[str, Callable[[Facts], Draft]] = {
    "recall_due": c_recall, "appointment_tomorrow": c_appointment, "chronic_refill_due": c_refill,
    "customer_lapsed_soft": c_lapsed, "customer_lapsed_hard": c_lapsed, "trial_followup": c_trial,
    "wedding_package_followup": c_wedding,
}


# ================================================================ assembly
def _clean(text: str) -> str:
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" +([.,;:!?])", r"\1", text)
    text = re.sub(r"\.\.+", ".", text)
    text = re.sub(r" *\n *", "\n", text)
    return text.strip()


def _scrub_taboos(f: Facts, text: str) -> tuple[str, list[str]]:
    hits = []
    for taboo in f.taboos:
        if taboo and re.search(re.escape(taboo), text, flags=re.I):
            hits.append(taboo)
            text = re.sub(re.escape(taboo), "", text, flags=re.I)
    return text, hits


def _lower_first(text: str) -> str:
    if not text or re.match(r"^[A-Z]{2,}|^[A-Z][a-z]+[A-Z]|^Dr\.", text):
        return text
    first = text.split(" ", 1)[0]
    if first.lower() in ("your", "it's", "quick", "heads-up", "since", "you're", "you've", "on", "here's",
                         "compliance", "festival", "urgent", "one", "picking"):
        return text[0].lower() + text[1:]
    return text


def _merchant_tail(f: Facts, draft: Draft) -> str:
    if draft.cta != "binary_yes_no" or not f.merchant_code_mix:
        return ""
    return " Haan bolein toh main shuru kar deti hoon."


def _rationale(f: Facts, d: Draft, cta_note: str) -> str:
    parts = [f"Signal: {d.signal} (urgency {f.trigger.get('urgency', '?')}, {f.trigger.get('source', '?')})."]
    if d.anchor:
        runner = ", ".join(a.type for a in d.dropped[:3])
        parts.append(f"Anchor: {d.anchor.type} [{d.anchor.provenance}]" + (f", chosen over {runner}." if runner else "."))
    if d.judgment:
        parts.append(d.judgment)
    parts.append(cta_note)
    if f.used:
        parts.append("Grounded: " + "; ".join(f.used[:8]) + ".")
    return " ".join(parts)


def compose_full(category: dict, merchant: dict, trigger: dict,
                 customer: Optional[dict] = None) -> ComposedMessage:
    f = Facts(category, merchant, trigger, customer)
    kind = f.kind
    customer_scoped = trigger.get("scope") == "customer" or kind in CUSTOMER_HANDLERS

    if customer_scoped and customer:
        draft = CUSTOMER_HANDLERS.get(kind, c_default)(f)
    elif customer_scoped:
        draft = m_customer_missing(f)
    else:
        draft = MERCHANT_HANDLERS.get(kind, h_default)(f)

    if draft.customer_facing:
        send_as = "merchant_on_behalf"
        opener = greeting(f)
        sep = " " if opener.endswith((".", "!", "✨", "🦷", "💪", "💊", "🍽️")) else ". "
        body = f"{opener}{sep}{draft.hook} {draft.support} {draft.ask}"
        name_param = f.cust_name or "there"
        template = f"merchant_{kind}_v1"
    else:
        send_as = "vera"
        sal = f.salutation()
        opener = f"{sal}," if sal.lower().startswith("dr") else f"Hi {sal},"
        hook = _lower_first(draft.hook)
        body = f"{opener} {hook} {draft.support} {draft.ask}{_merchant_tail(f, draft)}"
        name_param = sal
        template = f"vera_{kind}_v1"

    body = _clean(body)
    body, taboo_hits = _scrub_taboos(f, body)
    body = _clean(body)

    cta_notes = {"binary_yes_no": "CTA: single YES/NO as the last line.",
                 "binary_confirm_cancel": "CTA: single CONFIRM (action flow).",
                 "multi_choice_slot": "CTA: slot choice (booking flow) with free-text fallback.",
                 "open_ended": "CTA: open-ended ask (curiosity / asking-the-merchant)."}
    rationale = _rationale(f, draft, cta_notes.get(draft.cta, f"CTA: {draft.cta}."))
    if taboo_hits:
        rationale += f" Taboo scrubbed: {taboo_hits}."

    suppression_key = trigger.get("suppression_key") or \
        f"{kind}:{merchant.get('merchant_id', '')}:{(customer or {}).get('customer_id', '')}".rstrip(":")
    detail = _clean(f"{draft.hook} {draft.support}")
    params = [name_param, _clean(f"{draft.hook} {draft.support}"), _clean(draft.ask)]
    return ComposedMessage(body=body, cta=draft.cta, send_as=send_as, suppression_key=suppression_key,
                           rationale=rationale, template_name=template, template_params=params,
                           topic=humanize(kind), detail=detail, steps=draft.steps, options=draft.options)


def compose(category: dict, merchant: dict, trigger: dict, customer: Optional[dict] = None) -> dict[str, Any]:
    """Returns {body, cta, send_as, suppression_key, rationale}. Deterministic for identical inputs."""
    return compose_full(category, merchant, trigger, customer).public()


def default_steps(category: Optional[dict], merchant: Optional[dict]) -> list[dict[str, str]]:
    """Next steps for conversations the bot didn't open itself (e.g. replay tests)."""
    if not merchant:
        return [{"label": "the next step", "done": "On it — drafting the next step now and sending it here for your CONFIRM before anything goes live."}]
    f = Facts(category or {}, merchant, {"kind": "reply"}, None)
    weights = {"no_offer": 1.0, "ctr_gap": 0.9, "review_neg": 0.8, "unverified": 0.8, "live_offer": 0.6,
               "sub_ending": 0.8, "sub_expired": 0.9}
    anchor, _ = f.best_anchor(weights)
    if anchor is None:
        return [post_step("your business")]
    return [anchor_action(f, anchor)[2]]
