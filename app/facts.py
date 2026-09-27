"""Grounded accessors over the 4 contexts.

Every value the composer puts in a message body is read through `Facts`, which records the
source field path in `self.used`. The rationale lists those paths, so any number in a body
can be traced to the input that produced it.
"""
import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from .timeutil import parse_iso

AUDIENCE_NOUN = {"dentists": "patients", "gyms": "members", "salons": "clients",
                 "pharmacies": "customers", "restaurants": "customers"}
CUSTOMER_EMOJI = {"dentists": "🦷", "salons": "✨", "gyms": "💪", "pharmacies": "💊",
                  "restaurants": "🍽️"}
COHORT_PHRASES = {
    "high_risk_adult_count": "{n} high-risk adult patients",
    "chronic_rx_count": "{n} chronic-Rx customers",
    "total_active_members": "{n} active members",
    "lapsed_180d_plus": "{n} {aud} who haven't visited in 6+ months",
    "lapsed_90d_plus": "{n} {aud} who haven't visited in 90+ days",
}
HINDI_TOKENS = {"hai", "haan", "han", "nahi", "nahin", "kya", "karo", "kar", "mujhe", "aap", "aapka",
                "theek", "thik", "chalo", "bhejo", "bhej", "ji", "abhi", "baad", "mein", "kaise",
                "kitna", "kitne", "bataiye", "batao", "accha", "acha", "zaroor", "karna", "hoga",
                "ke", "liye", "yeh", "hoon", "hain", "aapki", "sabhi", "shukriya", "bahut", "kripya",
                "kijiye", "dijiye", "karein", "nahin", "chahiye", "kal", "haanji", "bilkul"}


# ---------------------------------------------------------------- formatting
def fmt_int(n: Any) -> str:
    try:
        return f"{int(round(float(n))):,}"
    except (TypeError, ValueError):
        return str(n)


def fmt_pct(x: Any, signed: bool = False) -> str:
    """Fraction -> percent. 0.021 -> '2.1%', 0.38 -> '38%'."""
    v = float(x) * 100
    mag = abs(v)
    text = f"{mag:.1f}".rstrip("0").rstrip(".") if mag < 10 and mag != int(mag) else f"{round(mag):d}"
    if signed:
        return ("+" if v >= 0 else "-") + text + "%"
    return text + "%"


def fmt_date(value: Any, with_year: bool = False) -> str:
    dt = parse_iso(value) if isinstance(value, str) else None
    if dt is None:
        return str(value)
    return f"{dt.day} {dt.strftime('%b')}" + (f" {dt.year}" if with_year else "")


def humanize(token: Any) -> str:
    text = str(token).replace("_", " ").strip()
    text = re.sub(r"(\d+) (month|day|week|year)", r"\1-\2", text)
    return text


def first_sentence(text: str, max_len: int = 170) -> str:
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    out = parts[0]
    if len(out) > max_len:
        out = out[: max_len].rsplit(" ", 1)[0] + "…"
    return out


def strip_trailing_punct(text: str) -> str:
    return text.rstrip(" .;:—-")


def is_hinglish_text(text: str) -> bool:
    if re.search(r"[ऀ-ॿ]", text or ""):
        return True
    tokens = re.findall(r"[a-z]+", (text or "").lower())
    hits = len(set(tokens) & HINDI_TOKENS)
    return hits >= 2 or (hits == 1 and len(tokens) <= 3)


def price_in(title: str) -> Optional[str]:
    m = re.search(r"₹\s?[\d,]+", title or "")
    return m.group(0).replace(" ", "") if m else None


@dataclass
class Anchor:
    type: str
    magnitude: float
    text: str
    provenance: str


class Facts:
    def __init__(self, category: dict, merchant: dict, trigger: dict,
                 customer: Optional[dict] = None) -> None:
        self.category = category or {}
        self.merchant = merchant or {}
        self.trigger = trigger or {}
        self.customer = customer
        self.used: list[str] = []

    # ------------------------------------------------------------ provenance
    def use(self, path: str, value: Any) -> Any:
        entry = f"{path}={value}"
        if entry not in self.used:
            self.used.append(entry)
        return value

    # ------------------------------------------------------------ category
    @property
    def slug(self) -> str:
        return self.category.get("slug") or self.merchant.get("category_slug") or ""

    @property
    def voice(self) -> dict:
        return self.category.get("voice") or {}

    @property
    def tone(self) -> str:
        return self.voice.get("tone") or ""

    @property
    def taboos(self) -> list[str]:
        return [re.sub(r"\s*\(.*\)$", "", t).strip() for t in
                (self.voice.get("vocab_taboo") or self.voice.get("taboos") or []) if t]

    def vocab(self, term: str, fallback: str) -> str:
        allowed = [v.lower() for v in self.voice.get("vocab_allowed") or []]
        return term if term.lower() in allowed else fallback

    @property
    def merchant_code_mix(self) -> bool:
        mix = (self.voice.get("code_mix") or "").lower()
        return mix.startswith("hindi_english") and "hi" in self.languages

    @property
    def audience(self) -> str:
        return AUDIENCE_NOUN.get(self.slug, "customers")

    @property
    def peer(self) -> dict:
        return self.category.get("peer_stats") or {}

    def peer_scope(self) -> str:
        scope = self.peer.get("scope") or ""
        scope = re.sub(r"_?\d{4}$", "", scope)
        return humanize(scope) if scope else f"similar {self.slug}"

    def digest_item(self, item_id: Optional[str]) -> Optional[dict]:
        for item in self.category.get("digest") or []:
            if item_id and item.get("id") == item_id:
                return item
        return None

    def digest_of_kind(self, kinds: Iterable[str], keywords: Iterable[str] = ()) -> Optional[dict]:
        kinds = list(kinds)
        items = self.category.get("digest") or []
        for kw in keywords:
            for item in items:
                if kw.lower() in (item.get("title", "") + " " + item.get("summary", "")).lower():
                    return item
        for kind in kinds:
            for item in items:
                if item.get("kind") == kind:
                    return item
        return None

    def catalog_offer(self, keywords: Iterable[str] = (), audiences: Iterable[str] = ("new_user", "all"),
                      prefer_types: Iterable[str] = ("service_at_price", "free_service", "free_trial")) -> Optional[dict]:
        """Service+price templates beat percentage discounts (brief §3 pain point #3)."""
        catalog = self.category.get("offer_catalog") or []
        audiences = list(audiences)
        for kw in keywords:
            for offer in catalog:
                if kw.lower() in offer.get("title", "").lower():
                    return offer
        for t in prefer_types:
            for offer in catalog:
                if offer.get("type") == t and offer.get("audience") in audiences:
                    return offer
        return catalog[0] if catalog else None

    def seasonal_beat(self, keywords: Iterable[str]) -> Optional[dict]:
        for kw in keywords:
            for beat in self.category.get("seasonal_beats") or []:
                if kw.lower() in (beat.get("note", "") + " " + beat.get("month_range", "")).lower():
                    return beat
        return None

    def trend(self, keywords: Iterable[str] = ()) -> Optional[dict]:
        trends = self.category.get("trend_signals") or []
        for kw in keywords:
            for t in trends:
                if kw.lower() in t.get("query", "").lower():
                    return t
        return max(trends, key=lambda t: t.get("delta_yoy", 0)) if trends else None

    # ------------------------------------------------------------ merchant
    @property
    def identity(self) -> dict:
        return self.merchant.get("identity") or {}

    @property
    def merchant_name(self) -> str:
        return self.identity.get("name") or "your business"

    @property
    def owner_first(self) -> Optional[str]:
        return self.identity.get("owner_first_name")

    @property
    def locality(self) -> Optional[str]:
        return self.identity.get("locality")

    @property
    def languages(self) -> list[str]:
        return self.identity.get("languages") or ["en"]

    def salutation(self) -> str:
        first = self.owner_first
        if not first:
            return f"{self.merchant_name} team"
        for example in self.voice.get("salutation_examples") or []:
            if "{" in example:
                sal = re.sub(r"\{[^}]+\}", first, example)
                return sal if sal.lower().startswith("dr") else re.sub(r"^(hi|hello)\s+", "", sal, flags=re.I)
        return first

    @property
    def perf(self) -> dict:
        return self.merchant.get("performance") or {}

    @property
    def delta(self) -> dict:
        return self.perf.get("delta_7d") or {}

    @property
    def offers(self) -> list[dict]:
        return self.merchant.get("offers") or []

    @property
    def active_offers(self) -> list[str]:
        return [o["title"] for o in self.offers if o.get("status") == "active" and o.get("title")]

    def active_offer_matching(self, keywords: Iterable[str]) -> Optional[str]:
        for kw in keywords:
            for title in self.active_offers:
                if kw.lower() in title.lower():
                    return title
        return None

    @property
    def aggregate(self) -> dict:
        return self.merchant.get("customer_aggregate") or {}

    @property
    def signals(self) -> list[str]:
        return self.merchant.get("signals") or []

    @property
    def review_themes(self) -> list[dict]:
        return self.merchant.get("review_themes") or []

    @property
    def subscription(self) -> dict:
        return self.merchant.get("subscription") or {}

    def last_vera_body(self) -> Optional[str]:
        for turn in reversed(self.merchant.get("conversation_history") or []):
            if turn.get("from") == "vera":
                return turn.get("body")
        return None

    # ------------------------------------------------------------ trigger
    @property
    def kind(self) -> str:
        return self.trigger.get("kind") or "unknown"

    @property
    def tp(self) -> dict:
        return self.trigger.get("payload") or {}

    @property
    def is_placeholder(self) -> bool:
        return bool(self.tp.get("placeholder")) or not self.tp

    # ------------------------------------------------------------ customer
    @property
    def cust_identity(self) -> dict:
        return (self.customer or {}).get("identity") or {}

    @property
    def cust_name(self) -> Optional[str]:
        return self.cust_identity.get("name")

    @property
    def cust_rel(self) -> dict:
        return (self.customer or {}).get("relationship") or {}

    @property
    def cust_prefs(self) -> dict:
        return (self.customer or {}).get("preferences") or {}

    @property
    def cust_hinglish(self) -> bool:
        pref = (self.cust_identity.get("language_pref") or "").lower()
        return pref in ("hi", "hindi") or pref.startswith("hi-") or "hi-en" in pref

    def last_service(self) -> Optional[str]:
        services = [s for s in self.cust_rel.get("services_received") or [] if s and s != "..."]
        return services[-1] if services else None

    # ------------------------------------------------------------ anchors
    def merchant_anchors(self) -> list[Anchor]:
        """Candidate supporting facts from merchant state, each with a magnitude in [0, 1]."""
        out: list[Anchor] = []
        perf, peer, delta, aud = self.perf, self.peer, self.delta, self.audience
        scope = self.peer_scope()

        ctr, avg_ctr = perf.get("ctr"), peer.get("avg_ctr")
        if ctr is not None and avg_ctr:
            if ctr < avg_ctr:
                out.append(Anchor("ctr_gap", min(1.0, (avg_ctr - ctr) / avg_ctr * 2),
                                  f"your profile click-through is {fmt_pct(ctr)} vs {fmt_pct(avg_ctr)} for {scope}",
                                  f"performance.ctr={ctr}; peer_stats.avg_ctr={avg_ctr}"))
            elif ctr > avg_ctr * 1.1:
                out.append(Anchor("ctr_lead", min(0.6, (ctr - avg_ctr) / avg_ctr),
                                  f"your profile click-through is {fmt_pct(ctr)}, ahead of the {fmt_pct(avg_ctr)} average for {scope}",
                                  f"performance.ctr={ctr}; peer_stats.avg_ctr={avg_ctr}"))

        calls, avg_calls = perf.get("calls"), peer.get("avg_calls_30d")
        if calls is not None and avg_calls and calls < avg_calls * 0.8:
            out.append(Anchor("calls_gap", min(1.0, (avg_calls - calls) / avg_calls),
                              f"{fmt_int(calls)} calls in the last {perf.get('window_days', 30)} days vs ~{fmt_int(avg_calls)} for {scope}",
                              f"performance.calls={calls}; peer_stats.avg_calls_30d={avg_calls}"))

        for metric in ("calls", "views"):
            d = delta.get(f"{metric}_pct")
            if d is None:
                continue
            if d <= -0.1:
                out.append(Anchor(f"{metric}_down", min(1.0, abs(d) * 1.5),
                                  f"{metric} are down {fmt_pct(d)} this week",
                                  f"performance.delta_7d.{metric}_pct={d}"))
            elif d >= 0.1:
                out.append(Anchor(f"{metric}_up", min(1.0, d * 1.5),
                                  f"{metric} are up {fmt_pct(d)} this week",
                                  f"performance.delta_7d.{metric}_pct={d}"))

        if self.active_offers:
            title = self.active_offers[0]
            out.append(Anchor("live_offer", 0.5, f"your '{title}' is live", f"offers[active].title={title}"))
        else:
            cat = self.catalog_offer()
            if cat:
                out.append(Anchor("no_offer", 0.6,
                                  f"there's no live offer on your listing right now — '{cat['title']}' is a proven format for {self.slug}",
                                  f"offers[active]=[]; offer_catalog.{cat.get('id')}={cat['title']}"))

        for theme in sorted(self.review_themes, key=lambda t: -(t.get("occurrences_30d") or 0)):
            occ = theme.get("occurrences_30d") or 0
            label = humanize(theme.get("theme", ""))
            quote = theme.get("common_quote")
            neg = theme.get("sentiment") == "neg"
            text = f"{occ} reviews in the last 30 days mention {label}" + (f" (\"{quote}\")" if quote else "")
            out.append(Anchor("review_neg" if neg else "review_pos", min(1.0, occ / 5) * (1.0 if neg else 0.7),
                              text, f"review_themes.{theme.get('theme')}.occurrences_30d={occ}"))

        for key, phrase in COHORT_PHRASES.items():
            n = self.aggregate.get(key)
            if n:
                out.append(Anchor("cohort", 0.7, phrase.format(n=fmt_int(n), aud=aud),
                                  f"customer_aggregate.{key}={n}"))
                break

        if self.identity.get("verified") is False:
            out.append(Anchor("unverified", 0.8, "your Google profile is still unverified",
                              "identity.verified=False"))

        sub = self.subscription
        if sub.get("status") == "expired" and sub.get("days_since_expiry"):
            out.append(Anchor("sub_expired", 0.9, f"your plan lapsed {sub['days_since_expiry']} days ago",
                              f"subscription.days_since_expiry={sub['days_since_expiry']}"))
        elif sub.get("days_remaining") is not None and sub.get("days_remaining") <= 15:
            out.append(Anchor("sub_ending", 0.85, f"your {sub.get('plan', '')} plan has {sub['days_remaining']} days left".replace("  ", " "),
                              f"subscription.days_remaining={sub['days_remaining']}"))
        return out

    def best_anchor(self, weights: dict[str, float], exclude: Iterable[str] = ()) -> tuple[Optional[Anchor], list[Anchor]]:
        """Pick the single supporting fact with the highest weight x magnitude for this trigger kind."""
        exclude = set(exclude)
        scored = []
        for idx, a in enumerate(self.merchant_anchors()):
            w = weights.get(a.type, 0.0)
            if w <= 0 or a.type in exclude:
                continue
            scored.append((w * a.magnitude, -idx, a))
        scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
        ranked = [s[2] for s in scored]
        return (ranked[0] if ranked else None), ranked[1:]
