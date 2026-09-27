"""Rule-based intent classification for inbound merchant/customer replies.

Order matters: opt-out and auto-reply are checked before anything that could read as
acceptance ("Thank you for contacting us" must never count as a "thank you" / yes).
"""
import re
from typing import Optional

OPT_OUT = [
    r"\bstop\b", r"\bunsubscribe\b", r"\bnot interested\b", r"\bno interest\b",
    r"\bdon'?t (message|text|contact|send|msg)\b", r"\bdo not (message|text|contact|send)\b",
    r"\bleave me alone\b", r"\bremove (me|my number)\b", r"\bblock(ed)?\b",
    r"\bband karo\b", r"\bmat bhej", r"\bmessage mat\b", r"\bnahi chahiye\b", r"\bnahin chahiye\b",
    r"\binterest nahi\b", r"\bpareshan mat\b",
]
HOSTILE = [
    r"\buseless\b", r"\bwaste of (my )?time\b", r"\bidiot", r"\bstupid\b", r"\bnonsense\b", r"\bbakwas\b",
    r"\bpagal\b", r"\bfraud\b", r"\bscam\b", r"\bspam\b", r"\bbothering\b", r"\birritat", r"\bharass",
    r"\bshut up\b", r"\bbloody\b", r"\bdamn\b", r"\bf+u+c+k", r"\bwtf\b", r"\bchutiya", r"\bbewakoof",
    r"\bget lost\b", r"\bpathetic\b", r"\brubbish\b",
]
AUTO_REPLY = [
    r"thank(s| you) for (contacting|reaching|your message|messaging)", r"will (get back|respond|reply) (to you )?(shortly|soon|asap)",
    r"our team will", r"we('| a)re (currently )?(closed|unavailable|away)", r"out of (the )?office",
    r"business hours", r"auto(mated)?[- ]?(reply|response|message)", r"i am an automated", r"automated assistant",
    r"this is an automated", r"aapki jaankari ke liye", r"hamari team tak", r"jald hi sampark",
    r"we have received your (message|query)", r"for (urgent|immediate) (queries|assistance),? (please )?call",
]
DEFER = [
    r"\blater\b", r"\bbusy\b", r"\bnot now\b", r"\bnot today\b", r"\bin a meeting\b", r"\bcall (me )?later\b",
    r"\bbaad mein\b", r"\bbaad me\b", r"\babhi nahi\b", r"\babhi busy\b", r"\bkal\b", r"\btomorrow\b",
    r"\bnext week\b", r"\bgive me (some )?time\b", r"\bafter (some )?time\b", r"\bthoda time\b", r"\bweekend\b",
]
DECLINE = [
    r"^\s*no\b", r"^\s*nope\b", r"\bno thanks?\b", r"\bnot needed\b", r"\bdon'?t need\b", r"\bnot required\b",
    r"\btoo (expensive|costly|much)\b", r"\bmehenga\b", r"\bmehnga\b", r"\bafford\b", r"\balready (have|doing|using)\b",
    r"^\s*nahi\b", r"^\s*nahin\b", r"\bzaroorat nahi\b", r"\bnot for me\b", r"\bnot worth\b", r"\bwon'?t work\b",
    r"\bno need\b", r"\bskip\b", r"\bpass\b",
]
ACCEPT = [
    r"\byes\b", r"\byeah\b", r"\byep\b", r"\bya\b", r"\bhaan\b", r"\bhan ji\b", r"\bhaanji\b", r"\bji haan\b",
    r"^\s*ok(ay)?\b", r"\bok(ay)? (go|do|let|send|please|sure|fine)", r"\bsure\b", r"\bgo ahead\b", r"\bproceed\b",
    r"\blet'?s do it\b", r"\blets do it\b", r"\blet'?s go\b", r"\bdo it\b", r"\bplease (send|do|go|share|draft|start|proceed|set)",
    r"\bsend (it|me|the|over)\b", r"\bsounds good\b", r"\bgreat idea\b", r"\bgood idea\b", r"\binterested\b",
    r"\bchalo\b", r"\bkar do\b", r"\bkardo\b", r"\bbhej do\b", r"\bbhejo\b", r"\btheek hai\b", r"\bthik hai\b",
    r"\bconfirm(ed)?\b", r"\bbook (it|me)\b", r"\bdone\b", r"\bagreed?\b", r"\bperfect\b",
    r"\bi want to (join|start|do|go)\b", r"\bjudna hai\b", r"\bjudrna hai\b", r"\bjoin karna\b", r"\bshuru karo\b",
    r"\bwhat'?s next\b", r"\bwhats next\b", r"\bnext step\b", r"👍", r"✅", r"🙏",
]
INFO = [
    r"\bdetails?\b", r"\btell me more\b", r"\bmore info", r"\bhow (does|do|will) (it|this|that) work",
    r"\bexplain\b", r"\bwhat is (this|it)\b", r"\bkya hai\b", r"\bsamjhao\b", r"\bbataiye\b", r"\bbatao\b",
    r"\bhow much\b", r"\bprice\b", r"\bcost\b", r"\bkitna\b", r"\bkitne\b", r"\bcharges?\b", r"\bfees?\b",
    r"\bwhich (one|study|paper)\b", r"\bsource\b",
]
OFF_TOPIC = [
    r"\bgst\b", r"\bincome tax\b", r"\bitr\b", r"\btax (filing|return)", r"\bloan\b", r"\binsurance\b",
    r"\bchartered accountant\b", r"\baccountant\b", r"\bca\b", r"\blawyer\b", r"\blegal (case|notice)\b",
    r"\bvisa\b", r"\bpassport\b", r"\baadhaar\b", r"\bpan card\b", r"\belectricity bill\b", r"\brent agreement\b",
    r"\bmy (son|daughter|wife|husband)'?s?\b", r"\bcricket score\b", r"\bstock (tip|market)\b", r"\bcrypto\b",
]
THANKS_ONLY = r"^\s*(thanks|thank you|thx|ty|shukriya|dhanyavaad|dhanyawad)( so much| a lot| ji)?[\s!.🙏]*$"
ACK_ONLY = r"^\s*(ok|okay|k|noted|got it|fine|hmm+|acha|accha)[\s!.👍]*$"
SLOT_PICK = r"^\s*(?:option\s*)?([1-9])\s*[.!]?\s*$"


def _any(patterns: list[str], text: str) -> bool:
    return any(re.search(p, text) for p in patterns)


def classify(message: str, repeat_count: int = 1, expects_answer: bool = False,
             options: Optional[list[str]] = None) -> str:
    text = (message or "").strip().lower()
    if not text:
        return "unclear"
    if _any(AUTO_REPLY, text) or (repeat_count >= 2 and len(text) > 25):
        return "auto_reply"
    if _any(OPT_OUT, text):
        return "opt_out"
    if _any(HOSTILE, text):
        return "hostile"
    if options:
        m = re.match(SLOT_PICK, text)
        if m and 1 <= int(m.group(1)) <= len(options):
            return "slot_pick"
    if re.search(THANKS_ONLY, text):
        return "thanks"
    if re.search(ACK_ONLY, text):
        return "ack"
    if _any(OFF_TOPIC, text):
        return "off_topic"
    defer = _any(DEFER, text)
    decline = _any(DECLINE, text)
    accept = _any(ACCEPT, text)
    if defer and not re.search(r"\b(yes|haan|ok|sure|go ahead)\b", text):
        return "defer"
    if decline and not accept:
        return "decline"
    if accept:
        return "accept"
    if _any(INFO, text):
        return "info"
    if expects_answer and len(text) >= 2:
        return "answer"
    return "unclear"


def defer_seconds(message: str) -> int:
    text = (message or "").lower()
    if re.search(r"next week|weekend", text):
        return 7 * 86400 if "next week" in text else 3 * 86400
    if re.search(r"\bkal\b|tomorrow", text):
        return 86400
    if re.search(r"\b(\d+)\s*(min|minute)", text):
        return int(re.search(r"\b(\d+)\s*(min|minute)", text).group(1)) * 60
    if re.search(r"\b(\d+)\s*(hr|hour|ghante)", text):
        return int(re.search(r"\b(\d+)\s*(hr|hour|ghante)", text).group(1)) * 3600
    return 4 * 3600
