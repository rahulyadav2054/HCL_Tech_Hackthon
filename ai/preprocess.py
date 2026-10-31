"""Preprocessing and enrichment (Phase 2).

The same preprocess(ticket) runs on historical tickets at index time and on every new
ticket at query time, so both sides are cleaned identically. It returns masked text for
models plus structured entities for filters, boosts and explanations.

Steps: validate -> normalize -> mask PII -> extract entities -> build search text
-> build resolution text (historical only).

Usage: python -m ai.preprocess      (writes data/processed/tickets.parquet for historical + test tickets)
"""
from __future__ import annotations

import html
import json
import re
import sys
import threading
import time
from collections import OrderedDict
from functools import lru_cache

import pandas as pd
from pydantic import BaseModel, field_validator

from ai import config, dataset

# ============================================================================ 1. validation


class TicketIn(BaseModel):
    """Schema for an incoming or historical ticket. Missing metadata defaults to 'unknown'."""
    title: str
    description: str
    application: str = "unknown"
    environment: str = "unknown"
    ticket_id: str | None = None
    category: str | None = None
    subcategory: str | None = None
    priority: str | None = None
    assignment_group: str = "unknown"
    status: str = "unknown"
    created_at: str | None = None
    resolved_at: str | None = None
    resolution_notes: str = ""
    root_cause: str = ""
    kb_refs: str = ""
    sensitivity: str = "Internal"
    restricted_fields: str = ""

    @field_validator("title", "description")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if v is None or not str(v).strip():
            raise ValueError("must not be empty")
        return str(v)

    @field_validator("application", "environment", "assignment_group", "status", mode="before")
    @classmethod
    def default_unknown(cls, v):
        if v is None or (isinstance(v, float) and pd.isna(v)) or not str(v).strip():
            return "unknown"
        return str(v).strip()

    @field_validator("resolution_notes", "root_cause", "kb_refs", "restricted_fields", "resolved_at",
                     "created_at", mode="before")
    @classmethod
    def nan_to_empty(cls, v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return ""
        return str(v)


# ============================================================================ 2. normalization
_TAG = re.compile(r"<[^>]+>")
_REPLY = re.compile(r"(?im)^\s*-{2,}\s*original message\s*-{2,}|^\s*from:\s.*\n(?:.*\n){0,3}?\s*sent:\s")
_SIGNATURE = re.compile(
    r"(?is)(?:^|\n)\s*(?:regards|best regards|kind regards|thanks and regards|warm regards|cheers|--|"
    r"sent from my \w+)\s*,?\s*\n.*$")


def normalize(text: str) -> str:
    """Strip HTML, reply chains and signatures; collapse whitespace. Casing is kept for display."""
    text = html.unescape(text or "")
    text = re.sub(r"(?i)<br\s*/?>|</p>|</div>", "\n", text)
    text = _TAG.sub(" ", text)
    m = _REPLY.search(text)
    if m:
        text = text[:m.start()]
    text = _SIGNATURE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


# ============================================================================ 3. PII masking
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
IPV4_RE = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")
IPV6_RE = re.compile(r"\b(?:[0-9a-fA-F]{1,4}:){3,7}[0-9a-fA-F]{1,4}\b")
# laptops LT-1831, servers DB-SRV-02 / BKP-SRV-01, printers PRN-HYD-3F-01
HOST_RE = re.compile(r"\b(?:(?:LT|DT|PC|VM)-(?:[A-Z]{2,5}-)?\d{3,6}|[A-Z]{2,4}-SRV-\d{1,3}|PRN-[A-Z]{2,5}-\w{1,4}-\d{1,3})\b",
                     re.I)
EMPID_RE = re.compile(r"\bEMP\d{5,8}\b", re.I)
USERNAME_RE = re.compile(r"(?i)\b(user\s*name|username|user id|login id|login|user)\s*[:=]\s*([A-Za-z][\w.-]{2,})")
NAME_INTRO_RE = re.compile(r"(?i)\b(this is|my name is)\s+([a-z]+(?:\s+[a-z]+)?)(?=\s+from\b|\s+here\b)")
PHONE_RE = re.compile(r"(?<![\w.])(?:\+\d{1,3}[\s-]?)?(?:\(\d{2,4}\)[\s-]?)?(?:\d{10}|\d{2,5}(?:[\s-]\d{2,5}){1,3})(?!\w|\.\d)")

PLACEHOLDERS = ("<EMAIL>", "<PHONE>", "<IP>", "<USER>", "<HOST>")


class PIIVault:
    """Reversible placeholder map, kept in memory only for the current analyst session.

    Never written to disk, the audit log or the LLM prompt. Bounded LRU so it cannot grow forever.
    """

    def __init__(self, max_items: int = 500, ttl_s: int = 3600):
        self._data: OrderedDict[str, tuple[float, list]] = OrderedDict()
        self._max, self._ttl = max_items, ttl_s
        self._lock = threading.Lock()

    def put(self, key: str, mapping: list[tuple[str, str]]):
        with self._lock:
            self._data[key] = (time.time(), mapping)
            self._data.move_to_end(key)
            while len(self._data) > self._max:
                self._data.popitem(last=False)

    def get(self, key: str) -> list:
        with self._lock:
            item = self._data.get(key)
            if not item or time.time() - item[0] > self._ttl:
                return []
            return item[1]


VAULT = PIIVault()


@lru_cache(maxsize=1)
def _presidio():
    """Load Presidio with the small spaCy model; return None if unavailable (regex still applies)."""
    if config.PII_ENGINE != "presidio":
        return None
    try:
        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider
        provider = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy", "models": [{"lang_code": "en", "model_name": config.SPACY_MODEL}]})
        import logging
        logging.getLogger("presidio-analyzer").setLevel(logging.ERROR)
        return AnalyzerEngine(nlp_engine=provider.create_engine(), supported_languages=["en"])
    except Exception as exc:  # missing spaCy model etc.
        print(f"[preprocess] Presidio unavailable ({exc}); using regex masking only", file=sys.stderr)
        return None


def _phone_ok(s: str) -> bool:
    return sum(ch.isdigit() for ch in s) >= 9


def mask_pii(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Replace PII with placeholders. Returns (masked_text, [(placeholder, original), ...])."""
    mapping: list[tuple[str, str]] = []

    def sub(pattern, placeholder, s, group=0, check=None):
        def repl(m):
            val = m.group(group)
            if check and not check(val):
                return m.group(0)
            mapping.append((placeholder, val))
            return m.group(0).replace(val, placeholder) if group else placeholder
        return pattern.sub(repl, s)

    text = sub(EMAIL_RE, "<EMAIL>", text)
    text = sub(IPV6_RE, "<IP>", text)
    text = sub(IPV4_RE, "<IP>", text)
    text = sub(HOST_RE, "<HOST>", text)
    text = sub(EMPID_RE, "<USER>", text)
    text = sub(USERNAME_RE, "<USER>", text, group=2)
    text = sub(NAME_INTRO_RE, "<USER>", text, group=2)
    text = sub(PHONE_RE, "<PHONE>", text, check=_phone_ok)

    engine = _presidio()
    if engine is not None and text:
        results = engine.analyze(text=text, language="en",
                                 entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "IP_ADDRESS"],
                                 allow_list=list(_allow_list()), score_threshold=0.6)
        label = {"PERSON": "<USER>", "EMAIL_ADDRESS": "<EMAIL>", "PHONE_NUMBER": "<PHONE>", "IP_ADDRESS": "<IP>"}
        for r in sorted(results, key=lambda r: r.start, reverse=True):
            span = text[r.start:r.end]
            if any(p in span for p in PLACEHOLDERS) or _looks_technical(span):
                continue
            # single capitalised words from the small NER model are mostly product names
            # ("Chrome", "Visio"); only accept full names such as "Emily Brown"
            if r.entity_type == "PERSON" and not re.fullmatch(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+", span):
                continue
            mapping.append((label[r.entity_type], span))
            text = text[:r.start] + label[r.entity_type] + text[r.end:]
    return text, mapping


_FIELD_SEP = "\n\n§§\n\n"


def _mask_fields(fields: list[str]):
    """Mask several fields with one NER pass (much faster); fall back to per-field on a split mismatch."""
    masked, mapping = mask_pii(_FIELD_SEP.join(fields))
    parts = masked.split(_FIELD_SEP)
    if len(parts) != len(fields):
        parts, mapping = [], []
        for f in fields:
            m, mp = mask_pii(f) if f else ("", [])
            parts.append(m)
            mapping += mp
    return (*[p.strip() for p in parts], mapping)


def _looks_technical(span: str) -> bool:
    """Guard against NER false positives on product names, codes and technical words."""
    low = span.lower()
    if "_" in span or (any(ch.isdigit() for ch in span) and not _phone_ok(span)):
        return True
    return low in _allow_list() or any(w in _allow_list() for w in low.split())


@lru_cache(maxsize=1)
def _allow_list() -> frozenset:
    words = set(APP_ALIASES)
    for app in known_apps():
        words.add(app.lower())
        words.update(w.lower() for w in re.findall(r"[A-Za-z]{3,}", app))
    words.update({"vpn", "wifi", "wi-fi", "outlook", "teams", "excel", "sap", "gui", "crm", "sso", "mfa", "dns",
                  "bitlocker", "intune", "ssis", "etl", "tempdb", "credential", "manager", "windows", "office",
                  "authenticator", "portal", "logon", "pad", "safe", "mode", "urgent", "hi", "hello", "dear",
                  "thanks", "regards", "please", "help", "error", "printer", "dock", "badge", "headset",
                  "adobe", "acrobat", "visio", "chrome", "edge", "firefox", "zoom", "pro", "jira", "office365"})
    return frozenset(words)


# ============================================================================ 4. entity extraction
_ERROR_PATTERNS = [
    (re.compile(r"\bORA-\d{5}\b", re.I), lambda m: m.group(0).upper()),
    (re.compile(r"\bAADSTS\d{5,6}\b", re.I), lambda m: m.group(0).upper()),
    (re.compile(r"\b0x[0-9a-f]{8}\b", re.I), lambda m: m.group(0).lower()),
    (re.compile(r"\bevent(?:\s*id)?\s*(\d{3,5})\b", re.I), lambda m: m.group(1)),  # KB lists bare "4740"
    (re.compile(r"\b[45]\d\d\s[245]\.\d\.\d{1,3}\b"), lambda m: m.group(0)),
    (re.compile(r"\bHTTP\s*(?:error\s*)?([45]\d\d)\b", re.I), lambda m: f"HTTP {m.group(1)}"),
    (re.compile(r"\b(?:shows|returns|showing)\s+(50[0-4])\b", re.I), lambda m: f"HTTP {m.group(1)}"),
    (re.compile(r"\b(?:error|exit code|code|err)\s*[:#]?\s*(\d{3,5})\b", re.I), lambda m: m.group(1)),
    (re.compile(r"\b[A-Z]{2,}(?:_[A-Z]{2,}){1,5}\b", re.I), lambda m: m.group(0).upper()),
]

SYMPTOMS = {
    "cannot connect": r"cannot connect|can't connect|not connecting|fails? to connect|unable to connect|connection refused|cannot reach",
    "cannot login": r"cannot log ?(?:in|on)|can't log ?in|login fail|logon fails|unable to (?:log|sign) ?in|login loop|login error|invalid user",
    "timeout": r"time ?out|timing out|timed out",
    "slow": r"\bslow|takes (?:minutes|long)|loading forever|runs forever|very long to load",
    "crash": r"crash|closes unexpectedly|stops responding|not responding|closes right after",
    "blue screen": r"blue screen|bsod",
    "access denied": r"access denied|permission denied|insufficient privileges|do not have access|need permission|blocked",
    "locked out": r"\block(?:ed|s|out)",
    "blank screen": r"blank|white window|black screen",
    "not syncing": r"not syncing|not receiving|stuck on updating",
    "offline": r"\boffline\b",
    "not detected": r"not detected|not recognised|not recognized|not found",
    "stuck": r"\bstuck\b|hanging|freez",
    "bounce": r"bounc|undeliverable",
    "quota": r"\bfull\b|over quota|size limit",
    "dropping": r"dropping|disconnects|drops",
    "prompt loop": r"keeps asking|prompt loop|prompts? every|repeatedly",
    "failed": r"\bfail(?:s|ed|ing|ure)?\b",
    "wrong result": r"wrong|incorrect|duplicate|missing|doubled",
    "not charging": r"not charging",
    "garbled": r"garbled|garbage|random symbols",
}
_SYMPTOM_RE = {k: re.compile(v, re.I) for k, v in SYMPTOMS.items()}

TRIGGERS = {
    "after password change": r"password (?:change|reset)|changed (?:my )?password|password expired|old password",
    "after firewall change": r"firewall",
    "after update": r"windows update|\bpatch|\bupdate\b|\bupdated\b|upgrade",
    "after migration": r"migrat|moved to the new server",
    "after release": r"\brelease\b|deploy",
    "after reboot": r"reboot|restart",
    "new device": r"new phone|changed my phone|replaced my phone|factory|new laptop|new headset|new badge",
    "new joiner": r"new joiner|joined the company|start date",
    "after add-in install": r"add-in",
}
_TRIGGER_RE = {k: re.compile(v, re.I) for k, v in TRIGGERS.items()}

APP_ALIASES = {
    "globalprotect": "GlobalProtect VPN", "anyconnect": "Cisco AnyConnect", "teams": "Microsoft Teams",
    "wifi": "Corporate Wi-Fi", "wi-fi": "Corporate Wi-Fi", "wireless": "Corporate Wi-Fi",
    "okta": "Okta Verify", "power bi": "Power BI Desktop", "docusign": "DocuSign", "service now": "ServiceNow",
    "sql server": "SQL Server", "postgres": "PostgreSQL", "mysql": "MySQL", "ad account": "Active Directory",
    "dock": "Docking station", "docking station": "Docking station", "exchange": "Exchange Online",
    "windows update": "Windows Update", "wan": "WAN link", "ethernet": "Wired LAN", "desk port": "Wired LAN",
}
# Application names that are ordinary words: only taken from the application field, never from free text
GENERIC_APPS = {"Laptop", "Printer", "Headset", "Office", "Windows", "Edge", "Java", "Python", "Chrome",
                "Intranet", "Microsoft 365"}


@lru_cache(maxsize=1)
def known_apps() -> tuple[str, ...]:
    """Application names built from the application column of the historical data."""
    path = config.PROCESSED_DIR / "apps.json"
    if path.exists():
        return tuple(json.loads(path.read_text(encoding="utf-8")))
    if config.HISTORICAL_TICKETS.exists():
        return tuple(sorted(a for a in dataset.historical()["application"].unique() if a))
    return tuple()


@lru_cache(maxsize=1)
def _app_patterns():
    pats = []
    for app in known_apps():
        if app in GENERIC_APPS:
            continue
        base = re.sub(r"\s*\(.*\)$", "", app)
        pats.append((re.compile(r"\b" + re.escape(base) + r"\b", re.I), app))
    for alias, app in APP_ALIASES.items():
        pats.append((re.compile(r"\b" + re.escape(alias) + r"\b", re.I), app))
    return pats


def _phrase_re(phrases):
    return re.compile(r"\b(?:" + "|".join(re.escape(p) for p in phrases) + r")\b", re.I)


@lru_cache(maxsize=1)
def _scope_patterns():
    """Scope phrases = ai/policy.yaml + the dataset's taxonomy priority_signals (P1 all, P2 team, P3 single)."""
    pol, sig = dataset.policy(), dataset.priority_signals()
    sp = pol["scope_phrases"]
    team = sp["team"] + [s for s in sig.get("P2", []) if s != "nobody can"]
    return {
        "team": re.compile(_phrase_re(team).pattern + r"|(?:entire|whole) (?:[a-z]+ )?(?:team|office|floor|department)",
                           re.I),
        "all_users": re.compile(_phrase_re(sp["all_users"] + [s for s in sig.get("P1", [])
                                                              if s not in pol["security_keywords"]]).pattern
                                + r"|(?:whole|entire) (?:[a-z]+ )?(?:site|branch|company)", re.I),
        "single_user": _phrase_re(sp["single_user"] + sig.get("P3", [])),
        "security": _phrase_re(pol["security_keywords"]),
        "production": re.compile(r"\bproduction\b|\bprod\b|\boutage\b", re.I),
        "signals": {p: _phrase_re(v) for p, v in sig.items() if v},
    }


def extract_entities(text: str) -> dict:
    codes: list[str] = []
    for pattern, fmt in _ERROR_PATTERNS:
        for m in pattern.finditer(text):
            val = fmt(m)
            if val not in codes and not val.startswith("<"):
                codes.append(val)
    # "error 809" can also be matched as a bare number inside other codes; keep distinct values only
    apps = []
    for pattern, app in _app_patterns():
        if pattern.search(text) and app not in apps:
            apps.append(app)
    symptoms = [k for k, r in _SYMPTOM_RE.items() if r.search(text)]
    triggers = [k for k, r in _TRIGGER_RE.items() if r.search(text)]
    sp = _scope_patterns()
    if sp["team"].search(text):
        scope = "team"
    elif sp["all_users"].search(text):
        scope = "all_users"
    elif sp["single_user"].search(text) or re.search(r"\b(?:i|my|me)\b", text, re.I):
        scope = "single_user"
    else:
        scope = "unknown"
    signals = sorted({m.group(0).lower() for r in sp["signals"].values() for m in r.finditer(text)})
    return {"error_codes": codes, "apps": apps, "symptoms": symptoms, "triggers": triggers, "scope": scope,
            "production": bool(sp["production"].search(text)), "security": bool(sp["security"].search(text)),
            "signals": signals}


# ============================================================================ 5-6. text builders
def build_search_text(title: str, description: str, entities: dict) -> str:
    ent = " ".join(entities["error_codes"] + entities["apps"] + entities["symptoms"] + entities["triggers"])
    return re.sub(r"\s+", " ", f"{title} {title} {description} {ent}".lower()).strip()


def build_resolution_text(root_cause: str, resolution_notes: str) -> str:
    if not (root_cause or resolution_notes):
        return ""
    rc = root_cause.strip().rstrip(".")
    return f"Root cause: {rc}. Fix: {resolution_notes.strip()}" if rc else f"Fix: {resolution_notes.strip()}"


def preprocess(ticket: dict | TicketIn, vault_key: str | None = None) -> dict:
    """Clean, mask and enrich one ticket. Identical for historical and new tickets."""
    t = ticket if isinstance(ticket, TicketIn) else TicketIn(**ticket)
    title_n, desc_n = normalize(t.title), normalize(t.description)
    # entities are extracted from normalized text (codes, apps, symptoms are not PII)
    entities = extract_entities(f"{title_n}. {desc_n}")
    title_m, desc_m, rc_m, notes_m, mapping = _mask_fields(
        [title_n, desc_n, normalize(t.root_cause), normalize(t.resolution_notes)])
    if vault_key:
        VAULT.put(vault_key, mapping)
    app = t.application
    app_inferred = app if app != "unknown" else (entities["apps"][0] if entities["apps"] else "unknown")
    return {
        "ticket_id": t.ticket_id,
        "title": title_m,
        "description": desc_m,
        "search_text": build_search_text(title_m, desc_m, entities),
        "entities": entities,
        "root_cause": rc_m,
        "resolution_notes": notes_m,
        "resolution_text": build_resolution_text(rc_m, notes_m),
        "pii_masked": len(mapping),
        "metadata": {
            "application": app, "application_inferred": app_inferred, "environment": t.environment,
            "status": t.status, "assignment_group": t.assignment_group,
            "created_at": (t.created_at or "")[:10], "resolved_at": (t.resolved_at or "")[:10],
            "sensitivity": t.sensitivity, "kb_refs": t.kb_refs, "restricted_fields": t.restricted_fields,
            "category": t.category, "subcategory": t.subcategory, "priority": t.priority,
        },
    }


def process_all() -> pd.DataFrame:
    """Preprocess historical + test tickets and write data/processed/tickets.parquet.

    scenario_id is never read here (dataset.historical() drops it). Test tickets carry input fields only.
    """
    hist, test = dataset.historical(), dataset.test_tickets()
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (config.PROCESSED_DIR / "apps.json").write_text(
        json.dumps(sorted(a for a in hist["application"].unique() if a), indent=1), encoding="utf-8")
    known_apps.cache_clear()
    _app_patterns.cache_clear()
    _allow_list.cache_clear()
    rows = []
    t0 = time.time()
    records = [("historical", r) for r in hist.to_dict("records")] + [("test", r) for r in test.to_dict("records")]
    fields = set(TicketIn.model_fields)
    for i, (source, r) in enumerate(records, 1):
        rec = preprocess({k: v for k, v in r.items() if k in fields})
        md = rec.pop("metadata")
        rec.update(md)
        rec["source"] = source
        rec["entities"] = json.dumps(rec["entities"])
        rows.append(rec)
        if i % 500 == 0:
            print(f"  preprocessed {i}/{len(records)} ({time.time() - t0:.0f}s)")
    df = pd.DataFrame(rows)
    df.to_parquet(config.PROCESSED_TICKETS, index=False)
    print(f"wrote {config.PROCESSED_TICKETS} ({len(df)} tickets: {len(hist)} historical + {len(test)} test, "
          f"{df['pii_masked'].sum()} PII items masked)")
    return df


def load_processed(source: str | None = None) -> pd.DataFrame:
    """Processed tickets; source = 'historical', 'test' or None for both."""
    df = pd.read_parquet(config.PROCESSED_TICKETS)
    df["entities"] = df["entities"].apply(json.loads)
    if source:
        df = df[df["source"] == source].reset_index(drop=True)
    return df


if __name__ == "__main__":
    process_all()
