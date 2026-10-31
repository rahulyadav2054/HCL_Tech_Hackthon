"""Citation validator (Phase 6, section 9.4, FR-04).

Checks, all recorded in the audit log:
  1. schema    - done in resolve.generate (pydantic, one retry with the error, then fallback)
  2. every source_id is in the evidence pack            -> blocks hallucinated ticket/KB IDs
  3. every excerpt appears in that source text          -> normalized exact match, or
                                                           rapidfuzz partial_ratio >= 90
     failed excerpts are removed
  4. a path left with zero valid evidence is dropped
  5. citation pass rate = valid citations / total citations (feeds confidence signal V)
  6. steps containing risky verbs are flagged requires_approval
"""
from __future__ import annotations

import re

from rapidfuzz import fuzz

from ai import config

_SOURCE_ID = re.compile(r"^\s*\[?\s*((?:INC\d{5,})|(?:KB\d{3,}))\s*(?:§\s*([A-Za-z]+))?\s*\]?\s*$", re.I)
_RISKY = [re.compile(p, re.I) for p in config.RISKY_PATTERNS]


def _norm(text: str) -> str:
    text = text.lower().replace("…", " ").replace("...", " ")
    text = re.sub(r"[\"'`“”‘’]", "", text)
    return re.sub(r"\s+", " ", text).strip(" .;:,")


def parse_source_id(raw: str) -> tuple[str | None, str | None]:
    m = _SOURCE_ID.match(raw or "")
    if not m:
        return None, None
    return m.group(1).upper(), (m.group(2).capitalize() if m.group(2) else None)


def source_texts(pack: list[dict]) -> dict[str, dict[str | None, str]]:
    """source_id -> {section or None: text}. KB articles may contribute several sections."""
    out: dict[str, dict] = {}
    for item in pack:
        out.setdefault(item["source_id"].upper(), {})[item["section"]] = item["text"]
    return out


def excerpt_match(excerpt: str, source: str) -> int:
    e, s = _norm(excerpt), _norm(source)
    if not e:
        return 0
    if e in s:
        return 100
    return int(fuzz.partial_ratio(e, s))


def risky_reasons(texts: list[str]) -> list[str]:
    hits = []
    for t in texts:
        for r in _RISKY:
            m = r.search(t)
            if m:
                hits.append(m.group(0).lower())
    return sorted(set(hits))


def validate(output: dict | None, pack: list[dict]) -> dict:
    sources = source_texts(pack)
    checks, paths, dropped = [], [], []
    total = valid = 0
    hallucinated = set()
    for path in (output or {}).get("paths", []):
        kept = []
        for cit in path.get("evidence", []):
            total += 1
            sid, section = parse_source_id(cit.get("source_id", ""))
            check = {"path_rank": path.get("rank"), "source_id": cit.get("source_id"), "excerpt": cit.get("excerpt", "")}
            if sid is None or sid not in sources:
                check.update(status="unknown_source", similarity=0)
                hallucinated.add(cit.get("source_id"))
            else:
                texts = sources[sid]
                text = texts.get(section) if section in texts else " ".join(texts.values())
                sim = excerpt_match(cit.get("excerpt", ""), text)
                if sim < config.EXCERPT_FUZZY_MIN and section in texts:  # section mislabelled: try the whole KB
                    sim = max(sim, excerpt_match(cit.get("excerpt", ""), " ".join(texts.values())))
                ok = sim >= config.EXCERPT_FUZZY_MIN
                check.update(status="valid" if ok else "excerpt_not_found", similarity=sim)
                if len(cit.get("excerpt", "").split()) > config.EXCERPT_MAX_WORDS:
                    check["warning"] = f"excerpt longer than {config.EXCERPT_MAX_WORDS} words"
                if ok:
                    valid += 1
                    kept.append({**cit, "source_id": sid, "section": section, "match_similarity": sim})
            checks.append(check)
        if not kept:
            dropped.append({"rank": path.get("rank"), "title": path.get("title"), "reason": "no valid evidence"})
            continue
        risky = risky_reasons([path.get("title", "")] + list(path.get("steps", [])))
        paths.append({**path, "evidence": kept,
                      "requires_approval": bool(path.get("requires_approval")) or bool(risky),
                      "approval_reasons": risky})
    for i, p in enumerate(paths, start=1):
        p["rank"] = i
    return {
        "paths": paths,
        "checks": checks,
        "total_citations": total,
        "valid_citations": valid,
        "citation_pass_rate": round(valid / total, 4) if total else 0.0,
        "removed_citations": [c for c in checks if c["status"] != "valid"],
        "dropped_paths": dropped,
        "hallucinated_ids": sorted(h for h in hallucinated if h),
        "conflicts": (output or {}).get("conflicts", "none"),
        "missing_information": (output or {}).get("missing_information", []),
    }
