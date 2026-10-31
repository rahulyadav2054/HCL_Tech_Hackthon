"""Audit log, analyst feedback and users (backend persistence, FR-06 / FR-08). SQLite via SQLAlchemy.

Every analysis writes one row to `analyses` BEFORE the result is returned; every analyst click
writes a row to `feedback` linked to it. Only masked text is ever stored (never raw PII).

Feedback is used:
  1. immediately: a match marked incorrect is hidden on re-analysis of the same ticket
     (same masked input hash)
  2. as a small ranking boost: net-positive matches get up to +5%, net-negative up to -5%
  3. for retraining: corrected labels are exported (manual --train, so changes are reviewable)
  4. for reporting: usefulness rate per confidence band
"""
from __future__ import annotations

import uuid
from collections import defaultdict
from datetime import datetime, timedelta

from sqlalchemy import JSON, Column, DateTime, String, Text, create_engine, select
from sqlalchemy.orm import declarative_base, sessionmaker

from ai import config

Base = declarative_base()

TARGET_TYPES = ("match", "resolution_path", "classification")
VERDICTS = ("useful", "incorrect", "unresolved")
DECISIONS = ("Accepted", "Edited", "Rejected", "Escalated")


class User(Base):
    __tablename__ = "users"
    user_id = Column(String, primary_key=True)
    role = Column(String, nullable=False)
    password_hash = Column(String)  # pbkdf2_sha256 (backend/auth.py); never the password itself


class Analysis(Base):
    __tablename__ = "analyses"
    analysis_id = Column(String, primary_key=True)
    created_at = Column(DateTime, nullable=False, index=True)
    user_id = Column(String, nullable=False)
    user_role = Column(String, nullable=False)
    input_hash = Column(String, index=True)
    input_masked = Column(JSON)
    classification_json = Column(JSON)
    retrieved_json = Column(JSON)
    evidence_ids = Column(JSON)
    recommendation_json = Column(JSON)
    validation_json = Column(JSON)
    confidence_json = Column(JSON)
    fallback_json = Column(JSON)
    band = Column(String, index=True)
    versions = Column(JSON)
    latency_ms = Column(JSON)
    final_decision = Column(String)
    decision_comment = Column(Text)
    decided_by = Column(String)
    decided_at = Column(DateTime)


class Feedback(Base):
    __tablename__ = "feedback"
    feedback_id = Column(String, primary_key=True)
    analysis_id = Column(String, nullable=False, index=True)
    target_type = Column(String, nullable=False)
    target_id = Column(String, nullable=False)
    verdict = Column(String, nullable=False)
    corrected_label = Column(String)
    comment = Column(Text)
    user_id = Column(String, nullable=False)
    created_at = Column(DateTime, nullable=False)


_engine = None
_Session = None


def engine():
    global _engine, _Session
    if _engine is None:
        config.APP_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _engine = create_engine(f"sqlite:///{config.APP_DB_PATH}", connect_args={"check_same_thread": False})
        _Session = sessionmaker(bind=_engine, expire_on_commit=False)
    return _engine


def session():
    engine()
    return _Session()


def init_db() -> None:
    Base.metadata.create_all(engine())
    with engine().begin() as conn:  # databases created before JWT auth lack the password column
        cols = {row[1] for row in conn.exec_driver_sql("PRAGMA table_info(users)")}
        if "password_hash" not in cols:
            conn.exec_driver_sql("ALTER TABLE users ADD COLUMN password_hash TEXT")
    from backend.auth import hash_password
    with session() as s:
        for uid, role in config.DEFAULT_USERS:  # data/users.csv is the source of truth for roles
            u = s.get(User, uid)
            if u is None:
                u = User(user_id=uid, role=role)
                s.add(u)
            else:
                u.role = role
            if not u.password_hash and config.DEMO_PASSWORD:  # demo seed; change with backend.auth --set-password
                u.password_hash = hash_password(config.DEMO_PASSWORD)
        s.commit()


def get_password_hash(user_id: str) -> str | None:
    with session() as s:
        u = s.get(User, user_id)
        return u.password_hash if u else None


def set_password_hash(user_id: str, password_hash: str) -> None:
    with session() as s:
        u = s.get(User, user_id)
        if u is None:
            raise KeyError(user_id)
        u.password_hash = password_hash
        s.commit()


def get_role(user_id: str) -> str | None:
    with session() as s:
        u = s.get(User, user_id)
        return u.role if u else None


def list_users() -> list[dict]:
    with session() as s:
        return [{"user_id": u.user_id, "role": u.role} for u in s.scalars(select(User))]


# ----------------------------------------------------------------------------- writes
def write_analysis(rec: dict) -> None:
    with session() as s:
        s.add(Analysis(**rec))
        s.commit()


def set_decision(analysis_id: str, decision: str, user_id: str, role: str, comment: str = "") -> dict:
    decision = decision.capitalize()
    if decision not in DECISIONS:
        raise ValueError(f"decision must be one of {DECISIONS}")
    with session() as s:
        a = s.get(Analysis, analysis_id)
        if a is None:
            raise KeyError(analysis_id)
        if a.user_id != user_id and role not in config.ROLES_WITH_AUDIT:
            raise PermissionError("only the analyst who ran the analysis (or an admin) can decide")
        a.final_decision, a.decision_comment = decision, comment
        a.decided_by, a.decided_at = user_id, datetime.now()
        s.commit()
        return {"analysis_id": analysis_id, "final_decision": decision, "decided_by": user_id}


def add_feedback(analysis_id: str, target_type: str, target_id: str, verdict: str, user_id: str, role: str,
                 corrected_label: str | None = None, comment: str | None = None) -> dict:
    verdict = verdict.lower()
    if target_type not in TARGET_TYPES:
        raise ValueError(f"target_type must be one of {TARGET_TYPES}")
    if verdict not in VERDICTS:
        raise ValueError(f"verdict must be one of {VERDICTS}")
    with session() as s:
        a = s.get(Analysis, analysis_id)
        if a is None:
            raise KeyError(analysis_id)
        if a.user_id != user_id and role not in config.ROLES_WITH_AUDIT:
            raise PermissionError("feedback is limited to the analyst who ran the analysis (or an admin)")
        fb = Feedback(feedback_id=str(uuid.uuid4()), analysis_id=analysis_id, target_type=target_type,
                      target_id=str(target_id), verdict=verdict, corrected_label=corrected_label,
                      comment=comment, user_id=user_id, created_at=datetime.now())
        s.add(fb)
        s.commit()
        return {"feedback_id": fb.feedback_id, "analysis_id": analysis_id, "verdict": verdict}


# ----------------------------------------------------------------------------- feedback use
def hidden_matches(input_hash: str) -> set[str]:
    """Matches marked incorrect in earlier analyses of the same (masked) ticket."""
    with session() as s:
        q = (select(Feedback.target_id).join(Analysis, Analysis.analysis_id == Feedback.analysis_id)
             .where(Analysis.input_hash == input_hash, Feedback.target_type == "match",
                    Feedback.verdict == "incorrect"))
        return set(s.scalars(q))


def feedback_adjustments() -> dict[str, float]:
    """doc_id -> boost in [-5%, +5%] from net (useful - incorrect) match votes across analyses."""
    net: dict[str, int] = defaultdict(int)
    with session() as s:
        for target, verdict in s.execute(select(Feedback.target_id, Feedback.verdict)
                                         .where(Feedback.target_type == "match")):
            net[target] += 1 if verdict == "useful" else -1 if verdict == "incorrect" else 0
    cap = config.FEEDBACK_BOOST_MAX
    return {k: max(-cap, min(cap, v * config.FEEDBACK_BOOST_PER_VOTE)) for k, v in net.items() if v}


# ----------------------------------------------------------------------------- reads
_A_COLS = [c.name for c in Analysis.__table__.columns]
_F_COLS = [c.name for c in Feedback.__table__.columns]


def _row(obj, cols):
    out = {}
    for c in cols:
        v = getattr(obj, c)
        out[c] = v.isoformat(timespec="seconds") if isinstance(v, datetime) else v
    return out


def get_analysis(analysis_id: str) -> dict | None:
    with session() as s:
        a = s.get(Analysis, analysis_id)
        return _row(a, _A_COLS) if a else None


def list_analyses(date_from: str | None = None, date_to: str | None = None, user_id: str | None = None) -> list[dict]:
    with session() as s:
        q = select(Analysis).order_by(Analysis.created_at.desc())
        if date_from:
            q = q.where(Analysis.created_at >= datetime.fromisoformat(date_from))
        if date_to:
            q = q.where(Analysis.created_at < datetime.fromisoformat(date_to) + timedelta(days=1))
        if user_id:
            q = q.where(Analysis.user_id == user_id)
        return [_row(a, _A_COLS) for a in s.scalars(q)]


def list_feedback(analysis_ids: list[str] | None = None) -> list[dict]:
    with session() as s:
        q = select(Feedback).order_by(Feedback.created_at)
        if analysis_ids is not None:
            q = q.where(Feedback.analysis_id.in_(analysis_ids))
        return [_row(f, _F_COLS) for f in s.scalars(q)]


def metrics() -> dict:
    """Usefulness rate by band, most-rejected matches, classification corrections."""
    analyses = {a["analysis_id"]: a for a in list_analyses()}
    fbs = list_feedback()
    by_band: dict[str, dict] = defaultdict(lambda: {"useful": 0, "incorrect": 0, "unresolved": 0})
    rejected: dict[str, int] = defaultdict(int)
    corrections = []
    for f in fbs:
        band = (analyses.get(f["analysis_id"]) or {}).get("band", "unknown")
        if f["target_type"] in ("resolution_path", "match"):
            by_band[band][f["verdict"]] += 1
        if f["target_type"] == "match" and f["verdict"] == "incorrect":
            rejected[f["target_id"]] += 1
        if f["target_type"] == "classification" and f["corrected_label"]:
            corrections.append({"analysis_id": f["analysis_id"], "corrected_label": f["corrected_label"]})
    usefulness = {b: {**v, "usefulness_rate": round(v["useful"] / max(1, sum(v.values())), 3)}
                  for b, v in by_band.items()}
    decisions = defaultdict(int)
    bands = defaultdict(int)
    for a in analyses.values():
        bands[a["band"]] += 1
        if a["final_decision"]:
            decisions[a["final_decision"]] += 1
    return {"analyses": len(analyses), "feedback": len(fbs), "bands": dict(bands), "decisions": dict(decisions),
            "usefulness_by_band": usefulness,
            "most_rejected_matches": sorted(rejected.items(), key=lambda kv: -kv[1])[:10],
            "classification_corrections": corrections[-20:]}


def corrected_labels() -> list[dict]:
    """Corrected classification labels for (manual) retraining."""
    return [f for f in list_feedback() if f["target_type"] == "classification" and f["corrected_label"]]
