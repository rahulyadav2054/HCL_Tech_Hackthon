"""Shared helpers for the evaluation scripts (run as modules: python -m ai.eval.<script>).

Ground truth comes from the dataset's labels (data/labels/): test_labels.csv, test_qrels.csv,
test_hard_negatives.csv. Evaluations write audit rows to data/eval_audit.sqlite so the demo audit
log stays clean. Import this module BEFORE anything else from ai/ or backend/.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("APP_DB_PATH", str(ROOT / "data" / "eval_audit.sqlite"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import warnings  # noqa: E402

warnings.filterwarnings("ignore")

import pandas as pd  # noqa: E402

from ai import config, dataset  # noqa: E402

REPORTS = config.REPORTS_DIR
REPORTS.mkdir(parents=True, exist_ok=True)
ADMIN_ROLE = next((r for r in config.ROLES_WITH_AUDIT), "L2_Admin")


def test_tickets() -> pd.DataFrame:
    return dataset.test_tickets().set_index("ticket_id", drop=False)


def as_input(row) -> dict:
    """What an analyst submits: title, description, application, environment (no labels)."""
    return {"title": row["title"], "description": row["description"],
            "application": row["application"] or "unknown", "environment": row["environment"] or "unknown"}


def save_json(name: str, obj) -> Path:
    p = REPORTS / name
    p.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")
    return p


def generated_paths(result: dict) -> list[dict]:
    """Validated paths including ones suppressed from display (Low band), read from the audit row."""
    from backend import audit
    row = audit.get_analysis(result["analysis_id"])
    return ((row or {}).get("recommendation_json") or {}).get("paths") or []


def path1_correct(paths: list[dict], relevant: set[str]) -> bool | None:
    """Automatic usefulness proxy: path 1 cites a historical ticket or KB article that the dataset's
    qrels mark as relevant for this test ticket. None when no path was produced."""
    if not paths:
        return None
    return any(e["source_id"] in relevant for e in paths[0]["evidence"])
