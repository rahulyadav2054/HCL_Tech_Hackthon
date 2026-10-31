"""Loader for the it-ticket-dataset (data/): the only module that knows the dataset's file layout.

  data/raw/historical_tickets.csv   2,000 tickets: search index + classifier training
  data/raw/test_tickets.csv         450 new tickets, input fields only
  data/raw/kb_articles.csv          51 KB articles (47 approved; drafts/superseded are never retrieved)
  data/labels/*                     ground truth for evaluation only (never fed to models)
  data/taxonomy.yaml, data/users.csv

scenario_id is dropped from historical tickets unless a caller explicitly asks for evaluation labels.

Import a (new version of the) dataset zip:  python -m ai.dataset --import path/to/it-ticket-dataset.zip
Summary:                                    python -m ai.dataset --summary
"""
from __future__ import annotations

import argparse
import shutil
import zipfile
from functools import lru_cache
from pathlib import Path, PurePosixPath

import pandas as pd
import yaml

from ai import config

SEED = 42


# ----------------------------------------------------------------------------- taxonomy + policy
@lru_cache(maxsize=1)
def taxonomy() -> dict:
    return yaml.safe_load(config.TAXONOMY_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def policy() -> dict:
    return yaml.safe_load(config.POLICY_PATH.read_text(encoding="utf-8"))


def categories() -> dict:
    """{category: {"default_group": str, "subcategories": [...]}}"""
    return taxonomy()["categories"]


def allowed_pairs() -> list[tuple[str, str]]:
    return [(c, s) for c, v in categories().items() for s in v["subcategories"]]


def priority_signals() -> dict[str, list[str]]:
    return taxonomy().get("priority_signals", {})


# ----------------------------------------------------------------------------- tickets + KB
def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def historical(with_eval_labels: bool = False) -> pd.DataFrame:
    df = _read(config.HISTORICAL_TICKETS)
    if not with_eval_labels:
        df = df.drop(columns=["scenario_id"], errors="ignore")
    return df


def test_tickets() -> pd.DataFrame:
    return _read(config.TEST_TICKETS)


def kb_articles(approved_only: bool = False) -> pd.DataFrame:
    df = _read(config.KB_ARTICLES)
    df["approved"] = df["approved"].str.lower().isin(["true", "1", "yes"])
    if approved_only:
        df = df[df["approved"]]
    return df.drop(columns=["scenario_id"], errors="ignore")


def split_list(value: str) -> list[str]:
    return [v.strip() for v in str(value or "").replace(";", ",").split(",") if v.strip()]


# ----------------------------------------------------------------------------- labels (evaluation only)
def test_labels() -> pd.DataFrame:
    return _read(config.TEST_LABELS).set_index("ticket_id", drop=False)


def qrels() -> dict[str, set[str]]:
    """test ticket -> set of relevant historical ticket / KB ids."""
    q = _read(config.TEST_QRELS)
    q = q[q["relevance"].astype(int) > 0]
    out: dict[str, set[str]] = {}
    for qid, did in zip(q["query_id"], q["doc_id"]):
        out.setdefault(qid, set()).add(did)
    return out


def hard_negatives() -> dict[str, set[str]]:
    h = _read(config.TEST_HARD_NEGATIVES)
    out: dict[str, set[str]] = {}
    for qid, did in zip(h["query_id"], h["doc_id"]):
        out.setdefault(qid, set()).add(did)
    return out


def historical_scenarios() -> dict[str, str]:
    """ticket_id -> scenario_id for historical tickets (calibration / tuning labels only)."""
    df = _read(config.HISTORICAL_TICKETS)
    return dict(zip(df["ticket_id"], df["scenario_id"]))


def test_sets(seed: int = SEED) -> dict[str, list[str]]:
    """Evaluation sets from test_labels.test_type (workflow section 4.4 / 15)."""
    lab = test_labels()
    by_type = {t: sorted(g["ticket_id"]) for t, g in lab.groupby("test_type")}
    standard = by_type.get("standard", [])
    shuffled = pd.Series(standard).sample(frac=1.0, random_state=seed).tolist()
    return {
        "standard": standard,                       # 400: classification + retrieval
        "normal_50": sorted(shuffled[:50]),         # confidence: should get a confident answer
        "rating_25": sorted(shuffled[50:75]),       # resolution usefulness rating (disjoint)
        "novel_25": by_type.get("novel", []),
        "vague_15": by_type.get("vague", []),
        "contradictory_10": by_type.get("contradictory", []),
    }


# ----------------------------------------------------------------------------- import
def import_zip(zip_path: Path) -> list[str]:
    """Copy the dataset's data/ folder into ./data (and the generator scripts into data/generator/).

    Only files under <root>/data/ and the two generator scripts are taken; paths are checked so a
    malformed archive cannot write outside ./data. Generator scripts are copied, never executed.
    """
    written = []
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            parts = PurePosixPath(info.filename).parts
            if ".." in parts or PurePosixPath(info.filename).is_absolute():
                continue
            if "data" in parts:
                rel = Path(*parts[parts.index("data") + 1:])
                target = config.DATA_DIR / rel
            elif parts[-1] in ("generate_data.py", "scenarios.py"):
                target = config.DATA_DIR / "generator" / parts[-1]
            elif parts[-1] == "DATA_CARD.md":
                target = config.DATA_DIR / "DATA_CARD.md"
            else:
                continue
            if config.DATA_DIR.resolve() not in target.resolve().parents:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            written.append(str(target.relative_to(config.ROOT)))
    taxonomy.cache_clear()
    return written


def summary() -> dict:
    h, t, kb, lab = historical(), test_tickets(), kb_articles(), test_labels()
    return {"historical": len(h), "test": len(t), "kb_articles": len(kb), "kb_approved": int(kb["approved"].sum()),
            "test_types": lab["test_type"].value_counts().to_dict(),
            "historical_status": h["status"].value_counts().to_dict(),
            "historical_sensitivity": h["sensitivity"].value_counts().to_dict(),
            "categories": h["category"].value_counts().to_dict()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--import", dest="zip_path", type=Path, help="path to it-ticket-dataset.zip")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    if a.zip_path:
        files = import_zip(a.zip_path)
        print(f"imported {len(files)} files into data/:\n  " + "\n  ".join(files))
        print("next: python -m ai.build_index && python -m ai.classify --train")
    if a.summary or not a.zip_path:
        import json
        print(json.dumps(summary(), indent=2))
