"""Classification evaluation (section 15, FR-01).

Test data: data/raw/test_tickets.csv with labels from data/labels/test_labels.csv. Headline metrics
on the 400 `standard` tickets, plus a breakdown by test_type (novel / vague / contradictory).
Metrics: accuracy, macro-F1, per-class precision/recall, confusion matrix, P1/P2 recall separately.
Experiments: TF-IDF vs embeddings vs combined features; with vs without LLM assist.

Usage: python -m ai.eval.eval_classification [--llm]
"""
from __future__ import annotations

import argparse

from ai.eval import common
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score

from ai import dataset, llm
from ai.classify import VARIANTS, Classifier, FeatureBuilder, load_training, make_model
from ai.embeddings import embed_docs, embed_queries
from ai.preprocess import load_processed
from ai.retrieve import Retriever


def predict_all(clf: Classifier, df: pd.DataFrame, emb: np.ndarray, use_llm: bool, retriever=None):
    preds = []
    for i, (_, r) in enumerate(df.iterrows()):
        rec = {"title": r["title"], "description": r["description"], "search_text": r["search_text"],
               "entities": r["entities"], "metadata": {"application": r["application"],
                                                       "environment": r["environment"]}}
        nearest = None
        if use_llm and retriever is not None:
            mask = retriever.filter_mask(common.ADMIN_ROLE, None, "ticket")
            cands = retriever.recall(r["search_text"], embed_queries([r["search_text"]])[0], mask)
            nearest = retriever.nearest_labelled(cands)
        preds.append(clf.predict(rec, emb[i], nearest=nearest, use_llm=use_llm))
    return preds


def metrics_block(y_true, y_pred) -> dict:
    return {"accuracy": round(accuracy_score(y_true, y_pred), 4),
            "macro_f1": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="also evaluate with the LLM second opinion (needs API key)")
    args = ap.parse_args()

    labels = dataset.test_labels()
    proc = load_processed("test").set_index("ticket_id", drop=False)
    test = proc.loc[labels.index].copy()
    for c in ("category", "subcategory", "priority", "test_type"):
        test[c] = labels.loc[test.index, c].values
    emb = embed_docs(test["search_text"].tolist())
    clf = Classifier()

    preds = predict_all(clf, test, emb, use_llm=False)
    test["pred_cat"] = [p["category"]["label"] for p in preds]
    test["pred_sub"] = [p["subcategory"]["label"] for p in preds]
    test["pred_pri"] = [p["priority"]["label"] for p in preds]
    test["pred_pri_model"] = [p["priority"]["model_label"] for p in preds]
    test["cat_conf"] = [p["category"]["confidence"] for p in preds]
    test["gate_pass"] = [p["source"] == "model" and not p["uncertain"] for p in preds]
    test["has_terms"] = [bool(p["top_terms"]) for p in preds]
    std = test[test["test_type"] == "standard"]

    res = {"n_test": len(test), "n_standard": len(std), "model_variant": clf.variant,
           "category": metrics_block(std["category"], std["pred_cat"]),
           "subcategory": metrics_block(std["subcategory"], std["pred_sub"]),
           "subcategory_pair_accuracy": round(float(((std["category"] == std["pred_cat"]) &
                                                     (std["subcategory"] == std["pred_sub"])).mean()), 4),
           "priority": metrics_block(std["priority"], std["pred_pri"]),
           "priority_model_only": metrics_block(std["priority"], std["pred_pri_model"]),
           "priority_recall": {p: round(recall_score(std["priority"], std["pred_pri"], labels=[p], average="macro",
                                                     zero_division=0), 4) for p in ("P1", "P2", "P3", "P4")},
           "priority_recall_model_only": {p: round(recall_score(std["priority"], std["pred_pri_model"], labels=[p],
                                                                average="macro", zero_division=0), 4)
                                          for p in ("P1", "P2")},
           "priority_support": std["priority"].value_counts().to_dict(),
           "gate": {"share_model_only": round(float(std["gate_pass"].mean()), 4),
                    "accuracy_model_only": round(float((std["category"] == std["pred_cat"])[std["gate_pass"]].mean()), 4),
                    "accuracy_deferred": round(float((std["category"] == std["pred_cat"])[~std["gate_pass"]].mean()), 4)
                    if (~std["gate_pass"]).any() else None},
           "explained_share": round(float(test["has_terms"].mean()), 4)}
    res["per_class_category"] = classification_report(std["category"], std["pred_cat"], output_dict=True, zero_division=0)
    res["per_class_priority"] = classification_report(std["priority"], std["pred_pri"], output_dict=True, zero_division=0)
    res["by_test_type"] = {k: {**metrics_block(g["category"], g["pred_cat"]), "n": len(g),
                               "low_confidence_share": round(float((~g["gate_pass"]).mean()), 4)}
                           for k, g in test.groupby("test_type")}

    # feature ablation: retrain each variant on all historical tickets, evaluate on standard test tickets
    train_df, train_emb = load_training()
    std_emb = emb[[test.index.get_loc(i) for i in std.index]]
    ablation = {}
    for v in VARIANTS:
        fb = FeatureBuilder(v).fit(train_df, train_emb)
        m = make_model().fit(fb.transform(train_df, train_emb), train_df["category"])
        ablation[v] = metrics_block(std["category"], m.predict(fb.transform(std, std_emb)))
        print(f"  ablation {v:9s} category macro-F1 {ablation[v]['macro_f1']:.4f}")
    res["feature_ablation_category"] = ablation

    if args.llm and llm.available():
        preds_llm = predict_all(clf, std, std_emb, use_llm=True, retriever=Retriever())
        cat_llm = [p["category"]["label"] for p in preds_llm]
        res["llm_assist"] = {"without": res["category"], "with": metrics_block(std["category"], cat_llm),
                             "calls": sum(1 for p in preds_llm if p["llm_assist"]),
                             "agreed": sum(1 for p in preds_llm if p["source"] == "llm_assist" and not p["uncertain"])}
    else:
        res["llm_assist"] = {"status": "not run (pass --llm with an API key configured)",
                             "close_calls_that_would_be_sent": int((~std["gate_pass"]).sum())}

    cats = sorted(std["category"].unique())
    cm = confusion_matrix(std["category"], std["pred_cat"], labels=cats)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(cats)), cats, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(cats)), cats, fontsize=8)
    for i in range(len(cats)):
        for j in range(len(cats)):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("labelled")
    ax.set_title(f"Category confusion matrix (standard test tickets, n={len(std)})")
    fig.tight_layout()
    fig.savefig(common.REPORTS / "classification_confusion.png", dpi=130)
    plt.close(fig)
    res["confusion_matrix"] = {"labels": cats, "matrix": cm.tolist()}

    wrong = std[(std["category"] != std["pred_cat"]) | (std["priority"] != std["pred_pri"])]
    wrong = wrong.assign(kind=np.where(wrong["category"] != wrong["pred_cat"], "category", "priority"))
    res["errors"] = [{"ticket_id": t, "title": r["title"], "kind": r["kind"], "label": r["category"] if r["kind"] == "category"
                      else r["priority"], "pred": r["pred_cat"] if r["kind"] == "category" else r["pred_pri"],
                      "confidence": r["cat_conf"], "scope": r["entities"].get("scope"),
                      "signals": ", ".join(r["entities"].get("signals", []))}
                     for t, r in wrong.sort_values(["kind", "cat_conf"], ascending=[True, False]).head(10).iterrows()]
    res["n_category_errors"] = int((std["category"] != std["pred_cat"]).sum())
    res["n_priority_errors"] = int((std["priority"] != std["pred_pri"]).sum())
    common.save_json("classification_results.json", res)
    print(f"category  acc {res['category']['accuracy']:.3f} macro-F1 {res['category']['macro_f1']:.3f}")
    print(f"subcat    acc {res['subcategory']['accuracy']:.3f} macro-F1 {res['subcategory']['macro_f1']:.3f}")
    print(f"priority  acc {res['priority']['accuracy']:.3f} macro-F1 {res['priority']['macro_f1']:.3f} "
          f"P1 recall {res['priority_recall']['P1']:.3f} P2 recall {res['priority_recall']['P2']:.3f}")
    return res


if __name__ == "__main__":
    main()
