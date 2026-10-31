# Data card: Synthetic IT ticket dataset

**Everything in this dataset is synthetic.** No real tickets, users or systems. Every row has `is_synthetic = true`. Regenerate identically with:

```
python generate_data.py --out data --seed 42
```

## How it was generated

1. **53 root-cause scenarios** (`scenarios.py`) across 7 categories and 24 subcategories. Each scenario has a fixed root cause, canonical fix steps, title/symptom templates, error codes, apps, priority mix and assignment group.
2. Each ticket is built from one scenario: random title + 1–2 symptom sentences + optional context, duration, impact (drives priority), "already tried", greeting/closing, and a signature containing fake PII (40% of tickets).
3. Noise: 20% lowercase titles, 12% lowercase descriptions, 10% with a typo.
4. Resolution notes are written from the scenario's fix steps (the key step is always kept, others 85% of the time), with varied agent phrasing and occasional red herrings ("rebooted first, no change").
5. Tickets from the **same scenario are related** (duplicate ground truth). **9 hard-negative scenario pairs** share keywords or error codes but have different causes (e.g. VPN error 809: cached credentials vs firewall rule).
6. **4 novel scenarios** exist only in the test set; they have no history or KB, so the system should escalate.

## Files

| File | Rows | Purpose |
| --- | --- | --- |
| `raw/historical_tickets.csv` | 2,000 | Search index + classifier training. Jan 2025 – Jun 2026 |
| `raw/kb_articles.csv` / `.json` | 51 | 47 approved KB articles + 4 drafts/superseded (`approved = false`, must not be retrieved) |
| `raw/test_tickets.csv` | 450 | New tickets, Jul – Oct 2026, **input fields only** |
| `labels/test_labels.csv` | 450 | Ground truth: category, subcategory, priority, scenario, expected behavior, reference fix |
| `labels/test_qrels.csv` | 16,522 | (test ticket, relevant historical ticket or KB) pairs for Recall@k / MRR / nDCG |
| `labels/test_hard_negatives.csv` | 5,268 | Lookalike tickets that should NOT rank high (hard-negative rate) |
| `labels/duplicate_pairs_to_verify.csv` | 70 | 50 related + 20 hard-negative pairs for a manual check; fill `reviewer_agrees` |
| `labels/scenarios.json` | 57 | Scenario definitions. **Evaluation only, never feed to models** |
| `taxonomy.yaml` | — | Category > subcategory tree, default escalation groups, priority rules and signal words |
| `users.csv` | 3 | Prototype roles: analyst1/analyst2 (L1), admin1 (sees Restricted) |

## Historical tickets at a glance

| Field | Distribution |
| --- | --- |
| Category | Access 384, Application 335, Network 314, Hardware 284, Configuration 253, Email and collaboration 249, Database 181 |
| Priority | P3 1,110 · P4 457 · P2 366 · P1 67 |
| Status | Closed 1,880 · Resolved 36 · Open 44 · In Progress 40 (no resolution) |
| Sensitivity | Internal 1,665 · Public 212 · Restricted 123 |
| Environment | Production 1,943 · UAT 38 · Dev 19 |
| Tickets per scenario | 16 – 69 |
| Avg description length | 28 words |
| Has KB reference | ~45% of resolved tickets whose scenario has a KB |
| Root cause filled | ~85% of resolved tickets |

## Test set composition

| `test_type` | Count | `expected_behavior` |
| --- | --- | --- |
| standard | 400 | `confident_answer_expected` |
| novel | 25 | `low_confidence_escalate` (4 scenarios with no history) |
| vague | 15 | `low_confidence_ask_clarifying_questions` |
| contradictory | 10 | `medium_or_low_flag_conflicting_evidence` (symptoms from a hard-negative pair) |

## Columns

**historical_tickets.csv:** `ticket_id, title, description, application, environment, category, subcategory, priority, assignment_group, status, created_at, resolved_at, resolution_notes, root_cause, kb_refs, sensitivity, scenario_id, is_synthetic`

`scenario_id` is for evaluation only. Drop it before training or indexing.

**test_labels.csv:** `ticket_id, test_type, scenario_id, category, subcategory, priority, expected_assignment_group, expected_behavior, reference_root_cause, reference_resolution, relevant_kb`

## Fake PII (for masking tests)

Emails `first.last@acmecorp.example` (reserved `.example` domain), phone numbers `+91 9XXXXXXXXX`, laptop hosts `LT-####`, servers `APP/DB/FS/BKP-SRV-##`, printers `PRN-HYD-3F-01`. 819 historical descriptions contain an email signature.

## Known limitations

- Template-based text is cleaner and more repetitive than real tickets; real data would score lower.
- English only; one ticket = one root cause (except the contradictory test set).
- Priority is driven by explicit impact phrases in ~65% of tickets; the rest carry no scope signal.
- Labels are generated, not human-verified. Use `duplicate_pairs_to_verify.csv` to spot-check.
