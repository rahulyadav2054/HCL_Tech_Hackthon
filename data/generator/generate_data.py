"""Generate the synthetic IT ticket dataset.

Usage:  python generate_data.py --out data --seed 42

Outputs (all synthetic, is_synthetic = true):
  raw/historical_tickets.csv     ~2,000 tickets (Jan 2025 - Jun 2026) -> index + classifier training
  raw/kb_articles.json / .csv    51 KB articles (47 approved, 4 drafts/superseded)
  raw/test_tickets.csv           450 new tickets (Jul - Oct 2026), input fields only
  labels/test_labels.csv         ground truth for every test ticket
  labels/test_qrels.csv          relevant historical tickets + KB per test ticket (retrieval eval)
  labels/test_hard_negatives.csv lookalike tickets that should NOT rank high
  labels/duplicate_pairs_to_verify.csv  70 pairs for a manual sanity check
  labels/scenarios.json          scenario definitions (evaluation only)
  taxonomy.yaml, users.csv
"""
import argparse
import csv
import json
import os
import random
import re
from datetime import datetime, timedelta

from scenarios import SCENARIOS, NOVEL_SCENARIOS, HARD_NEGATIVE_PAIRS, DRAFT_KBS

N_HIST = 2000
N_TEST_STD = 400
N_NOVEL = 25
N_VAGUE = 15
N_CONTRA = 10

HIST_START, HIST_END = datetime(2025, 1, 1), datetime(2026, 6, 30)
TEST_START, TEST_END = datetime(2026, 7, 1), datetime(2026, 10, 25)
DATA_CUT = datetime(2026, 6, 30)

FIRST = ["Rahul", "Priya", "Anil", "Sneha", "Vikram", "Divya", "Arjun", "Kavya", "Rohit", "Meera",
         "Sanjay", "Pooja", "Karthik", "Ananya", "Suresh", "Neha", "John", "Emma", "David", "Sara",
         "Michael", "Laura", "Imran", "Fatima", "Wei", "Lin", "Carlos", "Maria", "Tom", "Aisha"]
LAST = ["Kumar", "Sharma", "Reddy", "Iyer", "Patel", "Nair", "Rao", "Gupta", "Singh", "Menon",
        "Smith", "Brown", "Wilson", "Khan", "Chen", "Garcia", "Taylor", "Das", "Joshi", "Pillai"]
DOMAIN = "acmecorp.example"
SITES = ["Hyderabad", "Pune", "Chennai", "Bengaluru", "Noida", "London", "Dallas"]
FLOORS = ["the 2nd floor", "the 3rd floor", "floor 4", "the 5th floor", "the ground floor"]
MAILBOXES = [f"{m}@{DOMAIN}" for m in ["invoices", "support", "hr-team", "procurement", "payroll"]]

GREETINGS = ["Hi team,", "Hello,", "Hi,", "Dear IT,", "Hi support,", "Good morning,"]
DURATION = ["This has been happening since this morning.", "Started yesterday.",
            "It's been like this for 2 days.", "Since Monday.", "Started after lunch today."]
TRIED = ["I already restarted my laptop.", "I tried logging out and in again.",
         "Rebooting didn't help.", "I cleared my browser cache already."]
CLOSINGS = ["Please help.", "Please check urgently.", "Thanks.", "Kindly assist.",
            "Can someone look into this?", "Thanks in advance."]
IMPACT = {
    "P1": ["Production is down for all users.", "This is a critical outage, customers are affected.",
           "Whole site cannot work.", "Possible security incident, please treat as urgent."],
    "P2": ["My entire team ({n} people) is affected.", "We cannot process orders until this is fixed.",
           "This is blocking the finance month-end close.", "Several colleagues have the same problem."],
    "P3": ["Only my machine seems affected.", "I can work around it for now but it slows me down.",
           "It happens a few times a day.", "Just me, others are fine."],
    "P4": ["Not urgent.", "Whenever someone gets a chance.", "Low priority request.",
           "No rush on this."],
}
RES_OPEN = ["", "", "Remote session with user. ", "Called user and checked. ", "Investigated. ",
            "Connected via remote assist. "]
RES_RED_HERRING = ["Restarted the device first, issue persisted. ", "Asked user to reboot; no change. ",
                   "Checked for known outages, none found. "]
RES_CLOSE = ["User confirmed issue resolved.", "Confirmed working, closing.", "Resolved.",
             "Monitored, no recurrence.", ""]

VAGUE = [
    ("Issue", "System is not working please check asap."),
    ("Not working", "Something is wrong with my computer, please call me on {phone}."),
    ("Help needed", "cant work since morning pls help"),
    ("urgent", "Error coming again. Please fix."),
    ("Problem", "The application is giving some error, not sure which one."),
    ("Laptop issue", "My laptop has a problem."),
    ("Access", "Need access."),
    ("Error", "Getting an error when I click the button."),
    ("Network?", "Things are slow today, not sure if it's the network or the app."),
    ("please call", "Please call me back regarding my issue. {name}"),
    ("Not able to work", "Nothing is opening properly."),
    ("Email", "Something with email."),
    ("System down?", "Is the system down? I can't do anything."),
    ("Follow up", "Following up on my earlier issue, still not fixed."),
    ("Weird behaviour", "Screen behaves weird sometimes."),
]

TYPO_RE = re.compile(r"[A-Za-z]{5,}")


def wchoice(rng, dist):
    keys = list(dist)
    return rng.choices(keys, weights=[dist[k] for k in keys])[0]


def rand_dt(rng, start, end, business=True):
    while True:
        d = start + timedelta(seconds=rng.randint(0, int((end - start).total_seconds())))
        if not business:
            return d
        if d.weekday() < 5 and 8 <= d.hour < 19:
            return d
        if rng.random() < 0.12:  # some out-of-hours tickets
            return d


def person(rng):
    f, l = rng.choice(FIRST), rng.choice(LAST)
    return f"{f} {l}", f"{f.lower()}.{l.lower()}@{DOMAIN}"


def phone(rng):
    return "+91 9" + "".join(str(rng.randint(0, 9)) for _ in range(9))


def fill(text, v):
    for k, val in v.items():
        text = text.replace("{" + k + "}", str(val))
    return text


def typo(rng, text):
    words = TYPO_RE.findall(text)
    if not words:
        return text
    w = rng.choice(words)
    i = rng.randint(1, len(w) - 3)
    w2 = w[:i] + w[i + 1] + w[i] + w[i + 2:]
    return text.replace(w, w2, 1)


def make_values(rng, s):
    host = (f"LT-{rng.randint(1000, 9999)}" if s["category"] == "Hardware"
            else f"{rng.choice(['APP', 'DB', 'FS', 'BKP'])}-SRV-{rng.randint(1, 40):02d}")
    site = rng.choice(SITES)
    return dict(app=rng.choice(s["apps"]), code=rng.choice(s["codes"]), host=host, site=site,
                floor=rng.choice(FLOORS), desk=f"{rng.choice('ABCD')}-{rng.randint(100, 520)}",
                printer=f"PRN-{site[:3].upper()}-{rng.randint(1, 6)}F-{rng.randint(1, 9):02d}",
                mbx=rng.choice(MAILBOXES), n=rng.randint(5, 40))


def build_text(rng, s, prio, v, extra_symptom=None):
    title = fill(rng.choice(s["titles"]), v)
    if prio == "P1" and rng.random() < 0.5:
        title = "URGENT: " + title
    if rng.random() < 0.2:
        title = title.lower()

    parts = []
    if rng.random() < 0.4:
        parts.append(rng.choice(GREETINGS))
    syms = rng.sample(s["symptoms"], k=2 if rng.random() < 0.35 else 1)
    parts += [fill(x, v) for x in syms]
    if extra_symptom:
        parts.append(extra_symptom)
    if s["context"] and rng.random() < 0.5:
        parts.append(fill(rng.choice(s["context"]), v))
    if rng.random() < 0.4:
        parts.append(rng.choice(DURATION))
    if rng.random() < 0.65:
        parts.append(fill(rng.choice(IMPACT[prio]), v))
    if rng.random() < 0.3:
        parts.append(rng.choice(TRIED))
    if rng.random() < 0.25 and "{host}" not in " ".join(syms):
        parts.append(f"My laptop is {v['host'] if v['host'].startswith('LT') else 'LT-' + str(rng.randint(1000, 9999))}.")
    if rng.random() < 0.5:
        parts.append(rng.choice(CLOSINGS))
    desc = " ".join(parts)
    if rng.random() < 0.4:  # signature with PII for masking tests
        name, email = person(rng)
        desc += f"\n\nRegards,\n{name}\n{email} | {phone(rng)}"
    if rng.random() < 0.1:
        desc = typo(rng, desc)
    if rng.random() < 0.12:
        desc = desc.lower()
    return title, desc


def build_resolution(rng, s, v):
    steps = [fill(x, v) for x in s["steps"]]
    key = min(s["key_step"], len(steps) - 1)
    kept = [st for i, st in enumerate(steps) if i == key or rng.random() < 0.85]
    if len(kept) < 2:
        kept = steps[:2] if key < 2 else [steps[0], steps[key]]
    sep = rng.choice([". ", "; ", "\n- "])
    body = sep.join(kept)
    if sep == "\n- ":
        body = "- " + body
    note = rng.choice(RES_OPEN)
    if rng.random() < 0.2:
        note += rng.choice(RES_RED_HERRING)
    note += body + ("." if not body.endswith(".") else "")
    close = rng.choice(RES_CLOSE)
    return (note + (" " + close if close else "")).strip()


def resolve_hours(rng, prio):
    lo, hi = {"P1": (1, 6), "P2": (2, 24), "P3": (4, 72), "P4": (8, 120)}[prio]
    return rng.uniform(lo, hi)


def sensitivity(rng, s):
    if s["sens"]:
        return s["sens"]
    return wchoice(rng, {"Internal": 0.85, "Public": 0.10, "Restricted": 0.05})


def allocate(total, scenarios, minimum):
    w = sum(s["weight"] for s in scenarios)
    counts = {s["id"]: max(minimum, round(total * s["weight"] / w)) for s in scenarios}
    diff = total - sum(counts.values())
    ids = sorted(counts, key=lambda k: -counts[k])
    i = 0
    while diff != 0:
        k = ids[i % len(ids)]
        step = 1 if diff > 0 else -1
        if counts[k] + step >= minimum:
            counts[k] += step
            diff -= step
        i += 1
    return counts


def kb_record(s, kb_id, rng):
    v = dict(app=s["apps"][0], code=s["codes"][0] or "", host="the affected host", site="the site",
             floor="the floor", desk="the desk", printer="the printer", mbx="the shared mailbox", n="several")
    title_core = fill(s["titles"][0], v)
    return dict(kb_id=kb_id, title=f"Troubleshooting: {title_core}", scenario_id=s["id"],
                category=s["category"], subcategory=s["subcategory"], applies_to=", ".join(s["apps"]),
                error_codes=", ".join(c for c in s["codes"] if c),
                symptoms=" ".join(fill(x, v) for x in s["symptoms"][:3]),
                cause=s["root_cause"],
                resolution_steps="\n".join(f"{i + 1}. {fill(x, v)}" for i, x in enumerate(s["steps"])),
                owner_group=s["group"],
                last_reviewed=(datetime(2025, 1, 1) + timedelta(days=rng.randint(0, 520))).date().isoformat(),
                approved=True, version="1.0", note="")


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    raw, labels = os.path.join(a.out, "raw"), os.path.join(a.out, "labels")
    os.makedirs(raw, exist_ok=True)
    os.makedirs(labels, exist_ok=True)
    by_id = {s["id"]: s for s in SCENARIOS + NOVEL_SCENARIOS}

    # ---------------- KB articles
    kbs, kb_of = [], {}
    for s in SCENARIOS:
        if s["kb"]:
            kb_id = f"KB{len(kbs) + 1:04d}"
            kbs.append(kb_record(s, kb_id, rng))
            kb_of[s["id"]] = kb_id
    for d in DRAFT_KBS:
        s = by_id[d["scenario"]]
        kbs.append(dict(kb_id=f"KB{len(kbs) + 1:04d}", title=d["title"], scenario_id=s["id"],
                        category=s["category"], subcategory=s["subcategory"], applies_to=", ".join(s["apps"]),
                        error_codes=", ".join(c for c in s["codes"] if c),
                        symptoms=" ".join(s["symptoms"][:2]).replace("{app}", s["apps"][0]).replace("{code}", s["codes"][0]),
                        cause=d["cause"], resolution_steps="\n".join(f"{i + 1}. {x}" for i, x in enumerate(d["steps"])),
                        owner_group=s["group"], last_reviewed="2024-11-15", approved=False,
                        version="0.9", note=d["note"]))

    # ---------------- historical tickets
    hist = []
    for sid, n in allocate(N_HIST, SCENARIOS, 15).items():
        s = by_id[sid]
        for _ in range(n):
            prio = wchoice(rng, s["prio"])
            v = make_values(rng, s)
            title, desc = build_text(rng, s, prio, v)
            created = rand_dt(rng, HIST_START, HIST_END)
            row = dict(title=title, description=desc, application=v["app"],
                       environment=wchoice(rng, s["env"] or {"Production": 1.0}),
                       category=s["category"], subcategory=s["subcategory"], priority=prio,
                       assignment_group=s["group"], created_at=created, sensitivity=sensitivity(rng, s),
                       scenario_id=sid, is_synthetic=True)
            if rng.random() < 0.04:
                row.update(status=rng.choice(["Open", "In Progress"]), resolved_at="",
                           resolution_notes="", root_cause="", kb_refs="")
            else:
                resolved = created + timedelta(hours=resolve_hours(rng, prio))
                row.update(resolved_at=resolved,
                           status="Closed" if resolved < DATA_CUT - timedelta(days=7) else "Resolved",
                           resolution_notes=build_resolution(rng, s, v),
                           root_cause=s["root_cause"] if rng.random() < 0.85 else "",
                           kb_refs=kb_of.get(sid, "") if rng.random() < 0.45 else "")
            hist.append(row)
    hist.sort(key=lambda r: r["created_at"])
    for i, r in enumerate(hist):
        r["ticket_id"] = f"INC{10001 + i:07d}"

    # ---------------- test tickets
    test, tlab = [], []

    def add_test(s, test_type, expected, prio=None, title=None, desc=None, scen_id=None, app=None):
        prio = prio or wchoice(rng, s["prio"])
        v = make_values(rng, s)
        if title is None:
            title, desc = build_text(rng, s, prio, v)
        test.append(dict(title=title, description=desc, application=app if app is not None else v["app"],
                         environment=wchoice(rng, s["env"] or {"Production": 1.0}),
                         created_at=rand_dt(rng, TEST_START, TEST_END), status="New",
                         sensitivity=sensitivity(rng, s), is_synthetic=True))
        tlab.append(dict(test_type=test_type, scenario_id=scen_id or s["id"], category=s["category"],
                         subcategory=s["subcategory"], priority=prio, expected_assignment_group=s["group"],
                         expected_behavior=expected, reference_root_cause=s["root_cause"],
                         reference_resolution=" | ".join(s["steps"]),
                         relevant_kb=kb_of.get(s["id"], "") if test_type == "standard" else ""))

    for sid, n in allocate(N_TEST_STD, SCENARIOS, 3).items():
        for _ in range(n):
            add_test(by_id[sid], "standard", "confident_answer_expected")
    for i in range(N_NOVEL):
        s = NOVEL_SCENARIOS[i % len(NOVEL_SCENARIOS)]
        add_test(s, "novel", "low_confidence_escalate")
    for title, desc in VAGUE:
        name, _ = person(rng)
        test.append(dict(title=title, description=desc.replace("{phone}", phone(rng)).replace("{name}", name),
                         application="", environment="Production",
                         created_at=rand_dt(rng, TEST_START, TEST_END), status="New",
                         sensitivity="Internal", is_synthetic=True))
        tlab.append(dict(test_type="vague", scenario_id="", category="", subcategory="", priority="",
                         expected_assignment_group="Service-Desk-L1",
                         expected_behavior="low_confidence_ask_clarifying_questions",
                         reference_root_cause="", reference_resolution="", relevant_kb=""))
    for i in range(N_CONTRA):
        a_id, b_id, _ = HARD_NEGATIVE_PAIRS[i % len(HARD_NEGATIVE_PAIRS)]
        sa, sb = by_id[a_id], by_id[b_id]
        va, vb = make_values(rng, sa), make_values(rng, sb)
        vb["app"], vb["code"] = va["app"], va["code"] if va["code"] else vb["code"]
        prio = wchoice(rng, sa["prio"])
        title = fill(rng.choice(sa["titles"]), va)
        desc = (fill(rng.choice(sa["symptoms"]), va) + " " + fill(rng.choice(sb["symptoms"]), vb) + " "
                + rng.choice(["Not sure if these are related.", "Some people say it's only me, others say it's everyone.", ""]))
        add_test(sa, "contradictory", "medium_or_low_flag_conflicting_evidence", prio=prio, title=title,
                 desc=desc.strip(), scen_id=f"{a_id}|{b_id}", app=va["app"])

    order = sorted(range(len(test)), key=lambda i: test[i]["created_at"])
    test = [test[i] for i in order]
    tlab = [tlab[i] for i in order]
    for i, (t, l) in enumerate(zip(test, tlab)):
        t["ticket_id"] = l["ticket_id"] = f"INC{200001 + i:07d}"

    # ---------------- retrieval labels
    resolved_by_scen = {}
    for r in hist:
        if r["status"] in ("Resolved", "Closed"):
            resolved_by_scen.setdefault(r["scenario_id"], []).append(r["ticket_id"])
    partner = {}
    for a_id, b_id, why in HARD_NEGATIVE_PAIRS:
        partner.setdefault(a_id, []).append((b_id, why))
        partner.setdefault(b_id, []).append((a_id, why))
    qrels, hneg = [], []
    for t, l in zip(test, tlab):
        if l["test_type"] == "standard":
            sid = l["scenario_id"]
            for d in resolved_by_scen.get(sid, []):
                qrels.append(dict(query_id=t["ticket_id"], doc_id=d, doc_type="ticket", relevance=1))
            if kb_of.get(sid):
                qrels.append(dict(query_id=t["ticket_id"], doc_id=kb_of[sid], doc_type="kb", relevance=1))
            for b_id, why in partner.get(sid, []):
                for d in resolved_by_scen.get(b_id, []):
                    hneg.append(dict(query_id=t["ticket_id"], doc_id=d, query_scenario=sid,
                                     doc_scenario=b_id, reason=why))
        elif l["test_type"] == "contradictory":
            for sid in l["scenario_id"].split("|"):
                for d in resolved_by_scen.get(sid, []):
                    qrels.append(dict(query_id=t["ticket_id"], doc_id=d, doc_type="ticket", relevance=1))

    # pairs for a manual sanity check
    resolved = [r for r in hist if r["status"] in ("Resolved", "Closed")]
    pairs = []
    for _ in range(50):
        sid = rng.choice(list(resolved_by_scen))
        x, y = rng.sample(resolved_by_scen[sid], 2)
        pairs.append(dict(ticket_a=x, ticket_b=y, expected="related", pair_type="same_scenario"))
    for _ in range(20):
        a_id, b_id, _ = rng.choice(HARD_NEGATIVE_PAIRS)
        pairs.append(dict(ticket_a=rng.choice(resolved_by_scen[a_id]), ticket_b=rng.choice(resolved_by_scen[b_id]),
                          expected="not_related", pair_type="hard_negative"))
    rng.shuffle(pairs)
    by_tid = {r["ticket_id"]: r for r in hist}
    for p in pairs:
        p["title_a"], p["title_b"] = by_tid[p["ticket_a"]]["title"], by_tid[p["ticket_b"]]["title"]
        p["reviewer_agrees"] = ""

    # ---------------- write
    fmt = lambda d: d.strftime("%Y-%m-%d %H:%M") if isinstance(d, datetime) else d
    for r in hist + test:
        r["created_at"] = fmt(r["created_at"])
        if "resolved_at" in r:
            r["resolved_at"] = fmt(r["resolved_at"])
    hist_fields = ["ticket_id", "title", "description", "application", "environment", "category",
                   "subcategory", "priority", "assignment_group", "status", "created_at", "resolved_at",
                   "resolution_notes", "root_cause", "kb_refs", "sensitivity", "scenario_id", "is_synthetic"]
    write_csv(os.path.join(raw, "historical_tickets.csv"), hist, hist_fields)
    write_csv(os.path.join(raw, "test_tickets.csv"), test,
              ["ticket_id", "title", "description", "application", "environment", "status",
               "created_at", "sensitivity", "is_synthetic"])
    write_csv(os.path.join(labels, "test_labels.csv"), tlab,
              ["ticket_id", "test_type", "scenario_id", "category", "subcategory", "priority",
               "expected_assignment_group", "expected_behavior", "reference_root_cause",
               "reference_resolution", "relevant_kb"])
    write_csv(os.path.join(labels, "test_qrels.csv"), qrels, ["query_id", "doc_id", "doc_type", "relevance"])
    write_csv(os.path.join(labels, "test_hard_negatives.csv"), hneg,
              ["query_id", "doc_id", "query_scenario", "doc_scenario", "reason"])
    write_csv(os.path.join(labels, "duplicate_pairs_to_verify.csv"), pairs,
              ["ticket_a", "ticket_b", "title_a", "title_b", "pair_type", "expected", "reviewer_agrees"])
    kb_fields = ["kb_id", "title", "scenario_id", "category", "subcategory", "applies_to", "error_codes",
                 "symptoms", "cause", "resolution_steps", "owner_group", "last_reviewed", "approved",
                 "version", "note"]
    write_csv(os.path.join(raw, "kb_articles.csv"), kbs, kb_fields)
    with open(os.path.join(raw, "kb_articles.json"), "w", encoding="utf-8") as f:
        json.dump(kbs, f, indent=2)
    with open(os.path.join(labels, "scenarios.json"), "w", encoding="utf-8") as f:
        json.dump({"scenarios": SCENARIOS, "novel_scenarios": NOVEL_SCENARIOS,
                   "hard_negative_pairs": HARD_NEGATIVE_PAIRS}, f, indent=2)
    write_csv(os.path.join(a.out, "users.csv"),
              [dict(user_id="analyst1", role="L1", can_see="Public,Internal"),
               dict(user_id="analyst2", role="L1", can_see="Public,Internal"),
               dict(user_id="admin1", role="L2_Admin", can_see="Public,Internal,Restricted")],
              ["user_id", "role", "can_see"])

    tax = {}
    for s in SCENARIOS:
        tax.setdefault(s["category"], [])
        if s["subcategory"] not in tax[s["category"]]:
            tax[s["category"]].append(s["subcategory"])
    default_group = {"Access": "Access-Management", "Application": "App-Support-L2", "Network": "Network-L2",
                     "Email and collaboration": "Messaging", "Database": "DBA",
                     "Hardware": "Hardware-Support", "Configuration": "Desktop-Support"}
    lines = ["# Ticket taxonomy (category > subcategory) and default escalation groups", "categories:"]
    for c, subs in tax.items():
        lines += [f"  {c}:", f"    default_group: {default_group[c]}", "    subcategories:"]
        lines += [f"      - {x}" for x in subs]
    lines += ["priorities:",
              "  P1: {name: Critical, rule: 'Production outage or security incident, many users, no workaround'}",
              "  P2: {name: High, rule: 'Key function broken for a team or site, no workaround'}",
              "  P3: {name: Medium, rule: 'Single user impaired, workaround exists'}",
              "  P4: {name: Low, rule: 'Request, question or cosmetic issue'}",
              "priority_signals:",
              "  P1: ['production is down', 'all users', 'outage', 'security incident', 'phishing', 'customers are affected']",
              "  P2: ['entire team', 'whole office', 'several colleagues', 'cannot process', 'month-end', 'nobody can']",
              "  P3: ['only me', 'only my machine', 'work around', 'few times a day']",
              "  P4: ['not urgent', 'no rush', 'request', 'whenever']"]
    with open(os.path.join(a.out, "taxonomy.yaml"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"historical={len(hist)} kb={len(kbs)} test={len(test)} qrels={len(qrels)} "
          f"hard_negatives={len(hneg)} pairs={len(pairs)}")


if __name__ == "__main__":
    main()
