"""
log_pipeline.py
────────────────────────────────────────────────────────────────────────────
Synthetic CERT r6.2-style raw log generator + parser/feature-engineering
pipeline for the "Log Simulation" dashboard view.

Mirrors the project's Data Layer (Chapter 3, System Architecture):
    CERT Logs → Log Parser → Feature Engineer → Feature Store

This module:
  1. generate_log(scenario_key, user, day) → raw CERT-style CSV log lines
     for ONE user-day, across logon/file/email/http/device event types,
     plus a human-readable translation of the same events.

  2. parse_and_engineer(raw_csv_text) → aggregates the raw log lines
     into a single user-day feature row (counts, first/last hour,
     unique_pcs, activity_type_count, *_dev deviation features) — the same
     feature set the trained XGBoost model expects.
"""

import random
import string
from datetime import datetime, timedelta

# ── event-type → CERT-style column schema ─────────────────────────────────
# Mirrors the real CERT r6.2 logon.csv / file.csv / email.csv / http.csv /
# device.csv structure: {id},{date},{user},{pc},{activity}
EVENT_SCHEMAS = {
    "logon" : ["id", "date", "user", "pc", "activity"],          # Logon / Logoff
    "file"  : ["id", "date", "user", "pc", "filename", "activity"],  # File copy/open
    "email" : ["id", "date", "user", "pc", "to", "cc", "size", "attachments"],
    "http"  : ["id", "date", "user", "pc", "url", "activity"],
    "device": ["id", "date", "user", "pc", "activity"],          # USB connect/disconnect
}

_LOGON_ACTIVITIES  = ["Logon", "Logoff"]
_FILE_ACTIVITIES   = ["File Open", "File Copy", "File Write"]
_HTTP_ACTIVITIES   = ["WWW Visit", "WWW Download", "WWW Upload"]
_DEVICE_ACTIVITIES = ["Connect", "Disconnect"]

_SAMPLE_DOMAINS  = ["gmail.com", "yahoo.com", "dropbox.com", "wikileaks.org",
                     "company-internal.com", "linkedin.com", "github.com"]
_SAMPLE_FILES    = ["report.docx", "salary_data.xlsx", "source_code.zip",
                     "client_list.csv", "presentation.pptx", "backup.tar.gz",
                     "budget_2026.xlsx", "passwords.txt"]


def _rand_id(prefix="EVT"):
    return prefix + "".join(random.choices(string.digits, k=8))


def _fmt_dt(day_str: str, hour: int, minute: int = None) -> str:
    minute = minute if minute is not None else random.randint(0, 59)
    return f"{day_str} {hour:02d}:{minute:02d}:00"


# ════════════════════════════════════════════════════════════════════════
# 1. LOG GENERATION
# ════════════════════════════════════════════════════════════════════════

# scenario_key → generation rules (counts + hour ranges per event type)
SCENARIO_LOG_RULES = {
    "normal_activity": {
        "label"      : "Normal Activity",
        "description": "Inactive day — minimal file/device events, typical email & web browsing during work hours.",
        "logon"  : (1, 2,  8, 17),
        "file"   : (0, 0,  9, 17),
        "email"  : (6, 10, 9, 17),
        "http"   : (60, 100, 9, 17),
        "device" : (0, 0,  9, 17),
        "pcs"    : 1,
    },
    "data_exfiltration": {
        "label"      : "Data Exfiltration",
        "description": "Large file copy + USB device activity clustered late at night, multiple machines.",
        "logon"  : (2, 3,  20, 23),
        "file"   : (15, 22, 21, 23),
        "email"  : (1, 3,   9, 17),
        "http"   : (3, 8,   9, 17),
        "device" : (10, 16, 21, 23),
        "pcs"    : 2,
    },
    "email_exfiltration": {
        "label"      : "Email Exfiltration",
        "description": "Abnormally high outbound email volume, several with attachments, during work hours.",
        "logon"  : (1, 2,   8, 17),
        "file"   : (3, 7,   9, 17),
        "email"  : (30, 38, 8, 17),
        "http"   : (50, 90, 9, 17),
        "device" : (0, 0,   9, 17),
        "pcs"    : 1,
    },
    "off_hours_browsing": {
        "label"      : "Off-Hours Web Browsing",
        "description": "Heavy web activity between midnight and 4 AM, minimal other activity.",
        "logon"  : (1, 1,   20, 22),
        "file"   : (0, 0,   0, 4),
        "email"  : (0, 2,   0, 4),
        "http"   : (250, 320, 0, 4),
        "device" : (0, 0,   0, 4),
        "pcs"    : 1,
    },
}


def _rand_count(lo, hi):
    return random.randint(lo, hi) if hi > lo else lo


def generate_log(scenario_key: str, user: str = None, day: str = None) -> dict:
    """
    Generate a synthetic CERT r6.2-style raw log for one user-day.

    Returns a dict with:
        csv_text      : raw CERT-style CSV log lines (all event types mixed,
                         sorted by timestamp — like a real combined log dump)
        readable_text : human-readable line-by-line translation
        user, day     : the identifiers used
        scenario_key  : echoed back for reference
    """
    if scenario_key not in SCENARIO_LOG_RULES:
        scenario_key = "normal_activity"
    rules = SCENARIO_LOG_RULES[scenario_key]

    user = user or ("U" + "".join(random.choices(string.digits, k=4)))
    day  = day  or (datetime(2011, 1, 1) + timedelta(days=random.randint(0, 500))).strftime("%m/%d/%Y")

    pcs = [f"PC-{random.randint(1,20):02d}" for _ in range(max(1, rules["pcs"]))]

    events = []  # list of (datetime, event_type, row_dict)

    def add_events(event_type, lo, hi, h_lo, h_hi, builder):
        n = _rand_count(lo, hi)
        for _ in range(n):
            hour = random.randint(h_lo, h_hi) if h_hi >= h_lo else h_lo
            dt_str = _fmt_dt(day, hour)
            row = builder(dt_str)
            events.append((dt_str, event_type, row))

    # logon
    add_events("logon", *rules["logon"], lambda dt: {
        "id": _rand_id("LOG"), "date": dt, "user": user,
        "pc": random.choice(pcs), "activity": random.choice(_LOGON_ACTIVITIES),
    })
    # file
    add_events("file", *rules["file"], lambda dt: {
        "id": _rand_id("FIL"), "date": dt, "user": user,
        "pc": random.choice(pcs), "filename": random.choice(_SAMPLE_FILES),
        "activity": random.choice(_FILE_ACTIVITIES),
    })
    # email
    add_events("email", *rules["email"], lambda dt: {
        "id": _rand_id("EML"), "date": dt, "user": user,
        "pc": random.choice(pcs),
        "to": f"{random.choice(string.ascii_lowercase)}@{random.choice(_SAMPLE_DOMAINS)}",
        "cc": "", "size": random.randint(1000, 500000),
        "attachments": random.randint(0, 3),
    })
    # http
    add_events("http", *rules["http"], lambda dt: {
        "id": _rand_id("WEB"), "date": dt, "user": user,
        "pc": random.choice(pcs),
        "url": f"http://{random.choice(_SAMPLE_DOMAINS)}/page{random.randint(1,99)}",
        "activity": random.choice(_HTTP_ACTIVITIES),
    })
    # device
    add_events("device", *rules["device"], lambda dt: {
        "id": _rand_id("DEV"), "date": dt, "user": user,
        "pc": random.choice(pcs), "activity": random.choice(_DEVICE_ACTIVITIES),
    })

    # sort combined log by timestamp (like a real merged log dump)
    events.sort(key=lambda e: e[0])

    # ── build raw CSV text (CERT-style, one section per event type header) ──
    csv_lines = ["# event_type,id,date,user,pc,detail1,detail2,detail3,detail4"]
    for dt_str, etype, row in events:
        if etype == "logon":
            csv_lines.append(f"logon,{row['id']},{row['date']},{row['user']},{row['pc']},{row['activity']}")
        elif etype == "file":
            csv_lines.append(f"file,{row['id']},{row['date']},{row['user']},{row['pc']},{row['filename']},{row['activity']}")
        elif etype == "email":
            csv_lines.append(f"email,{row['id']},{row['date']},{row['user']},{row['pc']},{row['to']},{row['cc']},{row['size']},{row['attachments']}")
        elif etype == "http":
            csv_lines.append(f"http,{row['id']},{row['date']},{row['user']},{row['pc']},{row['url']},{row['activity']}")
        elif etype == "device":
            csv_lines.append(f"device,{row['id']},{row['date']},{row['user']},{row['pc']},{row['activity']}")
    csv_text = "\n".join(csv_lines)

    # ── build human-readable translation ─────────────────────────────────
    readable_lines = []
    for dt_str, etype, row in events:
        time_part = dt_str.split(" ")[1]
        if etype == "logon":
            readable_lines.append(f"[{time_part}] {row['user']} {row['activity'].lower()} on {row['pc']}")
        elif etype == "file":
            readable_lines.append(f"[{time_part}] {row['user']} performed '{row['activity']}' on '{row['filename']}' ({row['pc']})")
        elif etype == "email":
            att = f", {row['attachments']} attachment(s)" if row['attachments'] else ""
            readable_lines.append(f"[{time_part}] {row['user']} emailed {row['to']} ({row['size']} bytes{att})")
        elif etype == "http":
            readable_lines.append(f"[{time_part}] {row['user']} {row['activity'].lower()} → {row['url']}")
        elif etype == "device":
            readable_lines.append(f"[{time_part}] {row['user']} {row['activity'].lower()}ed USB device on {row['pc']}")
    readable_text = "\n".join(readable_lines) if readable_lines else "(no events generated)"

    return {
        "csv_text"     : csv_text,
        "readable_text": readable_text,
        "user"         : user,
        "day"          : day,
        "scenario_key" : scenario_key,
        "label"        : rules["label"],
        "description"  : rules["description"],
    }


# ════════════════════════════════════════════════════════════════════════
# 2. LOG PARSER + FEATURE ENGINEER
# ════════════════════════════════════════════════════════════════════════

class LogParseError(Exception):
    pass


def parse_and_engineer(raw_csv_text: str) -> dict:
    """
    Parse raw CERT-style CSV log lines and aggregate into a single
    user-day feature row, mirroring the project's Feature Engineer step.

    Expected line format (comment lines starting with # are skipped):
        event_type,id,date,user,pc,detail1[,detail2,detail3,detail4]

    Returns a dict of engineered features (NOT yet aligned to the model's
    FEATURE_COLS — that alignment happens in app.py, same as the rest of
    the pipeline, to keep this module dataset/model-agnostic).
    """
    lines = [l.strip() for l in raw_csv_text.strip().splitlines()
             if l.strip() and not l.strip().startswith("#")]

    if not lines:
        raise LogParseError("No log lines found. Generate or paste a log first.")

    counts   = {"logon": 0, "file": 0, "email": 0, "http": 0, "device": 0}
    hours    = []
    pcs_seen = set()
    users_seen = set()
    days_seen  = set()
    activity_types = set()

    for ln, line in enumerate(lines, start=1):
        parts = line.split(",")
        if len(parts) < 5:
            raise LogParseError(f"Line {ln} malformed (expected ≥5 comma-separated fields): {line}")

        etype = parts[0].strip().lower()
        if etype not in counts:
            raise LogParseError(f"Line {ln} has unknown event_type '{etype}' "
                                 f"(expected one of: logon, file, email, http, device)")

        _id, date_str, user, pc = parts[1], parts[2], parts[3], parts[4]
        counts[etype] += 1
        activity_types.add(etype)
        pcs_seen.add(pc.strip())
        users_seen.add(user.strip())

        # parse hour from date string "MM/DD/YYYY HH:MM:SS"
        try:
            dt = datetime.strptime(date_str.strip(), "%m/%d/%Y %H:%M:%S")
            hours.append(dt.hour)
            days_seen.add(dt.strftime("%m/%d/%Y"))
        except ValueError:
            raise LogParseError(f"Line {ln} has unparsable date '{date_str}' "
                                 f"(expected MM/DD/YYYY HH:MM:SS)")

    if len(users_seen) > 1:
        raise LogParseError(f"Log contains multiple users {users_seen} — "
                             f"this pipeline processes one user-day at a time.")
    if len(days_seen) > 1:
        raise LogParseError(f"Log spans multiple days {days_seen} — "
                             f"this pipeline processes one user-day at a time.")

    user = users_seen.pop() if users_seen else "UNKNOWN"
    day  = days_seen.pop()  if days_seen  else "UNKNOWN"

    first_hour = min(hours) if hours else 0
    last_hour  = max(hours) if hours else 0

    features = {
        "user"               : user,
        "day"                : day,
        "logon_count"        : counts["logon"],
        "file_count"         : counts["file"],
        "email_count"        : counts["email"],
        "http_count"         : counts["http"],
        "device_count"       : counts["device"],
        "unique_pcs"         : len(pcs_seen),
        "first_hour"         : first_hour,
        "last_hour"          : last_hour,
        "activity_type_count": len(activity_types),
        "n_log_lines"        : len(lines),
    }
    return features


# ════════════════════════════════════════════════════════════════════════
# 3. BATCH GENERATION — multiple random users/user-days in one shot
# ════════════════════════════════════════════════════════════════════════

def generate_batch(n: int = 30, seed: int = None) -> list:
    """
    Automatically generate `n` synthetic user-day raw logs across random
    users and a random mix of scenarios, then parse each one into an
    engineered feature row. This is the backend half of the fully-automatic
    "Log Simulation" pipeline — no manual log editing involved.

    Scenario mix is weighted so most generated days look normal (~70%),
    with the remaining ~30% spread across the suspicious scenarios — this
    mirrors the extreme class imbalance described in the Stage 3 report
    (Section 4.2: only 25 insider instances out of ~1.39M rows).

    Returns a list of dicts, each containing:
        user, day, scenario_key, scenario_label,
        csv_text, readable_text,
        engineered (dict of features incl. user/day/n_log_lines)
    """
    rng = random.Random(seed)  # local RNG so it doesn't disturb global random state

    scenario_keys    = list(SCENARIO_LOG_RULES.keys())
    normal_key       = "normal_activity"
    suspicious_keys  = [k for k in scenario_keys if k != normal_key]

    # 70% normal / 30% suspicious (spread evenly across suspicious scenarios)
    weighted_choices = [normal_key] * 7 + suspicious_keys * 3

    results = []
    used_users = set()

    for _ in range(n):
        scenario_key = rng.choice(weighted_choices)

        # unique-ish random user id per generated record
        user = "U" + "".join(rng.choices(string.digits, k=4))
        while user in used_users:
            user = "U" + "".join(rng.choices(string.digits, k=4))
        used_users.add(user)

        day = (datetime(2011, 1, 1) +
               timedelta(days=rng.randint(0, 500))).strftime("%m/%d/%Y")

        # temporarily seed the module-level `random` so generate_log()'s
        # internal random.* calls are reproducible per-record when a seed
        # is supplied, without affecting the rest of the app's RNG state
        if seed is not None:
            random.seed(rng.randint(0, 10_000_000))

        log = generate_log(scenario_key, user=user, day=day)

        try:
            engineered = parse_and_engineer(log["csv_text"])
        except LogParseError:
            # extremely unlikely (generator always produces valid lines),
            # but skip gracefully rather than crash a whole batch
            continue

        results.append({
            "user"          : user,
            "day"           : day,
            "scenario_key"  : scenario_key,
            "scenario_label": log["label"],
            "csv_text"      : log["csv_text"],
            "readable_text" : log["readable_text"],
            "engineered"    : engineered,
        })

    return results