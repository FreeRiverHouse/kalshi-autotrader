#!/usr/bin/env python3
"""
Kalshi AI Feedback Watchdog
Runs every 30 minutes via LaunchAgent.

Workflow:
  1. Fetch current metrics + read key source files
  2. Ask Kimi K2 (via NVIDIA API) for code review & improvement suggestions
  3. Parse feedback into prioritised task list → data/trading/grok-tasks.json
  4. Pick top-priority pending task → ask Claude Sonnet to implement it
  5. Apply the change + restart affected process
  6. Log everything → data/trading/grok-watchdog.log
"""

import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

BASE          = Path(__file__).parent
TASKS_FILE    = BASE / "data" / "trading" / "grok-tasks.json"
LOG_FILE      = BASE / "data" / "trading" / "grok-watchdog.log"
BACKUP_DIR    = BASE / "data" / "trading" / "backups"
DASHBOARD_URL = "http://localhost:8888"
PYTHON        = "/opt/homebrew/bin/python3.11"

# Source files to review (sections only — keep prompts short)
REVIEW_FILES = {
    "kalshi-autotrader.py": {
        "sections": ["CALIBRATION_FACTOR", "MIN_EDGE", "KELLY_FRACTION",
                     "DECAY_FACTOR", "dynamic_edge_min", "heuristic_score"],
        "max_lines": 300,
    },
    "kalshi-dashboard.py": {
        "sections": ["compute_metrics", "charts()", "ch-bankroll", "ch-roi"],
        "max_lines": 200,
    },
    "kalshi-feedback-agent.py": {
        "sections": ["RULES", "tune_params"],
        "max_lines": 150,
    },
}

# Auto-apply only tasks with these risk levels
AUTO_APPLY_RISKS = {"low", "medium"}

# Safety: never touch these patterns autonomously
FORBIDDEN_PATTERNS = [
    r"ORDER_EXECUTION", r"place_order", r"LIVE_MODE", r"real.*trade",
    r"kalshi.*api.*key", r"PRIVATE_KEY",
]


# ── Key loading ───────────────────────────────────────────────────────────────

def _load_keys() -> dict:
    """Load API keys from env or ~/.clawdbot/.env.trading"""
    keys = {
        "anthropic": os.environ.get("ANTHROPIC_API_KEY", ""),
        "nvidia":    os.environ.get("NVIDIA_API_KEY", ""),
    }
    env_file = Path.home() / ".clawdbot" / ".env.trading"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                keys["anthropic"] = line.split("=", 1)[1].strip()

    # Nvidia key from clawdbot config
    if not keys["nvidia"]:
        try:
            cfg = json.loads((Path.home() / ".clawdbot" / "clawdbot.json").read_text())
            nv  = cfg.get("models", {}).get("providers", {}).get("nvidia", {})
            keys["nvidia"] = nv.get("apiKey", "")
        except Exception:
            pass
    return keys


KEYS = _load_keys()


# ── Logging ───────────────────────────────────────────────────────────────────

def log(msg: str, level: str = "INFO"):
    ts   = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [{level}] {msg}"
    print(line, flush=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


# ── Source extraction ─────────────────────────────────────────────────────────

def extract_context(filename: str, cfg: dict) -> str:
    path = BASE / filename
    if not path.exists():
        return f"[{filename}: not found]"
    lines = path.read_text().splitlines()
    hits  = []
    for i, line in enumerate(lines):
        for kw in cfg["sections"]:
            if kw.lower() in line.lower():
                start = max(0, i - 3)
                end   = min(len(lines), i + 15)
                hits.append((start, end))
                break
    if not hits:
        # Fallback: first N lines
        return "\n".join(lines[:cfg["max_lines"]])
    # Merge overlapping ranges
    merged = []
    for s, e in sorted(hits):
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append([s, e])
    total, result = 0, []
    for s, e in merged:
        chunk = lines[s:e]
        result.append(f"# … (lines {s+1}-{e}) …\n" + "\n".join(chunk))
        total += len(chunk)
        if total >= cfg["max_lines"]:
            break
    return "\n\n".join(result)


def build_code_context() -> str:
    parts = []
    for fname, cfg in REVIEW_FILES.items():
        ctx = extract_context(fname, cfg)
        parts.append(f"=== {fname} ===\n{ctx}")
    return "\n\n".join(parts)


# ── Metrics fetch ─────────────────────────────────────────────────────────────

def fetch_metrics() -> dict:
    try:
        with urllib.request.urlopen(f"{DASHBOARD_URL}/api/metrics", timeout=5) as r:
            return json.loads(r.read())
    except Exception as e:
        return {"error": str(e)}


# ── Kimi K2 code review ───────────────────────────────────────────────────────

def kimi_review(code_ctx: str, metrics: dict) -> str:
    if not KEYS["nvidia"]:
        return "(NVIDIA_API_KEY not available — skipping Kimi review)"
    prompt = f"""Sei un senior quantitative developer che fa code review del Kalshi AutoTrader.

METRICHE CORRENTI:
{json.dumps(metrics, indent=2, default=str)[:800]}

CODICE CHIAVE:
{code_ctx[:3000]}

Analizza il codice e le metriche. Fornisci ESATTAMENTE 5 suggerimenti di miglioramento nel seguente formato JSON array:
[
  {{
    "id": "unique_slug",
    "title": "Titolo breve (max 60 char)",
    "description": "Cosa fare e perché (max 150 char)",
    "file": "kalshi-autotrader.py",
    "impact": "high|medium|low",
    "risk": "low|medium|high",
    "category": "performance|accuracy|monitoring|dashboard|algo"
  }}
]

REGOLE:
- Non suggerire cambi all'esecuzione ordini reali (solo paper mode)
- Priorità: accuracy del forecaster > dashboard clarity > algo tuning
- Solo cambi implementabili in <20 righe di Python
- Rispondi SOLO con il JSON array, nessun testo extra"""

    payload = json.dumps({
        "model":      "moonshotai/kimi-k2-instruct",
        "max_tokens": 1000,
        "messages":   [{"role": "user", "content": prompt}],
        "temperature": 0.3,
    }).encode()
    req = urllib.request.Request(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {KEYS['nvidia']}",
            "Content-Type":  "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            resp = json.loads(r.read())
            return resp["choices"][0]["message"]["content"]
    except Exception as e:
        return f"(Kimi error: {e})"


# ── Parse tasks ───────────────────────────────────────────────────────────────

_IMPACT_RANK = {"high": 0, "medium": 1, "low": 2}


def parse_and_merge_tasks(kimi_output: str) -> list[dict]:
    """Parse Kimi JSON, merge with existing tasks, deduplicate."""
    # Load existing
    existing = []
    if TASKS_FILE.exists():
        try:
            existing = json.loads(TASKS_FILE.read_text())
        except Exception:
            pass
    existing_ids = {t["id"] for t in existing}

    # Parse new
    new_tasks = []
    try:
        m = re.search(r'\[.*\]', kimi_output, re.DOTALL)
        if m:
            raw = json.loads(m.group(0))
            for t in raw:
                tid = t.get("id", "")
                if tid and tid not in existing_ids:
                    new_tasks.append({
                        "id":          tid,
                        "title":       t.get("title", ""),
                        "description": t.get("description", ""),
                        "file":        t.get("file", "kalshi-autotrader.py"),
                        "impact":      t.get("impact", "medium"),
                        "risk":        t.get("risk", "medium"),
                        "category":    t.get("category", "algo"),
                        "status":      "pending",
                        "created_at":  datetime.now(timezone.utc).isoformat(),
                        "done_at":     None,
                        "error":       None,
                    })
    except Exception as e:
        log(f"Task parse error: {e}", "WARN")

    merged = existing + new_tasks
    # Sort: pending first, then by impact
    merged.sort(key=lambda t: (
        0 if t["status"] == "pending" else 1,
        _IMPACT_RANK.get(t["impact"], 1),
    ))
    log(f"Tasks: {len(existing)} existing + {len(new_tasks)} new = {len(merged)} total")
    return merged


def save_tasks(tasks: list[dict]):
    TASKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    TASKS_FILE.write_text(json.dumps(tasks, indent=2, default=str))


# ── Claude implementation ─────────────────────────────────────────────────────

def claude_implement(task: dict) -> dict | None:
    """Ask Claude Sonnet to generate a specific code edit for the task."""
    if not KEYS["anthropic"]:
        return None
    target_file = BASE / task["file"]
    if not target_file.exists():
        return None
    file_content = target_file.read_text()

    # Safety check
    for pat in FORBIDDEN_PATTERNS:
        if re.search(pat, task["description"] + task["title"], re.IGNORECASE):
            log(f"Task {task['id']} blocked by safety filter (pattern: {pat})", "WARN")
            return None

    prompt = f"""Sei un Python developer. Implementa questo task nel file indicato.

TASK:
- ID: {task['id']}
- Title: {task['title']}
- Description: {task['description']}
- File: {task['file']}
- Risk: {task['risk']}

FILE CORRENTE (estratto rilevante):
```python
{file_content[:4000]}
```

Rispondi SOLO con un JSON object nel seguente formato:
{{
  "old_code": "codice esatto da sostituire (stringa esatta, incluso whitespace)",
  "new_code": "nuovo codice da inserire al posto",
  "explanation": "cosa cambia e perché (max 100 char)"
}}

REGOLE CRITICHE:
- old_code deve essere una stringa ESATTA trovata nel file (copia incolla)
- Il cambiamento deve essere minimale e chirurgico
- Non toccare order execution, API keys, o credenziali
- Se il task non è implementabile in modo sicuro, rispondi: {{"error": "reason"}}"""

    payload = json.dumps({
        "model":      "claude-sonnet-4-6",
        "max_tokens": 1500,
        "messages":   [{"role": "user", "content": prompt}],
    }).encode()

    is_oauth = KEYS["anthropic"].startswith("sk-ant-oat")
    headers  = {
        "anthropic-version": "2023-06-01",
        "content-type":      "application/json",
    }
    if is_oauth:
        headers["authorization"]  = f"Bearer {KEYS['anthropic']}"
        headers["anthropic-beta"] = "oauth-2025-04-20"
    else:
        headers["x-api-key"] = KEYS["anthropic"]

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload, headers=headers,
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            resp   = json.loads(r.read())
            text   = resp["content"][0]["text"]
            m = re.search(r'\{.*\}', text, re.DOTALL)
            if m:
                return json.loads(m.group(0))
    except Exception as e:
        log(f"Claude API error: {e}", "WARN")
    return None


# ── Apply edit ────────────────────────────────────────────────────────────────

def apply_edit(task: dict, edit: dict) -> bool:
    old_code = edit.get("old_code", "")
    new_code = edit.get("new_code", "")
    if not old_code or not new_code or old_code == new_code:
        return False

    target = BASE / task["file"]
    content = target.read_text()

    if old_code not in content:
        log(f"old_code not found in {task['file']} — skipping", "WARN")
        return False

    # Backup
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup = BACKUP_DIR / f"{task['file'].replace('/', '_')}_{task['id']}_{ts}.bak"
    shutil.copy2(target, backup)
    log(f"Backup → {backup.name}")

    # Apply
    new_content = content.replace(old_code, new_code, 1)
    target.write_text(new_content)
    log(f"Applied edit to {task['file']}: {edit.get('explanation', '')}")
    return True


def restart_process(filename: str):
    """Restart the affected process after code change."""
    if "dashboard" in filename:
        subprocess.run(["pkill", "-f", "kalshi-dashboard.py"], capture_output=True)
        time.sleep(1)
        subprocess.Popen(
            [PYTHON, "-u", str(BASE / "kalshi-dashboard.py")],
            stdout=open("/tmp/kalshi-dashboard.log", "a"),
            stderr=subprocess.STDOUT, cwd=str(BASE),
        )
        log("Dashboard restarted")
    elif "autotrader" in filename:
        subprocess.run(["pkill", "-f", "kalshi-autotrader.py"], capture_output=True)
        time.sleep(1)
        subprocess.Popen(
            [PYTHON, "-u", str(BASE / "kalshi-autotrader.py"), "--loop", "300"],
            stdout=open(str(BASE / "data" / "trading" / "kalshi-autotrader.log"), "a"),
            stderr=subprocess.STDOUT, cwd=str(BASE),
        )
        log("Autotrader restarted")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log("=" * 60)
    log("Grok Watchdog run starting")

    metrics    = fetch_metrics()
    code_ctx   = build_code_context()

    log("Calling Kimi K2 for code review…")
    kimi_out   = kimi_review(code_ctx, metrics)
    log(f"Kimi output snippet: {kimi_out[:200]}")

    tasks      = parse_and_merge_tasks(kimi_out)
    save_tasks(tasks)

    # Find top pending task eligible for auto-apply
    pending = [t for t in tasks if t["status"] == "pending"
               and t["risk"] in AUTO_APPLY_RISKS]

    if not pending:
        log("No eligible pending tasks to implement")
        log("Grok Watchdog run complete")
        log("=" * 60)
        return 0

    task = pending[0]
    log(f"Working task [{task['impact'].upper()}] {task['id']}: {task['title']}")

    # Mark in-progress
    task["status"] = "in_progress"
    save_tasks(tasks)

    edit = claude_implement(task)
    if not edit:
        log(f"Claude returned no edit for {task['id']}", "WARN")
        task["status"] = "pending"
        save_tasks(tasks)
        log("Grok Watchdog run complete")
        log("=" * 60)
        return 0

    if "error" in edit:
        log(f"Claude flagged task as unsafe: {edit['error']}", "WARN")
        task["status"]  = "skipped"
        task["error"]   = edit["error"]
        task["done_at"] = datetime.now(timezone.utc).isoformat()
        save_tasks(tasks)
        log("Grok Watchdog run complete")
        log("=" * 60)
        return 0

    success = apply_edit(task, edit)
    if success:
        task["status"]  = "done"
        task["done_at"] = datetime.now(timezone.utc).isoformat()
        save_tasks(tasks)
        restart_process(task["file"])
        log(f"Task {task['id']} DONE ✓")
    else:
        task["status"] = "pending"   # retry next time
        save_tasks(tasks)
        log(f"Task {task['id']} apply failed — will retry", "WARN")

    log("Grok Watchdog run complete")
    log("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
