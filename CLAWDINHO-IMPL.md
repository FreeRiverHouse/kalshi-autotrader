# CLAWDINHO IMPLEMENTATION GUIDE
# Kalshi AutoTrader — Task Implementation

**TU SEI CLAWDINHO. Leggi questo file e implementa i task in ordine.**
**NON saltare step. NON improvvisare. Segui esattamente.**

---

## SETUP INIZIALE (fare UNA VOLTA)

```bash
# Directory progetto
cd /Users/mattia/Projects/kalshi-autotrader

# Leggi la Grok API key
GROK_KEY=$(python3 -c "import json,os; d=json.load(open(os.path.expanduser('~/.clawdbot/clawdbot.json'))); print(d['models']['providers']['xai']['apiKey'])")

# Verifica che il trader gira
launchctl list | grep kalshi-autotrader
```

---

## TASK-002: Sostituisci Anthropic con Grok per forecast non-crypto

**File da modificare:** `/Users/mattia/Projects/kalshi-autotrader/kalshi-autotrader.py`

**Problema:** La funzione `call_claude()` usa Anthropic con chiave invalida (401). Rimpiazzala con Grok xAI.

### Step 1 — Trova la funzione `call_claude`
```bash
grep -n "def call_claude" kalshi-autotrader.py
# Annota il numero di riga
```

### Step 2 — Crea script Python per patchare il file
```python
# Salva come /tmp/patch_grok.py e lancia con python3
import json, os

# Leggi la Grok API key
cfg = json.load(open(os.path.expanduser('~/.clawdbot/clawdbot.json')))
grok_key = cfg['models']['providers']['xai']['apiKey']

content = open('/Users/mattia/Projects/kalshi-autotrader/kalshi-autotrader.py').read()

# Trova e rimpiazza il corpo di call_claude con una versione che usa Grok
old_func_start = 'def call_claude(system_prompt: str, user_prompt: str, max_tokens: int = 2000) -> dict:'

# Nuovo corpo della funzione — usa Grok xAI invece di Anthropic
new_func = f'''def call_claude(system_prompt: str, user_prompt: str, max_tokens: int = 2000) -> dict:
    """Call Grok xAI API (replaces Anthropic — key was invalid)."""
    import urllib.request
    key = "{grok_key}"
    if not key:
        return {{"error": "No Grok API key", "content": "", "tokens_used": 0}}
    payload = json.dumps({{
        "model": "grok-3-fast",
        "messages": [
            {{"role": "system", "content": system_prompt}},
            {{"role": "user", "content": user_prompt}}
        ],
        "max_tokens": max_tokens,
        "temperature": 0.3
    }}).encode()
    req = urllib.request.Request(
        "https://api.x.ai/v1/chat/completions",
        data=payload,
        headers={{
            "Authorization": f"Bearer {{key}}",
            "Content-Type": "application/json"
        }}
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            resp = json.loads(r.read())
        content = resp["choices"][0]["message"]["content"]
        tokens = resp.get("usage", {{}}).get("total_tokens", 0)
        return {{"content": content, "tokens_used": tokens, "error": None}}
    except Exception as e:
        return {{"error": str(e), "content": "", "tokens_used": 0}}
'''

# Trova la fine della funzione call_claude originale
lines = content.split('\n')
start_line = None
for i, l in enumerate(lines):
    if old_func_start in l:
        start_line = i
        break

if start_line is None:
    print("ERRORE: funzione call_claude non trovata!")
    exit(1)

# Trova la fine della funzione (prossima def allo stesso livello)
end_line = start_line + 1
while end_line < len(lines):
    l = lines[end_line]
    if l.startswith('def ') or l.startswith('class '):
        break
    end_line += 1

# Rimpiazza
before = '\n'.join(lines[:start_line])
after = '\n'.join(lines[end_line:])
new_content = before + '\n' + new_func + '\n' + after

open('/Users/mattia/Projects/kalshi-autotrader/kalshi-autotrader.py', 'w').write(new_content)
print(f"OK — call_claude rimpiazzata con Grok (linee {start_line}-{end_line})")
```

### Step 3 — Lancia il patch
```bash
python3 /tmp/patch_grok.py
```

### Step 4 — Verifica
```bash
# Controlla che la funzione ora menzioni grok
grep -A5 "def call_claude" kalshi-autotrader.py | head -6
# Deve mostrare "grok-3-fast" e "api.x.ai"
```

### Step 5 — Test rapido
```bash
python3 -c "
import sys; sys.path.insert(0,'.')
# Test chiamata Grok
import json, urllib.request, os
key = json.load(open(os.path.expanduser('~/.clawdbot/clawdbot.json')))['models']['providers']['xai']['apiKey']
payload = json.dumps({'model':'grok-3-fast','messages':[{'role':'user','content':'Say: GROK_OK'}],'max_tokens':10}).encode()
req = urllib.request.Request('https://api.x.ai/v1/chat/completions', data=payload,
    headers={'Authorization':f'Bearer {key}','Content-Type':'application/json'})
with urllib.request.urlopen(req, timeout=10) as r:
    resp = json.loads(r.read())
print(resp['choices'][0]['message']['content'])
"
# Deve stampare GROK_OK
```

### Step 6 — Riavvia il trader
```bash
launchctl unload ~/Library/LaunchAgents/com.frh.kalshi-autotrader.plist
sleep 2
launchctl load ~/Library/LaunchAgents/com.frh.kalshi-autotrader.plist
```

### Step 7 — Verifica logs dopo 3 minuti
```bash
sleep 180
grep "LLM Forecast\|Grok\|tokens" /tmp/kalshi-autotrader-bootstrap.log | tail -10
# Ora deve mostrare tokens > 0
```

### Step 8 — Commit
```bash
cd /Users/mattia/Projects/kalshi-autotrader
git add kalshi-autotrader.py
git commit -m "feat: sostituisci Anthropic con Grok xAI per forecast LLM"
git push
```

---

## TASK-004: Settlement automatico posizioni scadute

**File da modificare:** `/Users/mattia/Projects/kalshi-autotrader/kalshi-autotrader.py`

**Problema:** Le posizioni paper rimangono "open" per sempre. Il balance non si aggiorna.

### Step 1 — Trova dove vengono gestite le posizioni
```bash
grep -n "def manage_positions\|paper_trade_open\|paper_trade_close\|def load_paper_state" kalshi-autotrader.py
# Annota le righe
```

### Step 2 — Crea la funzione di settlement
Aggiungi questa funzione DOPO `manage_positions()`:

```python
def settle_paper_positions():
    """Check paper positions against Kalshi API and settle expired ones."""
    try:
        state = load_paper_state()
        positions = state.get("positions", [])
        if not positions:
            return 0
        
        settled = 0
        updated_positions = []
        
        for pos in positions:
            ticker = pos.get("ticker", "")
            action = pos.get("action", "BUY_NO")  # BUY_YES or BUY_NO
            contracts = pos.get("contracts", 1)
            entry_price = pos.get("price_cents", 50)
            
            # Check market result via Kalshi API
            result = kalshi_api("GET", f"/trade-api/v2/markets/{ticker}")
            if "error" in result:
                updated_positions.append(pos)
                continue
            
            market_result = result.get("market", {}).get("result", "")
            status = result.get("market", {}).get("status", "open")
            
            if not market_result and status == "open":
                # Still open, keep
                updated_positions.append(pos)
                continue
            
            # Market is resolved — settle
            if market_result == "yes":
                won = (action == "BUY_YES")
            elif market_result == "no":
                won = (action == "BUY_NO")
            else:
                # Voided/cancelled — refund
                state["current_balance_cents"] += contracts * entry_price
                settled += 1
                continue
            
            if won:
                # Win: pay out (100 - entry_price) per contract
                payout = contracts * (100 - entry_price)
                state["current_balance_cents"] += contracts * entry_price + payout
                state["stats"]["won"] = state["stats"].get("won", 0) + 1
                pnl = payout
            else:
                # Loss: lose entry price (already deducted)
                state["stats"]["lost"] = state["stats"].get("lost", 0) + 1
                pnl = -contracts * entry_price
            
            state["stats"]["pnl_cents"] = state["stats"].get("pnl_cents", 0) + pnl
            
            # Log closed trade
            closed = dict(pos)
            closed["result"] = market_result
            closed["won"] = won
            closed["pnl_cents"] = pnl
            closed["closed_at"] = datetime.now(timezone.utc).isoformat()
            state.setdefault("closed_trades", []).append(closed)
            settled += 1
        
        state["positions"] = updated_positions
        save_paper_state(state)
        
        if settled > 0:
            print(f"   💰 Settled {settled} paper positions")
        return settled
    
    except Exception as e:
        print(f"   ⚠️ Paper settlement error: {e}")
        return 0
```

### Step 3 — Chiama `settle_paper_positions()` all'inizio di ogni ciclo
Trova questa riga nel ciclo principale:
```bash
grep -n "Managing.*open positions\|manage_positions" kalshi-autotrader.py | head -5
```

Aggiungi PRIMA di `manage_positions()`:
```python
    # Settle expired paper positions
    if dry_run:
        settle_paper_positions()
```

### Step 4 — Commit
```bash
git add kalshi-autotrader.py
git commit -m "feat: auto-settle paper positions quando contratti scadono"
git push
```

---

## TASK-005: Escludi mercati 50/50 puri (YES=50, Vol=0)

**File:** `kalshi-autotrader.py` → funzione `filter_markets()`

### Step 1 — Trova filter_markets
```bash
grep -n "def filter_markets" kalshi-autotrader.py
```

### Step 2 — Aggiungi il filtro 50/50
Dentro `filter_markets()`, DOPO il check su `MIN_LIQUIDITY`, aggiungi:
```python
        # Skip pure 50/50 markets with no volume (no signal)
        if m.yes_price == 50 and m.volume == 0 and m.open_interest == 0:
            dte = m.days_to_expiry
            if dte > 0.083:  # Skip only if more than 2 hours to expiry
                continue
```

### Step 3 — Commit
```bash
git add kalshi-autotrader.py
git commit -m "feat: escludi mercati 50/50 puri senza volume (no signal)"
git push
```

---

## TASK-006: Kelly proporzionale all'edge

**File:** `kalshi-autotrader.py`

### Step 1 — Trova il calcolo Kelly
```bash
grep -n "kelly_frac\|KELLY_FRACTION\|kelly.*frac" kalshi-autotrader.py | head -10
```

### Step 2 — Trova la riga dove kelly_frac viene calcolato inizialmente e modifica:
Cerca:
```python
kelly_frac = edge * KELLY_FRACTION
```
O simile. Rimpiazza con:
```python
# Scale Kelly linearly with edge: 0% edge = 2% min bet, 20%+ edge = 50% max bet
if edge <= 0:
    kelly_frac = 0.02  # Minimum data collection bet
else:
    kelly_frac = min(0.50, max(0.02, edge * 3.0))
```

### Step 3 — Commit
```bash
git add kalshi-autotrader.py
git commit -m "feat: Kelly proporzionale all'edge (0%→2%, 20%+→50%)"
git push
```

---

## TASK-003: Traccia win rate per tipo di mercato

**File:** `kalshi-autotrader.py` + `kalshi-dashboard.py`

### Step 1 — Aggiungi market_type al trade log
Trova la funzione `log_trade()`:
```bash
grep -n "def log_trade" kalshi-autotrader.py
```

Dentro `log_trade()`, aggiungi campo `market_type` al JSON loggato:
```python
"market_type": classify_market_type(market),
```

### Step 2 — Aggiungi aggregazione win rate per tipo
Crea script `/Users/mattia/Projects/kalshi-autotrader/win-rate-by-type.py`:
```python
#!/usr/bin/env python3
"""Analizza win rate per tipo di mercato dai trade chiusi."""
import json
from pathlib import Path
from collections import defaultdict

state = json.loads(Path('data/trading/paper-trade-state.json').read_text())
closed = state.get('closed_trades', [])

by_type = defaultdict(lambda: {'won': 0, 'lost': 0, 'pnl': 0})
for t in closed:
    mtype = t.get('market_type', 'other')
    if t.get('won'):
        by_type[mtype]['won'] += 1
    else:
        by_type[mtype]['lost'] += 1
    by_type[mtype]['pnl'] += t.get('pnl_cents', 0) / 100

print(f"{'Tipo':<15} {'W':<6} {'L':<6} {'WR%':<8} {'PnL'}")
print('-' * 50)
for mtype, s in sorted(by_type.items()):
    total = s['won'] + s['lost']
    wr = s['won'] / total * 100 if total else 0
    print(f"{mtype:<15} {s['won']:<6} {s['lost']:<6} {wr:<8.1f} ${s['pnl']:.2f}")
```

### Step 3 — Commit
```bash
git add kalshi-autotrader.py win-rate-by-type.py
git commit -m "feat: traccia market_type nel trade log + script win rate per tipo"
git push
```

---

## DOPO OGNI TASK

1. Aggiorna `ROADMAP.md` — cambia Status da `TODO` a `DONE`
2. Commita il ROADMAP aggiornato
3. Verifica che il trader gira ancora: `launchctl list | grep kalshi-autotrader`
4. Guarda i log per 3 minuti: `tail -f /tmp/kalshi-autotrader-bootstrap.log`
5. Avvisa Ondinho che il task è done

---

## SE QUALCOSA VA MALE

```bash
# Ripristina l'ultima versione del file
cd /Users/mattia/Projects/kalshi-autotrader
git diff kalshi-autotrader.py  # Vedi cosa è cambiato
git checkout kalshi-autotrader.py  # Annulla modifiche

# Riavvia il trader
launchctl unload ~/Library/LaunchAgents/com.frh.kalshi-autotrader.plist
sleep 2
launchctl load ~/Library/LaunchAgents/com.frh.kalshi-autotrader.plist
```

---

*Guida scritta da Ondinho per Clawdinho — 2026-03-12*
