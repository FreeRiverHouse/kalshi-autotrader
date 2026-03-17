# 📘 MOP - Method of Procedure: Kalshi AutoTrader

## 1. Obiettivo e Architettura
Il Kalshi AutoTrader è un bot algoritmico basato su Python che scansiona i mercati Kalshi. Utilizza un LLM (Grok 4.20) per estrarre insight in linguaggio naturale e quantificare un "Forecast" (probabilità) con cui poi calcola l'**Edge** (vantaggio statistico) rispetto al prezzo di mercato.

Attualmente è configurato come servizio di background persistente per macOS (tramite `launchd`), diviso in due processi:
1. **Trader**: `kalshi-autotrader.py` (Gira ogni 5 minuti).
2. **Dashboard**: `kalshi-dashboard.py` (Hostata localmente su porta `8887`).

---

## 2. Modalità di Esecuzione (Paper vs Live)

Il bot opera primariamente in **Paper Mode** (`DRY_RUN = True` nel codice) per simulare l'ambiente di mercato live prima di esporsi a rischio finanziario. 

### ⚠️ Regola d'Oro del Paper Mode
**"Il simulatore DEVE riprodurre fedelmente i vincoli del mercato reale."**

In passato, l'algo di Paper Mode soffriva della "Trappola della Liquidità Zero":
- L'algo accettava di scommettere su mercati AMM illiquidi (Volume=0, OI=0) a prezzi teorici (es. 50¢).
- Nella realtà, in assenza di Bid/Ask reali profondi, piazzare contratti al mid-price svuoterebbe l'ordine e pagherebbe il picco dell'algo, risultando in mancati fill (ordini dormienti) o *slippage* disastroso.
- **Soluzione applicata:** Il filtro liquidità (`MIN_VOLUME`, `MIN_LIQUIDITY` o controllo esplicito sui 50/50 morti) DEVE rimanere rigorosamente attivo nel codice Python durante il Paper Mode. Dobbiamo simulare trade esclusivi su mercati sui quali *effettivamente* qualcun altro sta scommettendo.

---

## 3. Parametri Chiave di Tuning (Golden Config)

Le soglie sono hardcoded in `kalshi-autotrader.py` (sezione Parametri).
L'attuale settaggio derivato dall'analisi dati su larga scala è il seguente:

| Parametro | Valore | Replicato in Dashboard | Spiegazione |
|-----------|--------|------------------------|-------------|
| `MIN_EDGE_BUY_NO` | **0.03 (3%)** | Sì | Edge minimo per assumere una posizione NO. (La calibrazione ha dimostrato che un 3% stimato dall'LLM corrisponde a un reale ~30-40% storico). |
| `MIN_EDGE_BUY_YES` | **0.08 (8%)** | Sì | Edge minimo per posizioni YES (storicamente le probabilità YES sovrastimate dall'LLM necessitano di buffer aggressivi). |
| `MAX_POSITIONS` | **30** | Sì | Esposizione simultanea massima. (Il Win Rate è >95%, autorizzando apertura parallela elevata). |
| `MAX_BET_CENTS` | **100** | Sì | Capping del singolo trade ($1). Strategia micro-betting per limitare esposizione a cigni neri. |
| `KELLY_FRACTION` | **0.15** | Sì | Mitigatore di Kelly conservativo per il position sizing. |
| `PROF_TAKE_PCT`| **0.30** | — | Chiusura automatica a profitto latente del +30%. |

---

## 4. Procedure Operative Standard (SOP)

### A. Riavvio dei Servizi (Hard Restart)
Se il trader o la dashboard impazziscono, riavviare tramite `launchctl`:
```bash
launchctl unload ~/Library/LaunchAgents/com.frh.kalshi-autotrader.plist
launchctl unload ~/Library/LaunchAgents/com.frh.kalshi-dashboard.plist
sleep 2
launchctl load ~/Library/LaunchAgents/com.frh.kalshi-autotrader.plist
launchctl load ~/Library/LaunchAgents/com.frh.kalshi-dashboard.plist
```

### B. Messa a Terra (Transcodifica a Soldi Veri)
Passare in Live (`DRY_RUN = False`) è distruttivo se non monitorato.
Azione da intraprendere alla Linea 0 dello Switch-On:
1. Impostare un conto virtuale separato (max $20-$50).
2. Ridurre `MAX_BET_CENTS` al valore minimo supportato ($0.01 se Kalshi lo supporta, altrimenti $0.10/$1.00).
3. Ridurre `MAX_POSITIONS` a 5.
4. **Monitorare il Fill Rate:** Sorvegliare per 6 ore se l'algo viene effettivamente eseguito ai prezzi indicati o se paga troppo slippage.

### C. Reset del Paper Mode
L'andamento simulato può essere resettato dalla Dashboard (`http://localhost:8887`):
1. Scrollare in fondo alla Danger Zone.
2. Premere "**🔄 Reset Paper Mode ($100)**".
*(Nota: Azzerare i dati resettando i balance NON fa dimenticare al database SQLite lo storico che l'intelligenza utilizza per allenarsi sui bias di calibrazione).*

---

## 5. Manutenzione del Database
I dati vitali del trader persistono qui:
- `data/trading/kalshi-trades.jsonl`: Dump raw Json.
- `data/trading/trading_metrics_v2.db`: Db Relazionale principale.
- `data/trading/paper-trade-state.json`: Balance fittizio in Paper Mode.

**Se `v2.db` diventa enorme**, effettuare regolarmente backup o snapshot ma non cancellarlo: l'LLM ha impellente necessità dei dati pregressi per determinare i gap di probabilità `Forecast vs Actual` necessari alle equazioni di calibrazione.
