# AlphaDesk — Deep Dive: Componenti Critici

## 1. Calibrazione dei Parametri

### Perché è importante
I parametri default sono punti di partenza ragionevoli, ma ogni mercato e regime
richiede calibrazione. Un sistema non calibrato può sembrare profittevole in backtest
ma fallire in produzione (overfitting).

### Come calibrare

#### Metodo: Walk-Forward Optimization
Non ottimizzare su tutto il dataset — usa walk-forward:

```
|---- Training ----|-- Test --|
                   |---- Training ----|-- Test --|
                                      |---- Training ----|-- Test --|
```

1. Dividi i dati in finestre rolling (es. 12 mesi train, 3 mesi test)
2. Ottimizza i parametri sulla finestra di training
3. Testa sulla finestra out-of-sample
4. Media i risultati su tutte le finestre

#### Parametri chiave da calibrare per strategia:

**Momentum:**
- `breakout_period` (default 20): testa 10, 15, 20, 30, 40
- `atr_multiplier` (default 2.0): testa 1.5, 2.0, 2.5, 3.0
- `volume_threshold` (default 1.5): testa 1.2, 1.5, 2.0
- `min_momentum_3m` (default 0.05): testa 0.03, 0.05, 0.08, 0.10

**Mean Reversion:**
- `z_entry_long` (default -2.0): testa -1.5, -2.0, -2.5, -3.0
- `z_exit` (default 0.3): testa 0.0, 0.3, 0.5
- Lookback period per Z-score: testa 20, 40, 60

**FX Carry:**
- `min_carry_spread` (default 0.01): testa 0.005, 0.01, 0.015, 0.02
- `momentum_weight` vs `carry_weight`: testa diverse combinazioni
- `trend_filter_sma` (default 50): testa 20, 50, 100, 200

### Regola d'oro anti-overfitting
Se un parametro funziona solo con un valore molto specifico (es. breakout=17 giorni
funziona ma 16 e 18 no), è overfitting. I parametri robusti funzionano su un
**range** di valori, non su un singolo punto.


## 2. Edge Cases e Come Gestirli

### Flash Crash / Gap di Mercato
**Problema:** Il prezzo apre molto sotto il tuo stop loss. Lo stop-loss non ti
protegge dal gap.

**Soluzione nel codice:** Il `portfolio_risk.py` ha circuit breaker a livello
portfolio (-15% riduce, -25% chiude tutto). Questo protegge anche dai gap.

**Miglioramento consigliato:** Aggiungere un controllo su overnight gap:
```python
# In execution_engine: prima di aprire un trade
gap_pct = abs(today_open - yesterday_close) / yesterday_close
if gap_pct > 0.03:  # Gap > 3%
    logger.warning(f"Large gap detected: {gap_pct:.1%}, reducing size")
    signal.suggested_size_pct *= 0.5
```

### API Downtime
**Problema:** L'API di eToro non risponde. Le posizioni restano aperte senza
monitoraggio.

**Soluzione nel codice:** `etoro_client.py` ha retry con backoff esponenziale.
Lo scheduler ha `misfire_grace_time` per recuperare job persi.

**Miglioramento consigliato:** Aggiungere un heartbeat monitor:
```python
# Ogni 1 minuto, verifica che l'API risponda
# Se non risponde per > 5 minuti, invia alert Telegram
# Se non risponde per > 15 minuti, tenta di chiudere posizioni via fallback
```

### Slippage Elevato in Condizioni di Mercato Anomale
**Problema:** Durante eventi di mercato (earnings, FOMC, NFP), gli spread si
allargano enormemente e il tuo ordine viene eseguito a un prezzo molto diverso.

**Soluzione:** Il `position_sizer.py` include un budget per slippage. Ma in
condizioni estreme non è sufficiente.

**Miglioramento consigliato:**
```python
# Prima di eseguire un trade, controlla lo spread corrente
current_spread = ask_price - bid_price
normal_spread = instrument_metadata["avg_spread"]
if current_spread > 3 * normal_spread:
    logger.warning(f"Abnormal spread: {current_spread} vs normal {normal_spread}")
    return  # Non tradare in queste condizioni
```

### Correlation Breakdown
**Problema:** In crisi di mercato, le correlazioni convergono a 1 — tutti gli
asset scendono insieme. Il tuo portafoglio "diversificato" si comporta come
una singola posizione.

**Soluzione nel codice:** Il `portfolio_risk.py` ha limiti per esposizione correlata.

**Miglioramento consigliato:** Aggiungere un regime detector:
```python
# Calcola la correlazione media degli ultimi 20 giorni
# Se > 0.7, siamo in "risk-off regime"
# Azione: ridurre exposure totale del 50%, aumentare cash
avg_corr = correlation_matrix.values[np.triu_indices_from(
    correlation_matrix.values, k=1)].mean()
if avg_corr > 0.7:
    risk_regime = "RISK_OFF"
    # Dimezza tutte le posizioni
```


## 3. Miglioramenti Futuri (Roadmap)

### Fase 2 — Sentiment Analysis
Aggiungere un layer di NLP sui feed news per:
- Rilevare eventi market-moving prima che impattino il prezzo
- Sentiment scoring su titoli specifici
- Filtrare trades nelle ore pre/post earnings

Stack suggerito: RSS feeds → parsing → sentiment scoring con un modello leggero

### Fase 2 — Machine Learning Overlay
Usare i segnali delle 4 strategie come features per un meta-modello:
```python
# Input features:
# - Segnale momentum (score)
# - Segnale mean reversion (Z-score)
# - Factor score composito
# - FX carry score
# - Regime di mercato (VIX, yield curve)
# - Volume anomaly score
#
# Output: probabilità di trade profittevole
# Modello: gradient boosting (XGBoost/LightGBM)
```

### Fase 2 — Portfolio Optimization
Sostituire l'allocazione statica (30/20/20/30) con ottimizzazione dinamica:
- Mean-Variance di Markowitz
- Risk Parity
- Black-Litterman con le "views" generate dalle strategie

### Fase 3 — Execution Optimization
- Smart order routing: splitare ordini grandi in blocchi
- Time-weighted average price (TWAP)
- Monitoraggio della qualità di esecuzione (slippage effettivo vs stimato)

### Fase 3 — Backtester Avanzato
- Multi-strategy combinato con capital allocation dinamico
- Monte Carlo simulation per stress testing
- Walk-forward optimization automatizzata
- Transaction cost analysis (TCA) dettagliato


## 4. Checklist Pre-Produzione

Prima di passare da Demo a Real, verifica:

- [ ] **Almeno 30 giorni di Demo** senza errori di sistema
- [ ] **Backtest positivo** su almeno 2 anni di dati
- [ ] **Sharpe > 1.0** su backtest (idealmente > 1.5)
- [ ] **Max drawdown < 20%** su backtest
- [ ] **Telegram funzionante** — ricevi tutti gli alert
- [ ] **Circuit breakers testati** — simula un drawdown del 15% e del 25%
- [ ] **API rate limits OK** — nessun 429 negli ultimi 7 giorni
- [ ] **Logging funzionante** — i log ruotano correttamente
- [ ] **VPS stabile** — uptime > 99.5% nell'ultimo mese
- [ ] **Connessione internet ridondante** sul VPS
- [ ] **Inizia con il 25% del capitale** e scala gradualmente
- [ ] **Non cambiare parametri** dopo i primi 30 trade reali
   (lascia al sistema il tempo di esprimere il suo edge statistico)


## 5. Errori Comuni da Evitare

1. **Ottimizzare troppo:** più parametri calibri, più rischi di overfit
2. **Ignorare i costi:** spread + commissioni + slippage mangiano il rendimento
3. **Cambiare strategia dopo 5 trade perdenti:** le strategie quant hanno periodi di drawdown — è statisticamente normale
4. **Non avere stop loss:** il singolo errore che distrugge più conti di qualsiasi altro
5. **Leverage eccessivo:** il leverage amplifica sia guadagni che perdite
6. **Trading su news senza modello:** la reazione emotiva alle news è il contrario di ciò che fa un quant
7. **Non tenere un trade journal:** senza dati, non puoi migliorare il sistema
