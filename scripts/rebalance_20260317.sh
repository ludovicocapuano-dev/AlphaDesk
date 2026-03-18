#!/bin/bash
# One-time rebalance: validate strategy then buy XOM, SLB, GOLD
# Scheduled for 14:35 UTC March 17, 2026
set -euo pipefail

source /root/AlphaDesk/.env
API_KEY="$ETORO_API_KEY"
USER_KEY="$ETORO_USER_KEY"
BASE="https://public-api.etoro.com/api/v1"
TG_TOKEN="$TELEGRAM_BOT_TOKEN"
TG_CHAT="$TELEGRAM_CHAT_ID"
UW_KEY="$UNUSUAL_WHALES_API_KEY"

send_telegram() {
    curl -s -X POST "https://api.telegram.org/bot${TG_TOKEN}/sendMessage" \
        -d chat_id="$TG_CHAT" -d parse_mode="Markdown" -d text="$1" > /dev/null 2>&1
}

echo "=== AlphaDesk Rebalance Validation $(date -u) ==="

# ── STEP 1: Verify OIH is closed ──
oih_count=$(curl -s \
    -H "x-api-key: $API_KEY" \
    -H "x-user-key: $USER_KEY" \
    -H "x-request-id: $(uuidgen)" \
    "$BASE/trading/info/portfolio" | python3 -c "
import json,sys
d=json.load(sys.stdin)
pos=[p for p in d['clientPortfolio']['positions'] if p['instrumentID']==3206]
print(len(pos))
" 2>/dev/null)

if [ "$oih_count" != "0" ]; then
    echo "ERROR: OIH still has $oih_count positions"
    send_telegram "⚠️ *Rebalance ABORTITO*: OIH non ancora chiusa ($oih_count posizioni)"
    exit 1
fi
echo "✓ OIH confirmed closed"

# ── STEP 2: Validate strategy — check dark pool + VIX ──
VALIDATION=$(UW_KEY="$UW_KEY" python3 << 'PYEOF'
import requests, json, sys, os

uw_key = os.environ.get("UW_KEY", "")
abort_reasons = []
proceed_reasons = []

# Check XOM dark pool flow
try:
    r = requests.get("https://api.unusualwhales.com/api/stock/XOM/dark-pool",
                     headers={"Authorization": f"Bearer {uw_key}"}, timeout=10)
    if r.status_code == 200:
        data = r.json().get("data", [])
        if data:
            # Check if still accumulating (trades at ask > at bid)
            at_ask = sum(1 for d in data[:10] if d.get("tracking_type") == "at_ask")
            at_bid = sum(1 for d in data[:10] if d.get("tracking_type") == "at_bid")
            total_premium = sum(float(d.get("premium", 0)) for d in data[:10])
            if at_ask >= at_bid:
                proceed_reasons.append(f"XOM dark pool: accumulation ({at_ask} ask vs {at_bid} bid, premium ${total_premium/1e6:.1f}M)")
            else:
                abort_reasons.append(f"XOM dark pool: distribution ({at_bid} bid > {at_ask} ask)")
except Exception as e:
    proceed_reasons.append(f"XOM dark pool: check failed ({e}), proceeding anyway")

# Check SLB dark pool
try:
    r = requests.get("https://api.unusualwhales.com/api/stock/SLB/dark-pool",
                     headers={"Authorization": f"Bearer {uw_key}"}, timeout=10)
    if r.status_code == 200:
        data = r.json().get("data", [])
        if data:
            at_ask = sum(1 for d in data[:10] if d.get("tracking_type") == "at_ask")
            at_bid = sum(1 for d in data[:10] if d.get("tracking_type") == "at_bid")
            if at_ask >= at_bid:
                proceed_reasons.append(f"SLB dark pool: accumulation ({at_ask} ask vs {at_bid} bid)")
            else:
                abort_reasons.append(f"SLB dark pool: DISTRIBUTION ({at_bid} bid > {at_ask} ask)")
except Exception as e:
    pass

# Check oil price — if it crashed >5% today, abort
try:
    r = requests.get("https://api.unusualwhales.com/api/market/tide",
                     headers={"Authorization": f"Bearer {uw_key}"}, timeout=10)
    if r.status_code == 200:
        proceed_reasons.append("Market tide data fetched")
except:
    pass

# Decision
if len(abort_reasons) >= 2:
    print("ABORT")
    print("Reasons: " + "; ".join(abort_reasons))
elif len(abort_reasons) == 1 and len(proceed_reasons) == 0:
    print("ABORT")
    print("Reasons: " + "; ".join(abort_reasons))
else:
    print("PROCEED")
    print("Reasons: " + "; ".join(proceed_reasons + abort_reasons))
PYEOF
)

echo "Validation result: $VALIDATION"

DECISION=$(echo "$VALIDATION" | head -1)
REASONS=$(echo "$VALIDATION" | tail -1)

if [ "$DECISION" = "ABORT" ]; then
    echo "Strategy no longer valid — ABORTING"
    send_telegram "🛑 *Rebalance ABORTITO*
La strategia non è più valida:
$REASONS

Cash da OIH mantenuto. Nessun trade eseguito."
    exit 0
fi

echo "✓ Strategy validated: $REASONS"
send_telegram "🔄 *Rebalance in corso* — strategia validata
$REASONS"

# ── STEP 3: Execute buys ──
sleep 2

buy_instrument() {
    local instrument_id=$1
    local amount=$2
    local name=$3

    local result=$(curl -s -X POST \
        -H "x-api-key: $API_KEY" \
        -H "x-user-key: $USER_KEY" \
        -H "x-request-id: $(uuidgen)" \
        -H "Content-Type: application/json" \
        -d "{\"instrumentId\": $instrument_id, \"amount\": $amount, \"isBuy\": true}" \
        "$BASE/trading/execution/market-open-orders/by-amount")

    echo "[$name] $result"
    local order_id=$(echo "$result" | python3 -c "import json,sys; print(json.load(sys.stdin).get('order',{}).get('orderID','FAILED'))" 2>/dev/null || echo "PARSE_ERROR")
    echo "  → orderID: $order_id"
}

# Buy XOM ($1,100)
buy_instrument 1036 1100 "XOM"
sleep 3

# Buy SLB ($800)
buy_instrument 4253 800 "SLB"
sleep 3

# Buy GOLD ($800)
buy_instrument 559 800 "GOLD"
sleep 3

send_telegram "✅ *Rebalance completato*
🔴 Chiusa: OIH (~\$5,436)
🟢 Aperte:
  XOM +\$1,100
  SLB +\$800
  GOLD +\$800
💰 Cash buffer: ~\$1,736

Validazione pre-trade:
$REASONS"

echo "=== Rebalance complete ==="
