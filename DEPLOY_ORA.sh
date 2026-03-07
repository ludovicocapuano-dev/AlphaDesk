#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# AlphaDesk v2 — Deploy completo in un comando
# ═══════════════════════════════════════════════════════════════
#
# Esegui dal tuo Mac nella cartella alphadesk/:
#   chmod +x DEPLOY_ORA.sh
#   ./DEPLOY_ORA.sh
#
# Ti chiederà la password del VPS UNA SOLA volta.
# Dopo fa tutto automaticamente (~5 minuti).
# ═══════════════════════════════════════════════════════════════

VPS_IP="204.168.150.74"
VPS_USER="root"
APP_DIR="/home/alphadesk/app"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
QTS_DIR="${SCRIPT_DIR}/quant_trading_system"

echo "╔══════════════════════════════════════════════╗"
echo "║  AlphaDesk v2 — Deploy Automatico            ║"
echo "║  VPS: ${VPS_IP}                               ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# Check che la cartella esista
if [ ! -d "$QTS_DIR" ]; then
    echo "❌ Cartella quant_trading_system/ non trovata!"
    echo "   Assicurati di eseguire questo script dalla cartella alphadesk/"
    exit 1
fi

# ── FASE 1: Setup VPS ──
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 FASE 1/4 — Setup VPS (Python, PyTorch, Firewall)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "⏳ Ci vogliono ~3-5 minuti..."
echo ""

scp "${SCRIPT_DIR}/SETUP_VPS.sh" ${VPS_USER}@${VPS_IP}:/tmp/setup.sh
ssh ${VPS_USER}@${VPS_IP} "chmod +x /tmp/setup.sh && /tmp/setup.sh"

# ── FASE 2: Upload file Python ──
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📤 FASE 2/4 — Upload file Python"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Create __init__.py files
ssh ${VPS_USER}@${VPS_IP} "touch ${APP_DIR}/config/__init__.py ${APP_DIR}/core/__init__.py ${APP_DIR}/strategies/__init__.py ${APP_DIR}/risk/__init__.py ${APP_DIR}/utils/__init__.py ${APP_DIR}/backtester/__init__.py"

# Main files
scp "${QTS_DIR}/main.py" "${QTS_DIR}/scheduler.py" "${QTS_DIR}/requirements.txt" \
    ${VPS_USER}@${VPS_IP}:${APP_DIR}/

# Config
scp "${QTS_DIR}/config/settings.py" "${QTS_DIR}/config/instruments.py" \
    ${VPS_USER}@${VPS_IP}:${APP_DIR}/config/

# Core
scp "${QTS_DIR}/core/etoro_client.py" "${QTS_DIR}/core/data_engine.py" \
    "${QTS_DIR}/core/regime_detector.py" "${QTS_DIR}/core/outcome_labeler.py" \
    "${QTS_DIR}/core/ml_ensemble.py" "${QTS_DIR}/core/daily_retrain.py" \
    ${VPS_USER}@${VPS_IP}:${APP_DIR}/core/

# Strategies
scp "${QTS_DIR}/strategies/base_strategy.py" "${QTS_DIR}/strategies/momentum.py" \
    "${QTS_DIR}/strategies/mean_reversion.py" "${QTS_DIR}/strategies/factor_model.py" \
    "${QTS_DIR}/strategies/fx_carry.py" \
    ${VPS_USER}@${VPS_IP}:${APP_DIR}/strategies/

# Risk
scp "${QTS_DIR}/risk/position_sizer.py" "${QTS_DIR}/risk/portfolio_risk.py" \
    ${VPS_USER}@${VPS_IP}:${APP_DIR}/risk/

# Utils
scp "${QTS_DIR}/utils/db.py" "${QTS_DIR}/utils/logger.py" "${QTS_DIR}/utils/telegram_bot.py" \
    ${VPS_USER}@${VPS_IP}:${APP_DIR}/utils/

# Backtester
scp "${QTS_DIR}/backtester/engine.py" "${QTS_DIR}/backtester/charts.py" \
    "${QTS_DIR}/backtester/run_backtest.py" "${QTS_DIR}/backtester/run_synthetic.py" \
    ${VPS_USER}@${VPS_IP}:${APP_DIR}/backtester/

# Deploy docs
scp "${QTS_DIR}/deploy/TELEGRAM_SETUP.md" ${VPS_USER}@${VPS_IP}:${APP_DIR}/deploy/ 2>/dev/null || true

# Fix ownership
ssh ${VPS_USER}@${VPS_IP} "chown -R alphadesk:alphadesk ${APP_DIR}"

echo "✅ Tutti i file caricati"

# ── FASE 3: Verifica ──
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 FASE 3/4 — Verifica installazione"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

ssh ${VPS_USER}@${VPS_IP} << 'VERIFY'
echo "📂 File presenti:"
find /home/alphadesk/app -name "*.py" | wc -l
echo " file Python trovati"
echo ""
echo "🐍 Python:"
/home/alphadesk/app/venv/bin/python --version
echo ""
echo "📦 Pacchetti chiave:"
/home/alphadesk/app/venv/bin/pip list 2>/dev/null | grep -E "torch|numpy|pandas|httpx|APScheduler|websockets"
echo ""
echo "🔑 File .env:"
if [ -f /home/alphadesk/app/.env ]; then
    echo "  ✅ Presente"
    grep -c "INSERISCI_QUI" /home/alphadesk/app/.env > /dev/null 2>&1 && echo "  ⚠️  Le API key sono ancora da configurare!" || echo "  ✅ API key configurate"
else
    echo "  ❌ Mancante!"
fi
echo ""
echo "⚙️ Servizio systemd:"
systemctl is-enabled alphadesk 2>/dev/null && echo "  ✅ Abilitato" || echo "  ℹ️  Non ancora abilitato"
VERIFY

# ── FASE 4: Istruzioni finali ──
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔑 FASE 4/4 — AZIONE RICHIESTA: Configura le chiavi"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║                                                      ║"
echo "║  🎉 DEPLOY COMPLETATO!                               ║"
echo "║                                                      ║"
echo "║  Ora fai solo queste 2 cose:                         ║"
echo "║                                                      ║"
echo "║  1. INSERISCI LE TUE API KEY:                        ║"
echo "║     ssh root@204.168.150.74                          ║"
echo "║     nano /home/alphadesk/app/.env                    ║"
echo "║     → Sostituisci INSERISCI_QUI con le tue chiavi    ║"
echo "║     → Salva: Ctrl+X → Y → Enter                     ║"
echo "║                                                      ║"
echo "║  2. AVVIA ALPHADESK:                                 ║"
echo "║     systemctl enable alphadesk                       ║"
echo "║     systemctl start alphadesk                        ║"
echo "║     journalctl -u alphadesk -f    (per i log)        ║"
echo "║                                                      ║"
echo "╚══════════════════════════════════════════════════════╝"
