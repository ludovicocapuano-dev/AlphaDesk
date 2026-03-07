#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# AlphaDesk v2 — Upload files to VPS
# ═══════════════════════════════════════════════════════════════
#
# Esegui questo script DAL TUO MAC, non dal VPS!
#
# USO:
#   chmod +x UPLOAD_TO_VPS.sh
#   ./UPLOAD_TO_VPS.sh IL_TUO_IP_VPS
#
# Esempio:
#   ./UPLOAD_TO_VPS.sh 95.216.123.45
#
# ═══════════════════════════════════════════════════════════════

if [ -z "$1" ]; then
    echo "❌ Uso: ./UPLOAD_TO_VPS.sh <IP_DEL_VPS>"
    echo "   Esempio: ./UPLOAD_TO_VPS.sh 95.216.123.45"
    exit 1
fi

VPS_IP="$1"
VPS_USER="root"
APP_DIR="/home/alphadesk/app"
LOCAL_DIR="$(dirname "$0")/quant_trading_system"

if [ ! -d "$LOCAL_DIR" ]; then
    echo "❌ Cartella non trovata: $LOCAL_DIR"
    echo "   Assicurati che questo script sia nella stessa cartella di 'quant_trading_system/'"
    exit 1
fi

echo "╔══════════════════════════════════════════════╗"
echo "║  Uploading AlphaDesk v2 to VPS $VPS_IP"
echo "╚══════════════════════════════════════════════╝"
echo ""

# Upload Python files
echo "📤 Caricamento file Python..."

# Core files
scp "$LOCAL_DIR/main.py" "$LOCAL_DIR/scheduler.py" "$LOCAL_DIR/requirements.txt" \
    ${VPS_USER}@${VPS_IP}:${APP_DIR}/

# Config
scp "$LOCAL_DIR/config/settings.py" "$LOCAL_DIR/config/instruments.py" \
    ${VPS_USER}@${VPS_IP}:${APP_DIR}/config/

# Create __init__.py files for Python imports
ssh ${VPS_USER}@${VPS_IP} "touch ${APP_DIR}/config/__init__.py ${APP_DIR}/core/__init__.py ${APP_DIR}/strategies/__init__.py ${APP_DIR}/risk/__init__.py ${APP_DIR}/utils/__init__.py ${APP_DIR}/backtester/__init__.py"

# Core modules
scp "$LOCAL_DIR/core/etoro_client.py" "$LOCAL_DIR/core/data_engine.py" \
    "$LOCAL_DIR/core/regime_detector.py" "$LOCAL_DIR/core/outcome_labeler.py" \
    "$LOCAL_DIR/core/ml_ensemble.py" "$LOCAL_DIR/core/daily_retrain.py" \
    ${VPS_USER}@${VPS_IP}:${APP_DIR}/core/

# Strategies
scp "$LOCAL_DIR/strategies/base_strategy.py" "$LOCAL_DIR/strategies/momentum.py" \
    "$LOCAL_DIR/strategies/mean_reversion.py" "$LOCAL_DIR/strategies/factor_model.py" \
    "$LOCAL_DIR/strategies/fx_carry.py" \
    ${VPS_USER}@${VPS_IP}:${APP_DIR}/strategies/

# Risk
scp "$LOCAL_DIR/risk/position_sizer.py" "$LOCAL_DIR/risk/portfolio_risk.py" \
    ${VPS_USER}@${VPS_IP}:${APP_DIR}/risk/

# Utils
scp "$LOCAL_DIR/utils/db.py" "$LOCAL_DIR/utils/logger.py" "$LOCAL_DIR/utils/telegram_bot.py" \
    ${VPS_USER}@${VPS_IP}:${APP_DIR}/utils/

# Backtester
scp "$LOCAL_DIR/backtester/engine.py" "$LOCAL_DIR/backtester/charts.py" \
    "$LOCAL_DIR/backtester/run_backtest.py" "$LOCAL_DIR/backtester/run_synthetic.py" \
    ${VPS_USER}@${VPS_IP}:${APP_DIR}/backtester/

# Deploy docs
scp "$LOCAL_DIR/deploy/TELEGRAM_SETUP.md" \
    ${VPS_USER}@${VPS_IP}:${APP_DIR}/deploy/ 2>/dev/null || true

# Fix permissions
ssh ${VPS_USER}@${VPS_IP} "chown -R alphadesk:alphadesk ${APP_DIR}"

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║  ✅ Upload completato!                       ║"
echo "╠══════════════════════════════════════════════╣"
echo "║                                              ║"
echo "║  Prossimi passi sul VPS:                     ║"
echo "║                                              ║"
echo "║  1. ssh root@${VPS_IP}                       ║"
echo "║  2. nano /home/alphadesk/app/.env            ║"
echo "║     → inserisci le tue API key               ║"
echo "║  3. systemctl enable alphadesk               ║"
echo "║  4. systemctl start alphadesk                ║"
echo "║  5. journalctl -u alphadesk -f               ║"
echo "║                                              ║"
echo "╚══════════════════════════════════════════════╝"
