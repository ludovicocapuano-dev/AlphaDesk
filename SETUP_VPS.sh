#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# AlphaDesk v2 — VPS One-Shot Setup Script
# ═══════════════════════════════════════════════════════════════
#
# ISTRUZIONI:
# 1. Accedi al tuo VPS dalla console web Hetzner
#    (Cloud Console → il tuo server → "Console" in alto a destra)
# 2. Copia e incolla questo intero script nella console
# 3. Alla fine, modifica il file .env con le tue chiavi
#
# Questo script:
# ✅ Aggiorna il sistema
# ✅ Installa Python 3.11, pip, venv
# ✅ Crea l'utente 'alphadesk' dedicato
# ✅ Configura la struttura delle cartelle
# ✅ Crea l'ambiente virtuale Python
# ✅ Installa tutte le dipendenze (incluso PyTorch CPU)
# ✅ Configura il servizio systemd per H24
# ✅ Configura SSH con chiave (opzionale)
# ✅ Imposta il firewall base
# ✅ Crea il file .env template
#
# ═══════════════════════════════════════════════════════════════

set -e  # Exit on error

echo "╔══════════════════════════════════════════════╗"
echo "║     AlphaDesk v2 — VPS Setup Starting        ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# ── 1. System Update ──
echo "📦 [1/8] Aggiornamento sistema..."
apt update && apt upgrade -y
apt install -y python3.11 python3.11-venv python3.11-dev python3-pip \
    git curl wget unzip htop tmux ufw fail2ban sqlite3

# ── 2. Create dedicated user ──
echo "👤 [2/8] Creazione utente alphadesk..."
if ! id "alphadesk" &>/dev/null; then
    useradd -m -s /bin/bash alphadesk
    echo "Utente 'alphadesk' creato"
else
    echo "Utente 'alphadesk' già esistente"
fi

# ── 3. Create directory structure ──
echo "📁 [3/8] Struttura cartelle..."
mkdir -p /home/alphadesk/app/{config,core,strategies,risk,utils,backtester,deploy,reports,data,logs,models}
chown -R alphadesk:alphadesk /home/alphadesk/app

# ── 4. Python virtual environment ──
echo "🐍 [4/8] Ambiente virtuale Python..."
sudo -u alphadesk python3.11 -m venv /home/alphadesk/app/venv
cat > /tmp/install_deps.sh << 'DEPS'
#!/bin/bash
source /home/alphadesk/app/venv/bin/activate
pip install --upgrade pip

# Core dependencies
pip install httpx websockets numpy pandas scipy APScheduler statsmodels yfinance pandas-datareader

# PyTorch CPU-only (molto più leggero, ~200MB vs 2GB)
pip install torch --index-url https://download.pytorch.org/whl/cpu

echo "✅ Tutte le dipendenze installate"
pip list | grep -E "torch|numpy|pandas|httpx"
DEPS
chmod +x /tmp/install_deps.sh
sudo -u alphadesk bash /tmp/install_deps.sh

# ── 5. Create .env template ──
echo "🔑 [5/8] Template .env..."
cat > /home/alphadesk/app/.env << 'ENV'
# ═══════════════════════════════════════════
# AlphaDesk v2 — Configurazione
# ═══════════════════════════════════════════
# MODIFICA QUESTI VALORI CON LE TUE CHIAVI!
# ═══════════════════════════════════════════

# ── eToro API ──
ETORO_USER_KEY=INSERISCI_QUI_LA_TUA_USER_KEY
ETORO_API_KEY=INSERISCI_QUI_LA_TUA_API_KEY

# Ambiente: "Demo" per test, "Real" per trading reale
# ⚠️ PARTI SEMPRE IN DEMO PER ALMENO 2-3 MESI!
ETORO_ENV=Demo

# ── Telegram (opzionale) ──
# Segui deploy/TELEGRAM_SETUP.md per configurare
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
ENV
chown alphadesk:alphadesk /home/alphadesk/app/.env
chmod 600 /home/alphadesk/app/.env  # Solo alphadesk può leggere

# ── 6. Systemd service ──
echo "⚙️ [6/8] Servizio systemd..."
cat > /etc/systemd/system/alphadesk.service << 'SERVICE'
[Unit]
Description=AlphaDesk v2 Quantitative Trading System
After=network.target
Wants=network-online.target

[Service]
Type=simple
User=alphadesk
Group=alphadesk
WorkingDirectory=/home/alphadesk/app
EnvironmentFile=/home/alphadesk/app/.env
ExecStart=/home/alphadesk/app/venv/bin/python scheduler.py
Restart=always
RestartSec=30
StartLimitBurst=5
StartLimitIntervalSec=300

# Logging
StandardOutput=journal
StandardError=journal
SyslogIdentifier=alphadesk

# Security hardening
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=/home/alphadesk/app/data /home/alphadesk/app/logs /home/alphadesk/app/models
PrivateTmp=yes

[Install]
WantedBy=multi-user.target
SERVICE

systemctl daemon-reload

# ── 7. Firewall ──
echo "🛡️ [7/8] Firewall..."
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
ufw --force enable
echo "Firewall attivo: solo SSH in entrata"

# ── 8. Fail2ban ──
echo "🔒 [8/8] Fail2ban..."
systemctl enable fail2ban
systemctl start fail2ban

# ── Summary ──
echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║     ✅ SETUP COMPLETATO!                             ║"
echo "╠══════════════════════════════════════════════════════╣"
echo "║                                                      ║"
echo "║  Ora devi fare solo 2 cose:                          ║"
echo "║                                                      ║"
echo "║  1. CARICARE I FILE del progetto:                    ║"
echo "║     I file Python vanno in /home/alphadesk/app/      ║"
echo "║     (vedi istruzioni sotto)                          ║"
echo "║                                                      ║"
echo "║  2. CONFIGURARE LE CHIAVI:                           ║"
echo "║     nano /home/alphadesk/app/.env                    ║"
echo "║     → Inserisci ETORO_USER_KEY e ETORO_API_KEY       ║"
echo "║     → Salva con Ctrl+X, Y, Enter                    ║"
echo "║                                                      ║"
echo "║  Poi avvia con:                                      ║"
echo "║     systemctl enable alphadesk                       ║"
echo "║     systemctl start alphadesk                        ║"
echo "║     journalctl -u alphadesk -f  (per i log)          ║"
echo "║                                                      ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "📂 Struttura creata:"
ls -la /home/alphadesk/app/
echo ""
echo "🐍 Python:"
/home/alphadesk/app/venv/bin/python --version
echo ""
echo "📦 Pacchetti installati:"
/home/alphadesk/app/venv/bin/pip list 2>/dev/null | grep -E "torch|numpy|pandas|httpx|APScheduler" || true
