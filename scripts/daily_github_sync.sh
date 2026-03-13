#!/bin/bash
# Daily GitHub sync: check for updates, sync tuner to production, notify via Telegram
set -euo pipefail

REPO_DIR="/root/AlphaDesk"
PROD_DIR="/home/alphadesk/app"
TUNER_FILE="autoresearch/strategy_tuner.py"
LOG_FILE="/root/AlphaDesk/logs/github_sync.log"

# Telegram config
source "$REPO_DIR/.env"
TG_TOKEN="$TELEGRAM_BOT_TOKEN"
TG_CHAT="$TELEGRAM_CHAT_ID"

send_telegram() {
    curl -s -X POST "https://api.telegram.org/bot${TG_TOKEN}/sendMessage" \
        -d chat_id="$TG_CHAT" \
        -d parse_mode="Markdown" \
        -d text="$1" > /dev/null 2>&1
}

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') $1" >> "$LOG_FILE"
}

cd "$REPO_DIR"

# Fetch remote
git fetch origin 2>/dev/null

LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/main 2>/dev/null || echo "$LOCAL")

if [ "$LOCAL" = "$REMOTE" ]; then
    log "No updates on GitHub"

    # Still check if tuner has diverged (autoresearch runs locally)
    if ! diff -q "$REPO_DIR/$TUNER_FILE" "$PROD_DIR/$TUNER_FILE" > /dev/null 2>&1; then
        cp "$REPO_DIR/$TUNER_FILE" "$PROD_DIR/$TUNER_FILE"
        systemctl restart alphadesk
        log "Tuner synced to production (local autoresearch changes)"
        send_telegram "🔬 *AutoResearch Sync*
Nuovi parametri ottimizzati applicati in produzione (autoresearch locale)."
    fi
    exit 0
fi

# Pull updates
log "Updates found: $LOCAL -> $REMOTE"
git pull origin main --ff-only 2>> "$LOG_FILE"

NEW_LOCAL=$(git rev-parse HEAD)
COMMITS=$(git log --oneline "$LOCAL..$NEW_LOCAL" 2>/dev/null)
COMMIT_COUNT=$(echo "$COMMITS" | wc -l)

# Check if tuner was updated
TUNER_CHANGED=false
if git diff --name-only "$LOCAL..$NEW_LOCAL" | grep -q "$TUNER_FILE"; then
    TUNER_CHANGED=true
fi

# Sync tuner to production
cp "$REPO_DIR/$TUNER_FILE" "$PROD_DIR/$TUNER_FILE"

# Restart service
systemctl restart alphadesk
log "Production updated and restarted"

# Build Telegram message
MSG="📦 *GitHub Sync*
$COMMIT_COUNT nuovi commit pullati.

\`\`\`
$COMMITS
\`\`\`"

if [ "$TUNER_CHANGED" = true ]; then
    MSG="$MSG

🔬 *Tuner aggiornato* — nuovi parametri applicati in produzione."
fi

send_telegram "$MSG"
