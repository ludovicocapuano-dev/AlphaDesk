#!/bin/bash
# ──────────────────────────────────────────────────────────────
# AlphaDesk — Monitor ai-hedge-fund repo for agent updates
# Checks for new commits in src/agents/ and notifies via Telegram
# ──────────────────────────────────────────────────────────────

set -euo pipefail

REPO_DIR="/root/ai-hedge-fund"
ALPHADESK_DIR="/root/AlphaDesk"
STATE_FILE="$ALPHADESK_DIR/data/agent_update_state.txt"

# Load Telegram credentials from .env
TELEGRAM_BOT_TOKEN=""
TELEGRAM_CHAT_ID=""
while IFS='=' read -r key value; do
    key=$(echo "$key" | xargs)
    value=$(echo "$value" | xargs)
    case "$key" in
        TELEGRAM_BOT_TOKEN) TELEGRAM_BOT_TOKEN="$value" ;;
        TELEGRAM_CHAT_ID) TELEGRAM_CHAT_ID="$value" ;;
    esac
done < "$ALPHADESK_DIR/.env"

send_telegram() {
    local msg="$1"
    if [[ -n "$TELEGRAM_BOT_TOKEN" && -n "$TELEGRAM_CHAT_ID" ]]; then
        curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
            -d chat_id="$TELEGRAM_CHAT_ID" \
            -d text="$msg" \
            -d parse_mode="HTML" > /dev/null 2>&1
    fi
}

# Ensure state dir exists
mkdir -p "$(dirname "$STATE_FILE")"

# Get current local HEAD
cd "$REPO_DIR"
OLD_HEAD=$(cat "$STATE_FILE" 2>/dev/null || git rev-parse HEAD)
OLD_AGENTS=$(ls src/agents/*.py 2>/dev/null | sort)

# Fetch latest
git fetch origin main --quiet 2>/dev/null || git fetch origin master --quiet 2>/dev/null

# Check for new commits
REMOTE_HEAD=$(git rev-parse FETCH_HEAD 2>/dev/null || echo "$OLD_HEAD")

if [[ "$OLD_HEAD" == "$REMOTE_HEAD" ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] No updates."
    exit 0
fi

# There are updates — pull them
git merge FETCH_HEAD --ff-only --quiet 2>/dev/null || {
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Merge conflict — manual intervention needed."
    send_telegram "⚠️ <b>ai-hedge-fund</b>: merge conflict. Manual pull needed."
    exit 1
}

NEW_HEAD=$(git rev-parse HEAD)

# Check what changed in agents/
AGENT_CHANGES=$(git diff --name-only "$OLD_HEAD" "$NEW_HEAD" -- src/agents/ 2>/dev/null || echo "")
NEW_AGENTS=$(ls src/agents/*.py 2>/dev/null | sort)

# Detect new agent files
NEW_AGENT_FILES=$(comm -13 <(echo "$OLD_AGENTS") <(echo "$NEW_AGENTS") 2>/dev/null || echo "")

# Count changes
NUM_CHANGED=$(echo "$AGENT_CHANGES" | grep -c '.' 2>/dev/null || echo "0")
NUM_NEW=$(echo "$NEW_AGENT_FILES" | grep -c '.' 2>/dev/null || echo "0")

# Get commit summary
COMMIT_LOG=$(git log --oneline "$OLD_HEAD..$NEW_HEAD" -- src/agents/ 2>/dev/null | head -10)

# Save new state
echo "$NEW_HEAD" > "$STATE_FILE"

if [[ "$NUM_CHANGED" -eq 0 && "$NUM_NEW" -eq 0 ]]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Repo updated ($OLD_HEAD → $NEW_HEAD) but no agent changes."
    exit 0
fi

# Build notification
MSG="🔔 <b>AI HEDGE FUND — AGENT UPDATE</b>\n"
MSG+="━━━━━━━━━━━━━━━━━━━━\n"
MSG+="📦 Repo: virattt/ai-hedge-fund\n"

if [[ "$NUM_NEW" -gt 0 ]]; then
    MSG+="\n🆕 <b>New agents ($NUM_NEW):</b>\n"
    while IFS= read -r f; do
        [[ -z "$f" ]] && continue
        basename=$(basename "$f" .py)
        MSG+="  • $basename\n"
    done <<< "$NEW_AGENT_FILES"
fi

if [[ "$NUM_CHANGED" -gt 0 ]]; then
    MSG+="\n📝 <b>Changed files ($NUM_CHANGED):</b>\n"
    while IFS= read -r f; do
        [[ -z "$f" ]] && continue
        MSG+="  • $(basename "$f")\n"
    done <<< "$AGENT_CHANGES"
fi

if [[ -n "$COMMIT_LOG" ]]; then
    MSG+="\n📋 <b>Commits:</b>\n"
    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        MSG+="  $line\n"
    done <<< "$COMMIT_LOG"
fi

MSG+="\n⚡ Controlla se serve aggiornare AlphaDesk!"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Agent updates detected!"
echo "$AGENT_CHANGES"

# Send Telegram notification
send_telegram "$MSG"

echo "Telegram notification sent."
