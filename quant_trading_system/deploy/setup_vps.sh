#!/bin/bash
# AlphaDesk — VPS Setup Script
# Run this on your VPS to deploy the trading system.
# Usage: bash setup_vps.sh

set -e

echo "══════════════════════════════════════"
echo "  AlphaDesk — VPS Setup"
echo "══════════════════════════════════════"

# 1. Create project directory
PROJECT_DIR="$HOME/alphadesk"
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

echo "[1/6] Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "[2/6] Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "[3/6] Creating directories..."
mkdir -p data logs

echo "[4/6] Creating .env file (EDIT THIS!)..."
if [ ! -f .env ]; then
cat > .env << 'ENVEOF'
# eToro API Credentials
ETORO_USER_KEY=your_user_key_here
ETORO_API_KEY=your_api_key_here
ETORO_ENV=Demo

# Telegram Notifications (optional)
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
ENVEOF
echo "   ⚠️  IMPORTANT: Edit .env with your actual credentials!"
else
echo "   .env already exists, skipping"
fi

echo "[5/6] Setting up systemd service..."
# Copy service file
sudo cp deploy/alphadesk.service /etc/systemd/system/
# Update paths in service file
sudo sed -i "s|YOUR_VPS_USER|$USER|g" /etc/systemd/system/alphadesk.service
sudo systemctl daemon-reload

echo "[6/6] Setup complete!"
echo ""
echo "══════════════════════════════════════"
echo "  Next Steps:"
echo "══════════════════════════════════════"
echo ""
echo "  1. Edit your credentials:"
echo "     nano $PROJECT_DIR/.env"
echo ""
echo "  2. Test in Demo mode first:"
echo "     cd $PROJECT_DIR && source venv/bin/activate"
echo "     python main.py"
echo ""
echo "  3. Start H24 service:"
echo "     sudo systemctl enable alphadesk"
echo "     sudo systemctl start alphadesk"
echo ""
echo "  4. Monitor:"
echo "     sudo journalctl -u alphadesk -f"
echo "     tail -f $PROJECT_DIR/logs/alphadesk.log"
echo ""
echo "  ⚠️  START WITH Demo ENVIRONMENT!"
echo "  Switch to Real only after thorough testing."
echo "══════════════════════════════════════"
