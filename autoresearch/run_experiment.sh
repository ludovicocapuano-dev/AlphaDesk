#!/bin/bash
# AutoResearch — Run a single experiment
# Usage: bash autoresearch/run_experiment.sh <strategy_name> <experiment_id>
#
# Example:
#   bash autoresearch/run_experiment.sh momentum exp_mom_001
#   bash autoresearch/run_experiment.sh factor_model exp_fm_001

set -e

STRATEGY=${1:?"Usage: $0 <strategy> <experiment_id>"}
EXP_ID=${2:?"Usage: $0 <strategy> <experiment_id>"}

cd "$(dirname "$0")/.."

# Activate venv
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

export PYTHONPATH="$(pwd)"

echo "════════════════════════════════════════════════════"
echo "  AutoResearch Experiment: $EXP_ID"
echo "  Strategy: $STRATEGY"
echo "  Time: $(date -u '+%Y-%m-%d %H:%M UTC')"
echo "════════════════════════════════════════════════════"

python3 autoresearch/backtest_runner.py \
    --strategy "$STRATEGY" \
    --experiment-id "$EXP_ID" \
    --timeout 300

EXIT_CODE=$?

echo ""
echo "════════════════════════════════════════════════════"
if [ $EXIT_CODE -eq 0 ]; then
    echo "  Experiment $EXP_ID completed successfully"
else
    echo "  Experiment $EXP_ID FAILED (exit code: $EXIT_CODE)"
fi
echo "════════════════════════════════════════════════════"

exit $EXIT_CODE
