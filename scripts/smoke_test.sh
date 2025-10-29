#!/usr/bin/env bash
set -euo pipefail

# Smoke test: ensure batch JSON predictions work and output a JSON array of expected length

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PRED_FILE="data/predict-sample.txt"
WRAPPER="./run.sh"
OUT="/tmp/mlapp_preds.json"

echo "Running smoke test: batch JSON predict from $PRED_FILE"

if [[ ! -f "$WRAPPER" ]]; then
  echo "Wrapper $WRAPPER not found"
  exit 2
fi

# Ensure model exists; if not, train briefly
if [[ ! -f model.zip ]]; then
  echo "No model.zip found, training..."
  $WRAPPER train
fi

# Run batch prediction to JSON using dotnet directly (avoids wrapper stdout noise)
dotnet run --project "$ROOT_DIR/MLAppTensorflow.csproj" --configuration Debug -- --predict-file "$PRED_FILE" --format json > "$OUT"

# Count non-empty lines in the predict file to determine expected count
expected_count=$(awk 'NF' "$PRED_FILE" | wc -l | tr -d '[:space:]')

# Use python to validate JSON shape and count
python3 - <<PY
import sys, json
p = '$OUT'
with open(p,'r') as f:
    data = json.load(f)
if not isinstance(data, list):
    print('FAIL: output is not a JSON array')
    sys.exit(2)
if len(data) != int($expected_count):
    print(f'FAIL: expected {int($expected_count)} items, got {len(data)}')
    sys.exit(3)
# verify keys
required = {'text','predicted','probability','score'}
for i,item in enumerate(data):
    if not required.issubset(set(item.keys())):
        print(f'FAIL: item {i} missing keys: {required - set(item.keys())}')
        sys.exit(4)
print('OK: smoke test passed — JSON array with', len(data), 'items')
PY

echo "Smoke test completed successfully. Output saved to $OUT"
