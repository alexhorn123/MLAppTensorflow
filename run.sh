#!/usr/bin/env bash
set -euo pipefail

# Lightweight wrapper for training and predicting with the demo app
# Features:
#  - `./run.sh train` runs training (dotnet run -- --train)
#  - `./run.sh predict "text"` predicts the supplied text
#  - `./run.sh predict` with no args prompts interactively (no need to escape '!')

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$PROJ_DIR/MLAppTensorflow.csproj"

if [[ ! -f "$PROJ" ]]; then
  echo "Project not found at $PROJ"
  exit 2
fi

cmd=${1-}
shift || true

case "$cmd" in
  train)
    echo "Training model..."
    dotnet run --project "$PROJ" --configuration Debug -- --train
    ;;
  predict)
    # Two modes:
    # - If first argument starts with --, forward all args to the app (allows --predict-file, --model, --format, etc.)
    # - Otherwise treat args as positional texts and forward them as: --predict <text1> <text2> ...
    if [[ $# -ge 1 ]]; then
      if [[ "$1" == --* ]]; then
        echo "Forwarding flags: $*"
        dotnet run --project "$PROJ" --configuration Debug -- "$@"
      else
        echo "Predicting texts: $*"
        dotnet run --project "$PROJ" --configuration Debug -- --predict "$@"
      fi
    else
      # Interactive prompt for a single text (read -r preserves literal characters like !)
      printf 'Enter text to predict: '
      IFS= read -r text
      if [[ -z "$text" ]]; then
        echo "No text supplied. Exiting."
        exit 1
      fi
      echo "Predicting: $text"
      dotnet run --project "$PROJ" --configuration Debug -- --predict "$text"
    fi
    ;;
  ""|help|-h|--help)
    echo "Usage: $0 <command> [args]"
    echo "Commands:" 
    echo "  train             Train and evaluate model (uses data/sentiment-labelled.csv)"
    echo "  predict [text]    Predict sentiment for text. If text omitted, runs interactive prompt (no quoting needed)."
    exit 0
    ;;
  *)
    echo "Unknown command: $cmd"
    echo "Run '$0 --help' for usage"
    exit 2
    ;;
esac
