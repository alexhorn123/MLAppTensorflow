MLAppTensorflow — small ML.NET sentiment demo

[![CI](https://github.com/alexhorn123/MLAppTensorflow/actions/workflows/ci.yml/badge.svg)](https://github.com/alexhorn123/MLAppTensorflow/actions/workflows/ci.yml)

This repository currently contains a minimal ML.NET example that:

- Uses a tiny CSV dataset (`data/sentiment-labelled.csv`) to train a sentiment analysis model.
- Saves the trained model to `model.zip`.
- Loads the trained model and runs sample predictions via a small CLI.

How to run

Quick start:

1. Build the project (do this once or after changes):
```bash
# Build
dotnet build -c Debug
```

2. Train a model:
```bash
# Train and save model.zip
./run.sh train
```

3. Make predictions:
```bash
# Interactive mode (best for text with special chars)
./run.sh predict
# Then type or paste your text when prompted

# Direct mode
./run.sh predict "I love this product!"

# Batch predict from file
./run.sh predict --predict-file data/predict-sample.txt --format json
```

The first run will train a model if none exists. The model is saved as `model.zip` and reused for future predictions.

Development

- Build and test workflow runs on push/PR to main branch.
- Smoke test verifies batch JSON predictions work correctly.

Notes

- The project uses `Microsoft.ML` to train a simple SDCA logistic regression model over text features.
- This is a lightweight demo to get started. For production use consider larger datasets, proper train/validation splits, preprocessing, and evaluation metrics.

Wrapper script

To avoid needing to escape special characters (like `!`) in the shell when passing text to the CLI, a small wrapper script is provided:

```bash
# from the project root (make executable first if needed)
chmod +x run.sh

# Train and evaluate
./run.sh train

# Predict with an argument (still needs normal quoting for spaces), or run interactively to avoid quoting issues:
./run.sh predict 'I love this!'

# Or interactive (no quoting required):
./run.sh predict
# then paste/type: I love this!
```

The interactive prompt reads text with `read -r` so `!` and other characters are accepted literally.


More examples

You can pass multiple texts on the command line (they'll all be predicted):

```bash
# positional texts (convenience - wrapper will add the --predict flag)
./run.sh predict "I love this" "Not great at all"

# or pass flags first (forwarded directly to the app) and request JSON output
./run.sh predict --format json "I love this" "Not great at all"

# Predict from a file (one text per line) and get JSON output
./run.sh predict --predict-file data/predict-sample.txt --format json

# Specify a different model file
./run.sh predict --model path/to/other-model.zip --predict "Some text"
```

The `run.sh` wrapper forwards flags to the application when a flag is the first argument after `predict`, and treats positional arguments as texts to predict otherwise. This keeps interactive usage simple while allowing full flexibility for batch and JSON output workflows.

Understanding predictions

The model predicts sentiment as positive (true) or negative (false). Output formats:

```bash
# Text output format (default)
Text: I love this product!
  Predicted: True  Probability: 95.7%  Score: 3.0998

# JSON output format (when using --format json)
[
  {
    "text": "I love this product!",
    "predicted": true,
    "probability": 0.9578221,
    "score": 3.122766
  }
]
```

Output fields:
- `predicted`: true = positive sentiment, false = negative sentiment
- `probability`: confidence score between 0 and 1 (higher = more confident)
- `score`: raw model score (higher numbers indicate more positive sentiment)

Files:
- `data/sentiment-labelled.csv`: training data (one text and label per line)
- `model.zip`: trained model file (created after running train)
- `data/predict-sample.txt`: example texts for batch prediction

