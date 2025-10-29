using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;

// Enhanced ML.NET sentiment trainer + CLI inference
// Features:
// - Train/test split and evaluation metrics
// - Improved preprocessing: normalization, stopwords removal, n-grams
// - CLI: `--train` to retrain, `--predict "text"` to run a prediction

public class SentimentData
{
    [LoadColumn(0)]
    public bool Label { get; set; }

    [LoadColumn(1)]
    public string Text { get; set; } = string.Empty;
}

public class SentimentPrediction
{
    [ColumnName("PredictedLabel")]
    public bool PredictedLabel { get; set; }

    public float Probability { get; set; }
    public float Score { get; set; }
}

class Program
{
    static void PrintUsage()
    {
        Console.WriteLine("Usage:");
        Console.WriteLine("  dotnet run --project . --train              # force training and evaluation");
        Console.WriteLine("  dotnet run --project . --predict \"text\"   # load model and predict text");
        Console.WriteLine("  dotnet run --project .                      # train if no model, then run sample predictions");
    }

    static void Main(string[] args)
    {
        var mlContext = new MLContext(seed: 1);

        string dataPath = Path.Combine(Directory.GetCurrentDirectory(), "data", "sentiment-labelled.csv");
        string modelPath = Path.Combine(Directory.GetCurrentDirectory(), "model.zip");

        if (args.Length > 0)
        {
            if (args[0] == "--help" || args[0] == "-h")
            {
                PrintUsage();
                return;
            }

            if (args[0] == "--train")
            {
                if (!File.Exists(dataPath))
                {
                    Console.WriteLine($"Training data not found at {dataPath}");
                    return;
                }

                TrainEvaluateAndSave(mlContext, dataPath, modelPath);
                return;
            }

            if (args[0] == "--predict")
            {
                if (args.Length < 2)
                {
                    Console.WriteLine("Please supply text to predict: --predict \"your text\"");
                    return;
                }

                string inputText = args[1];
                if (!File.Exists(modelPath))
                {
                    Console.WriteLine("Model not found. Train a model first with --train or run without args to auto-train if data is present.");
                    return;
                }

                PredictSingle(mlContext, modelPath, inputText);
                return;
            }

            // Unknown flag
            Console.WriteLine($"Unknown argument: {args[0]}");
            PrintUsage();
            return;
        }

        // Default behavior: train if no model, then run sample predictions and evaluation
        if (!File.Exists(dataPath))
        {
            Console.WriteLine($"Sample data not found at {dataPath}");
            Console.WriteLine("Add data/sentiment-labelled.csv or run with --predict after adding a model.");
            return;
        }

        if (!File.Exists(modelPath))
        {
            Console.WriteLine("No model found — training and evaluating a new model...");
            TrainEvaluateAndSave(mlContext, dataPath, modelPath);
        }
        else
        {
            Console.WriteLine("Model found — loading and running sample predictions...");
            ITransformer loadedModel;
            using (var stream = File.OpenRead(modelPath))
            {
                loadedModel = mlContext.Model.Load(stream, out var _);
            }

            var predEngine = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(loadedModel);

            var samples = new[] {
                new SentimentData { Text = "I love this product, it works great!" },
                new SentimentData { Text = "This is the worst experience I've had." },
                new SentimentData { Text = "Not bad, could be better." }
            };

            Console.WriteLine("\nSample predictions:");
            foreach (var s in samples)
            {
                var p = predEngine.Predict(s);
                Console.WriteLine($"Text: {s.Text}");
                Console.WriteLine($"  Predicted: {p.PredictedLabel}  Probability: {p.Probability:P1}  Score: {p.Score:F4}\n");
            }
        }
    }

    static void TrainEvaluateAndSave(MLContext mlContext, string dataPath, string modelPath)
    {
        var data = mlContext.Data.LoadFromTextFile<SentimentData>(dataPath, hasHeader: true, separatorChar: ',');

        // Train/test split
        var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.2, seed: 1);
        var trainData = split.TrainSet;
        var testData = split.TestSet;

        // Preprocessing and feature extraction options
        var textOptions = new TextFeaturizingEstimator.Options
        {
            CaseMode = TextNormalizingEstimator.CaseMode.Lower,
            KeepDiacritics = false,
            KeepPunctuations = false,
            KeepNumbers = true,
            OutputTokens = false,
            StopWordsRemoverOptions = new StopWordsRemovingEstimator.Options() { Language = Microsoft.ML.Transforms.Text.StopWordsRemovingEstimator.Language.English },
            WordFeatureExtractor = new WordBagEstimator.Options { NgramLength = 2, UseAllLengths = true },
            CharFeatureExtractor = null
        };

        var pipeline = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentData.Text), options: textOptions)
            .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: nameof(SentimentData.Label), featureColumnName: "Features"));

        var model = pipeline.Fit(trainData);

        // Evaluate
        var predictions = model.Transform(testData);
        var metrics = mlContext.BinaryClassification.Evaluate(predictions, labelColumnName: nameof(SentimentData.Label));

        Console.WriteLine("Evaluation metrics (on test set):");
        Console.WriteLine($"  Accuracy  : {metrics.Accuracy:P2}");
        Console.WriteLine($"  AUC       : {metrics.AreaUnderRocCurve:P2}");
        Console.WriteLine($"  F1 Score  : {metrics.F1Score:P2}");
        Console.WriteLine($"  Precision : {metrics.PositivePrecision:P2}");
        Console.WriteLine($"  Recall    : {metrics.PositiveRecall:P2}");

        // Save the trained model to a .zip file (overwrite)
        using (var fs = File.Create(modelPath))
        {
            mlContext.Model.Save(model, trainData.Schema, fs);
        }

        Console.WriteLine($"Model trained and saved to {modelPath}");
    }

    static void PredictSingle(MLContext mlContext, string modelPath, string text)
    {
        ITransformer loadedModel;
        using (var stream = File.OpenRead(modelPath))
        {
            loadedModel = mlContext.Model.Load(stream, out var _);
        }

        var predEngine = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(loadedModel);
        var input = new SentimentData { Text = text };
        var p = predEngine.Predict(input);
        Console.WriteLine($"Text: {text}");
        Console.WriteLine($"  Predicted: {p.PredictedLabel}  Probability: {p.Probability:P1}  Score: {p.Score:F4}");
    }
}
