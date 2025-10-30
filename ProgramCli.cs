using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;

// Enhanced ML.NET sentiment trainer + CLI inference
// Same behavior as requested, in a new clean file (ProgramCli.cs)

public class SentimentDataCli
{
    [LoadColumn(0)]
    public bool Label { get; set; }

    [LoadColumn(1)]
    public string Text { get; set; } = string.Empty;
}

public class SentimentPredictionCli
{
    [ColumnName("PredictedLabel")]
    public bool PredictedLabel { get; set; }

    public float Probability { get; set; }
    public float Score { get; set; }
}

class ProgramCli
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

        // Default data and model paths
        string dataPath = Path.Combine(Directory.GetCurrentDirectory(), "data", "sentiment-labelled.csv");
        string defaultModelPath = Path.Combine(Directory.GetCurrentDirectory(), "model.zip");

        // Basic args parsing (supports multiple texts, predict-file, custom model, and output format)
        bool wantTrain = false;
        string modelPath = defaultModelPath;
        string format = "text"; // or json
    string? predictFile = null;
        var predictTexts = new System.Collections.Generic.List<string>();

        for (int i = 0; i < args.Length; i++)
        {
            var a = args[i];
            if (a == "--help" || a == "-h") { PrintUsage(); return; }
            if (a == "--train") { wantTrain = true; continue; }
            if (a == "--model") { if (i + 1 < args.Length) { modelPath = args[++i]; continue; } else { Console.WriteLine("--model requires a path"); return; } }
            if (a == "--format") { if (i + 1 < args.Length) { format = args[++i].ToLowerInvariant(); continue; } else { Console.WriteLine("--format requires 'text' or 'json'"); return; } }
            if (a == "--predict-file") { if (i + 1 < args.Length) { predictFile = args[++i]; continue; } else { Console.WriteLine("--predict-file requires a path"); return; } }
            if (a == "--predict")
            {
                // collect the rest of arguments until next flag or end
                i++;
                while (i < args.Length && !args[i].StartsWith("--"))
                {
                    predictTexts.Add(args[i]);
                    i++;
                }
                i--;
                continue;
            }
            // Unknown arg: treat as a text to predict (convenience)
            predictTexts.Add(a);
        }

        if (wantTrain)
        {
            if (!File.Exists(dataPath)) { Console.WriteLine($"Training data not found at {dataPath}"); return; }
            TrainEvaluateAndSave(mlContext, dataPath, modelPath);
        }

        // If predict-file provided, read lines
        if (!string.IsNullOrEmpty(predictFile))
        {
            if (!File.Exists(predictFile)) { Console.WriteLine($"Predict file not found: {predictFile}"); return; }
            foreach (var line in File.ReadAllLines(predictFile))
            {
                var t = line.Trim();
                if (!string.IsNullOrEmpty(t)) predictTexts.Add(t);
            }
        }

        if (predictTexts.Count == 0)
        {
            // If no predictions requested, default behavior: train if no model, then run sample predictions
            if (!File.Exists(modelPath))
            {
                if (!File.Exists(dataPath)) { Console.WriteLine($"Sample data not found at {dataPath}"); Console.WriteLine("Add data/sentiment-labelled.csv or use --predict-file"); return; }
                Console.WriteLine("No model found — training and evaluating a new model...");
                TrainEvaluateAndSave(mlContext, dataPath, modelPath);
            }

            // load model and show sample
            ITransformer loadedModel;
            using (var stream = File.OpenRead(modelPath)) loadedModel = mlContext.Model.Load(stream, out var _);
            var predEngine = mlContext.Model.CreatePredictionEngine<SentimentDataCli, SentimentPredictionCli>(loadedModel);
            var samples = new[] {
                new SentimentDataCli { Text = "I love this product, it works great!" },
                new SentimentDataCli { Text = "This is the worst experience I've had." },
                new SentimentDataCli { Text = "Not bad, could be better." }
            };
            Console.WriteLine("\nSample predictions:");
            foreach (var s in samples)
            {
                var p = predEngine.Predict(s);
                Console.WriteLine($"Text: {s.Text}");
                Console.WriteLine($"  Predicted: {p.PredictedLabel}  Probability: {p.Probability:P1}  Score: {p.Score:F4}\n");
            }
            return;
        }

        // Load model
        if (!File.Exists(modelPath)) { Console.WriteLine($"Model not found at {modelPath}. Train first."); return; }
    ITransformer? loaded = null;
        using (var stream = File.OpenRead(modelPath)) loaded = mlContext.Model.Load(stream, out var _);

        var engine = mlContext.Model.CreatePredictionEngine<SentimentDataCli, SentimentPredictionCli>(loaded);

        // Prepare outputs
        var results = new System.Collections.Generic.List<object>();
        foreach (var txt in predictTexts)
        {
            var p = engine.Predict(new SentimentDataCli { Text = txt });
            if (format == "json")
            {
                results.Add(new { text = txt, predicted = p.PredictedLabel, probability = p.Probability, score = p.Score });
            }
            else
            {
                Console.WriteLine($"Text: {txt}");
                Console.WriteLine($"  Predicted: {p.PredictedLabel}  Probability: {p.Probability:P1}  Score: {p.Score:F4}\n");
            }
        }

        if (format == "json")
        {
            var options = new System.Text.Json.JsonSerializerOptions { WriteIndented = true };
            Console.WriteLine(System.Text.Json.JsonSerializer.Serialize(results, options));
        }
    }

    static void TrainEvaluateAndSave(MLContext mlContext, string dataPath, string modelPath)
    {
        var data = mlContext.Data.LoadFromTextFile<SentimentDataCli>(dataPath, hasHeader: true, separatorChar: ',');

        var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.2, seed: 1);
        var trainData = split.TrainSet;
        var testData = split.TestSet;

        // Use FeaturizeText with controlled options and disable the char extractor to avoid
        // variable-size char feature vectors that can cause schema mismatches on the trainer.
        // Use hashed n-grams so the output 'Features' is a fixed-size vector (2^numberOfBits).
        // This avoids schema issues caused by variable-length vectors while keeping n-gram features.
        var pipeline = mlContext.Transforms.Text.TokenizeIntoWords("Tokens", nameof(SentimentDataCli.Text))
            .Append(mlContext.Transforms.Text.RemoveDefaultStopWords("Tokens", "Tokens"))
            .Append(mlContext.Transforms.Text.ProduceHashedNgrams("Features", "Tokens", numberOfBits: 14, ngramLength: 2, useAllLengths: true))
            .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: nameof(SentimentDataCli.Label), featureColumnName: "Features"));

        var model = pipeline.Fit(trainData);

        var predictions = model.Transform(testData);

        // Check label distribution in the test set; for tiny datasets the test split
        // may contain only a single class which makes some metrics (AUC) undefined.
        var testLabels = mlContext.Data.CreateEnumerable<SentimentDataCli>(testData, reuseRowObject: false);
        int pos = 0, neg = 0, total = 0;
        foreach (var row in testLabels)
        {
            if (row.Label) pos++; else neg++;
            total++;
        }

        if (pos > 0 && neg > 0)
        {
            var metrics = mlContext.BinaryClassification.Evaluate(predictions, labelColumnName: nameof(SentimentDataCli.Label));
            Console.WriteLine("Evaluation metrics (on test set):");
            Console.WriteLine($"  Accuracy  : {metrics.Accuracy:P2}");
            Console.WriteLine($"  AUC       : {metrics.AreaUnderRocCurve:P2}");
            Console.WriteLine($"  F1 Score  : {metrics.F1Score:P2}");
            Console.WriteLine($"  Precision : {metrics.PositivePrecision:P2}");
            Console.WriteLine($"  Recall    : {metrics.PositiveRecall:P2}");
        }
        else
        {
            Console.WriteLine("Skipping detailed evaluation because test set does not contain both classes.");
            Console.WriteLine($"Test set size: {total}  Positives: {pos}  Negatives: {neg}");
            // As a fallback compute simple accuracy if possible
            try
            {
                var scored = mlContext.Data.CreateEnumerable<SentimentPredictionCli>(predictions, reuseRowObject: false);
                int correct = 0, count = 0;
                var actual = mlContext.Data.CreateEnumerable<SentimentDataCli>(testData, reuseRowObject: false);
                using (var aEnum = actual.GetEnumerator())
                using (var sEnum = scored.GetEnumerator())
                {
                    while (aEnum.MoveNext() && sEnum.MoveNext())
                    {
                        if (aEnum.Current.Label == sEnum.Current.PredictedLabel) correct++;
                        count++;
                    }
                }
                if (count > 0)
                    Console.WriteLine($"  Simple accuracy: {(double)correct / count:P2}");
            }
            catch
            {
                // ignore fallback errors
            }
        }

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

        var predEngine = mlContext.Model.CreatePredictionEngine<SentimentDataCli, SentimentPredictionCli>(loadedModel);
        var input = new SentimentDataCli { Text = text };
        var p = predEngine.Predict(input);
        Console.WriteLine($"Text: {text}");
        Console.WriteLine($"  Predicted: {p.PredictedLabel}  Probability: {p.Probability:P1}  Score: {p.Score:F4}");
    }
}
