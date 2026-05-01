using Microsoft.Data.SqlClient;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;
using System;
using System.Globalization;
using System.IO;

namespace QuickCommerceMLPOC
{
    // Input schema matching your Excel/CSV/TSV columns
    public class OrderData
    {
        [LoadColumn(0)] public string Order_ID { get; set; }
        [LoadColumn(1)] public string Company { get; set; }
        [LoadColumn(2)] public string City { get; set; }
        [LoadColumn(3)] public float Customer_Age { get; set; }
        [LoadColumn(4)] public float Order_Value { get; set; }
        [LoadColumn(5)] public float Delivery_Time_Min { get; set; }   // label
        [LoadColumn(6)] public float Distance_Km { get; set; }
        [LoadColumn(7)] public float Items_Count { get; set; }
        [LoadColumn(8)] public string Product_Category { get; set; }
        [LoadColumn(9)] public string Payment_Method { get; set; }
        [LoadColumn(10)] public float Customer_Rating { get; set; }
        [LoadColumn(11)] public float Discount_Applied { get; set; }
        [LoadColumn(12)] public float Delivery_Partner_Rating { get; set; }
    }

    public class DeliveryTimePrediction
    {
        [ColumnName("Score")]
        public float PredictedDeliveryMinutes { get; set; }
    }

    internal class Program
    {
        private const string LabelColumnName = nameof(OrderData.Delivery_Time_Min);
        private const string ConnectionString =
            "Server=(localdb)\\MSSQLLocalDB;Database=MLDatabase;Trusted_Connection=True;MultipleActiveResultSets=True;Encrypt=False";

        // If your file is saved from Excel as TSV, keep '\t'
        // If it is saved as CSV, change this to ','
        private const char DataSeparator = '\t';

        private static readonly string DataPath =
            @"E:\QuickCommerceMLPOC\QuickCommerceMLPOC\Data\quick_commerce_data_modified_cleaned.csv";

        private const string ModelName = "DeliveryTimeModel";

        static void Main(string[] args)
        {
            EnsureModelTableExists();
            var mlContext = new MLContext(seed: 123);

            if (ModelExistsInDatabase())
            {
                Console.WriteLine("Model already exists in database.");
                Console.Write("Do you want to retrain the model? (yes/no): ");
                var response = Console.ReadLine()?.Trim().ToLowerInvariant();

                if (response == "no")
                {
                    UseExistingModel(mlContext);
                    return;
                }
            }

            TrainModel(mlContext);
        }
        private static List<(string Name, IEstimator<ITransformer> Trainer)> GetTunedTrainers(MLContext mlContext)
        {
            var trainers = new List<(string Name, IEstimator<ITransformer> Trainer)>();

            // LightGBM tuning
            var lightGbmOptionsList = new[]
            {
        new Microsoft.ML.Trainers.LightGbm.LightGbmRegressionTrainer.Options
        {
            NumberOfLeaves = 20,
            NumberOfIterations = 100,
            LearningRate = 0.1
        },
        new Microsoft.ML.Trainers.LightGbm.LightGbmRegressionTrainer.Options
        {
            NumberOfLeaves = 50,
            NumberOfIterations = 200,
            LearningRate = 0.05
        },
        new Microsoft.ML.Trainers.LightGbm.LightGbmRegressionTrainer.Options
        {
            NumberOfLeaves = 100,
            NumberOfIterations = 300,
            LearningRate = 0.03
        }
    };

            foreach (var opt in lightGbmOptionsList)
            {
                trainers.Add((
                    $"LightGbm(L={opt.NumberOfLeaves},T={opt.NumberOfIterations},LR={opt.LearningRate})",
                    mlContext.Regression.Trainers.LightGbm(opt)
                ));
            }

            // FastTree tuning
            var fastTreeOptionsList = new[]
            {
        new Microsoft.ML.Trainers.FastTree.FastTreeRegressionTrainer.Options
        {
            NumberOfLeaves = 20,
            NumberOfTrees = 100,
            MinimumExampleCountPerLeaf = 10
        },
        new Microsoft.ML.Trainers.FastTree.FastTreeRegressionTrainer.Options
        {
            NumberOfLeaves = 50,
            NumberOfTrees = 200,
            MinimumExampleCountPerLeaf = 5
        }
    };

            foreach (var opt in fastTreeOptionsList)
            {
                trainers.Add((
                    $"FastTree(L={opt.NumberOfLeaves},T={opt.NumberOfTrees},Min={opt.MinimumExampleCountPerLeaf})",
                    mlContext.Regression.Trainers.FastTree(opt)
                ));
            }

            var sdcaOptionsList = new[]
            {
        new Microsoft.ML.Trainers.SdcaRegressionTrainer.Options
        {
            L2Regularization = 0.001f,
            MaximumNumberOfIterations = 100
        },
        new Microsoft.ML.Trainers.SdcaRegressionTrainer.Options
        {
            L2Regularization = 0.01f,
            MaximumNumberOfIterations = 200
        }
    };

            foreach (var opt in sdcaOptionsList)
            {
                trainers.Add((
                    $"Sdca(L2={opt.L2Regularization},Iter={opt.MaximumNumberOfIterations})",
                    mlContext.Regression.Trainers.Sdca(opt)
                ));
            }

            return trainers;
        }
        private static void TrainModel(MLContext mlContext)
        {
            Console.WriteLine("Loading data...");

            var data = LoadData(mlContext);
            var split = mlContext.Data.TrainTestSplit(data, testFraction: 0.2, seed: 123);

            // Different feature combinations (like your original idea)
            var featureSets = new[]
{
    new { Name = "AllFeatures", Features = new[]
        {
            "Company","City","Product_Category","Payment_Method",
            nameof(OrderData.Customer_Age),
            nameof(OrderData.Order_Value),
            nameof(OrderData.Distance_Km),
            nameof(OrderData.Items_Count),
            nameof(OrderData.Customer_Rating),
            nameof(OrderData.Discount_Applied),
            nameof(OrderData.Delivery_Partner_Rating)
        }
    },

    new { Name = "CoreLogistics", Features = new[]
        {
            nameof(OrderData.Distance_Km),
            nameof(OrderData.Items_Count),
            nameof(OrderData.Order_Value)
        }
    },

    new { Name = "Logistics_Location", Features = new[]
        {
            "City",
            nameof(OrderData.Distance_Km),
            nameof(OrderData.Items_Count),
            nameof(OrderData.Order_Value)
        }
    },

    new { Name = "Logistics_Company", Features = new[]
        {
            "Company",
            nameof(OrderData.Distance_Km),
            nameof(OrderData.Items_Count),
            nameof(OrderData.Order_Value)
        }
    },

    new { Name = "Logistics_Ratings", Features = new[]
        {
            nameof(OrderData.Distance_Km),
            nameof(OrderData.Items_Count),
            nameof(OrderData.Customer_Rating),
            nameof(OrderData.Delivery_Partner_Rating)
        }
    },

    new { Name = "Logistics_Category", Features = new[]
        {
            "Product_Category",
            nameof(OrderData.Distance_Km),
            nameof(OrderData.Items_Count)
        }
    },

    new { Name = "HighImpactMix", Features = new[]
        {
            "City",
            "Company",
            nameof(OrderData.Distance_Km),
            nameof(OrderData.Items_Count),
            nameof(OrderData.Order_Value)
        }
    },

    new { Name = "NoWeakFeatures", Features = new[]
        {
            "City",
            "Company",
            nameof(OrderData.Distance_Km),
            nameof(OrderData.Items_Count),
            nameof(OrderData.Order_Value)
        }
    },

    new { Name = "NoRatings_NoDiscount", Features = new[]
        {
            "City",
            "Company",
            nameof(OrderData.Distance_Km),
            nameof(OrderData.Items_Count),
            nameof(OrderData.Order_Value)
        }
    }
};

            double bestScore = double.MinValue;
            ITransformer bestModel = null;
            string bestFeatureSet = "";
            string bestTrainer = "";

            // weights
            double weightRmse = 0.5;
            double weightMae = 0.3;
            double weightR2 = 0.2;

            // define trainers OUTSIDE loop
            var trainers = GetTunedTrainers(mlContext);

            foreach (var set in featureSets)
            {
                foreach (var trainer in trainers)   
                {
                    try
                    {
                        Console.WriteLine($"\nFeatureSet: {set.Name}, Trainer: {trainer.Name}");

                    var pipeline = BuildPipeline(mlContext, set.Features, trainer.Trainer);

                    var model = pipeline.Fit(split.TrainSet);
                    var predictions = model.Transform(split.TestSet);

                    var metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: "Label");

                    Console.WriteLine($"RMSE: {metrics.RootMeanSquaredError:F4}");
                    Console.WriteLine($"MAE : {metrics.MeanAbsoluteError:F4}");
                    Console.WriteLine($"Rsquare  : {metrics.RSquared:F4}");

                    // scoring
                    double rmseScore = 1.0 / (1.0 + metrics.RootMeanSquaredError);
                    double maeScore = 1.0 / (1.0 + metrics.MeanAbsoluteError);
                    double r2Score = (metrics.RSquared + 1) / 2;

                    double finalScore =
                        (rmseScore * weightRmse) +
                        (maeScore * weightMae) +
                        (r2Score * weightR2);

                    Console.WriteLine($"Final Score: {finalScore:F4}");

                    if (finalScore > bestScore)
                    {
                        bestScore = finalScore;
                        bestModel = model;
                        bestFeatureSet = set.Name;
                        bestTrainer = trainer.Name;
                    }
                    }
                    catch (Exception ex)
                    {
                        Console.WriteLine($"Skipping {set.Name} + {trainer.Name} = {ex.Message}");
                        continue;
                    }
                }
            }

            Console.WriteLine($"\nBest Feature Set: {bestFeatureSet}");
            Console.WriteLine($"Best Trainer: {bestTrainer}");
            Console.WriteLine($"Best Score: {bestScore:F4}");
            using var ms = new MemoryStream();
            mlContext.Model.Save(bestModel, split.TrainSet.Schema, ms);

            SaveModelToDatabase(ms.ToArray(), bestFeatureSet, bestScore);

            Console.WriteLine("Best model saved to database.");

            UseExistingModel(mlContext);
        }
        private static IEstimator<ITransformer> BuildPipeline(
            MLContext mlContext,
            string[] features,
            IEstimator<ITransformer> trainer)
        {
            IEstimator<ITransformer> pipeline =
                mlContext.Transforms.CopyColumns("Label", nameof(OrderData.Delivery_Time_Min));

            var featureColumns = new List<string>();

            foreach (var feature in features)
            {
                if (feature == "Company" || feature == "City" ||
                    feature == "Product_Category" || feature == "Payment_Method")
                {
                    string encoded = feature + "_Encoded";

                    pipeline = pipeline.Append(
                        mlContext.Transforms.Categorical.OneHotEncoding(encoded, feature));

                    featureColumns.Add(encoded);
                }
                else
                {
                    featureColumns.Add(feature);
                }
            }

            pipeline = pipeline
                .Append(mlContext.Transforms.Concatenate("Features", featureColumns.ToArray()))
                .Append(trainer); 

            return pipeline;
        }
        private static IDataView LoadData(MLContext mlContext)
        {
            return mlContext.Data.LoadFromTextFile<OrderData>(
                path: DataPath,
                hasHeader: true,
                separatorChar: ',',   
                allowQuoting: true,
                trimWhitespace: true);
        }

        private static void EnsureModelTableExists()
        {
            using var conn = new SqlConnection(ConnectionString);
            conn.Open();

            string sql = @"
IF OBJECT_ID(N'dbo.MLModels', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.MLModels
    (
        Id INT IDENTITY(1,1) PRIMARY KEY,
        ModelName NVARCHAR(200) NOT NULL,
        ModelData VARBINARY(MAX) NOT NULL,
        Algorithm NVARCHAR(100) NOT NULL,
        RMSE FLOAT NOT NULL,
        Version NVARCHAR(50) NOT NULL,
        CreatedOn DATETIME2 NOT NULL DEFAULT SYSUTCDATETIME()
    );
END";

            using var cmd = new SqlCommand(sql, conn);
            cmd.ExecuteNonQuery();
        }

        private static bool ModelExistsInDatabase()
        {
            using var conn = new SqlConnection(ConnectionString);
            conn.Open();

            string query = "SELECT COUNT(1) FROM dbo.MLModels WHERE ModelName = @name";
            using var cmd = new SqlCommand(query, conn);
            cmd.Parameters.AddWithValue("@name", ModelName);

            int count = Convert.ToInt32(cmd.ExecuteScalar(), CultureInfo.InvariantCulture);
            return count > 0;
        }

        private static void SaveModelToDatabase(byte[] modelBytes, string algorithm, double rmse)
        {
            using var conn = new SqlConnection(ConnectionString);
            conn.Open();

            string query = @"
INSERT INTO dbo.MLModels (ModelName, ModelData, Algorithm, RMSE, Version)
VALUES (@name, @data, @algo, @rmse, @version)";

            using var cmd = new SqlCommand(query, conn);
            cmd.Parameters.AddWithValue("@name", ModelName);
            cmd.Parameters.Add("@data", System.Data.SqlDbType.VarBinary).Value = modelBytes;
            cmd.Parameters.AddWithValue("@algo", algorithm);
            cmd.Parameters.AddWithValue("@rmse", rmse);
            cmd.Parameters.AddWithValue("@version", DateTime.Now.ToString("yyyyMMddHHss"));

            cmd.ExecuteNonQuery();
        }

        private static ITransformer LoadLatestModel(MLContext mlContext)
        {
            using var conn = new SqlConnection(ConnectionString);
            conn.Open();

            string query = @"
SELECT TOP 1 ModelData
FROM dbo.MLModels
WHERE ModelName = @name
ORDER BY Id DESC";

            using var cmd = new SqlCommand(query, conn);
            cmd.Parameters.AddWithValue("@name", ModelName);

            var result = cmd.ExecuteScalar();
            if (result == null || result == DBNull.Value)
                throw new InvalidOperationException("No saved model found in database.");

            byte[] modelBytes = (byte[])result;

            using var ms = new MemoryStream(modelBytes);
            return mlContext.Model.Load(ms, out _);
        }

        private static void UseExistingModel(MLContext mlContext)
        {
            Console.WriteLine("\nLoading latest model...");
            var model = LoadLatestModel(mlContext);

            var predictionEngine =
                mlContext.Model.CreatePredictionEngine<OrderData, DeliveryTimePrediction>(model);

            Console.WriteLine("\nEnter Order Details:");

            var input = new OrderData
            {
                
                Company = ReadString("Company: "),
                City = ReadString("City: "),
                Customer_Age = ReadFloat("Customer Age: "),
                Order_Value = ReadFloat("Order Value: "),
                Distance_Km = ReadFloat("Distance (km): "),
                Items_Count = ReadFloat("Items Count: "),
                Product_Category = ReadString("Product Category: "),
                Payment_Method = ReadString("Payment Method: "),
                Customer_Rating = ReadFloat("Customer Rating: "),
                Discount_Applied = ReadFloat("Discount Applied: "),
                Delivery_Partner_Rating = ReadFloat("Delivery Partner Rating: ")
            };

            var prediction = predictionEngine.Predict(input);

            Console.WriteLine($"\nPredicted Delivery Time: {prediction.PredictedDeliveryMinutes:F2} minutes");
        }

        private static string ReadString(string prompt)
        {
            Console.Write(prompt);
            return Console.ReadLine()?.Trim() ?? string.Empty;
        }

        private static float ReadFloat(string prompt)
        {
            while (true)
            {
                Console.Write(prompt);
                var input = Console.ReadLine();

                if (float.TryParse(input, NumberStyles.Float, CultureInfo.InvariantCulture, out float value))
                    return value;

                if (float.TryParse(input, NumberStyles.Float, CultureInfo.CurrentCulture, out value))
                    return value;

                Console.WriteLine("Invalid number. Try again.");
            }
        }
    }
}