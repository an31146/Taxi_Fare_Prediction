using System;
using System.IO;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Core.Data;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Categorical;
using Microsoft.ML.Transforms.Normalizers;
using Microsoft.ML.Transforms.Text;

using static System.Console;

#pragma warning disable IDE0040
namespace TaxiFarePrediction
{
    class Program
    {
        static readonly string _trainDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-train.csv");
        static readonly string _testDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "taxi-fare-test.csv");
        static readonly string _modelPath = Path.Combine(Environment.CurrentDirectory, "Models", "Model.zip");
        static TextLoader _textLoader;

        static void Main(string[] args)
        {
            WriteLine(Environment.CurrentDirectory);

            MLContext mlContext = new MLContext(seed: 0);

            _textLoader = mlContext.Data.CreateTextReader(new TextLoader.Arguments()
            {
                Separator = ",",
                HasHeader = true,
                Column = new[]
                            {
                                new TextLoader.Column("VendorId", DataKind.Text, 0),
                                new TextLoader.Column("RateCode", DataKind.Text, 1),
                                new TextLoader.Column("PassengerCount", DataKind.R4, 2),
                                new TextLoader.Column("TripTime", DataKind.R4, 3),
                                new TextLoader.Column("TripDistance", DataKind.R4, 4),
                                new TextLoader.Column("PaymentType", DataKind.Text, 5),
                                new TextLoader.Column("FareAmount", DataKind.R4, 6)
                            }
            }
            );

            dynamic model;
            if (!File.Exists(_modelPath))
            {
                model = Train(mlContext, _trainDataPath);
                Evaluate(mlContext, model);
            }
            else
            {
                using (FileStream stream = new FileStream(_modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
                {
                    model = mlContext.Model.Load(stream);
                }
            }

            //TestTwoPredictions(mlContext);
            ReadAndTestRandomRowFromFile(mlContext);
            //TestRandomPrediction(mlContext);
        }

        public static ITransformer Train(MLContext mlContext, string dataPath)
        {
            IDataView dataView = _textLoader.Read(dataPath);

            var pipeline = mlContext.Transforms.CopyColumns("FareAmount", "Label")
                    .Append(mlContext.Transforms.Categorical.OneHotEncoding("VendorId"))
                    .Append(mlContext.Transforms.Categorical.OneHotEncoding("RateCode"))
                    .Append(mlContext.Transforms.Categorical.OneHotEncoding("PaymentType"))
                    .Append(mlContext.Transforms.Concatenate("Features", "VendorId", "RateCode", "PassengerCount", "TripTime", "TripDistance", "PaymentType"))
                    .Append(mlContext.Regression.Trainers.FastTree());

            WriteLine("=============== Create and Train the Model ===============");
            var model = pipeline.Fit(dataView);
            WriteLine("=============== End of training ===============\n");

            SaveModelAsFile(mlContext, model);
            return model;
        }

        private static void Evaluate(MLContext mlContext, ITransformer model)
        {
            IDataView dataView = _textLoader.Read(_testDataPath);

            var predictions = model.Transform(dataView);
            var metrics = mlContext.Regression.Evaluate(predictions, "Label", "Score");

            WriteLine();
            WriteLine($"*************************************************");
            WriteLine($"*       Model quality metrics evaluation         ");
            WriteLine($"*------------------------------------------------");
            WriteLine($"*       R2 Score:      {metrics.RSquared:0.##}");
            WriteLine($"*       RMS loss:      {metrics.Rms:#.##}");
            WriteLine($"*************************************************\n");
        }

        private static void TestTwoPredictions(MLContext mlContext)
        {
            //load the model
            ITransformer loadedModel;
            using (var stream = new FileStream(_modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                loadedModel = mlContext.Model.Load(stream);
            }

            //Prediction test
            // Create prediction function and make prediction.
            var predictionFunction = loadedModel.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(mlContext);
            //Sample: 
            //vendor_id,rate_code,passenger_count,trip_time_in_secs,trip_distance,payment_type,fare_amount
            //VTS,1,1,1140,3.75,CRD,15.5
            var taxiTripSample = new TaxiTrip()
            {
                VendorId = "VTS",
                RateCode = "1",
                PassengerCount = 4,
                TripTime = 1140,
                TripDistance = 3.75f,
                PaymentType = "CRD",
                FareAmount = 0 // To predict. Actual/Observed = 15.5
            };
            var prediction = predictionFunction.Predict(taxiTripSample);

            WriteLine($"**********************************************************************");
            WriteLine($"Predicted fare: {prediction.FareAmount:0.####}, actual fare: 15.5");
            WriteLine($"**********************************************************************\n");

            //  VTS 1   2   1140    6.29    CRD 21
            taxiTripSample = new TaxiTrip()
            {
                VendorId = "VTS",
                RateCode = "1",
                PassengerCount = 2,
                TripTime = 1140,
                TripDistance = 6.29f,
                PaymentType = "CRD",
                FareAmount = 0 // To predict. Actual/Observed = 21
            };
            prediction = predictionFunction.Predict(taxiTripSample);

            WriteLine($"**********************************************************************");
            WriteLine($"Predicted fare: {prediction.FareAmount:0.####}, actual fare: 21");
            WriteLine($"**********************************************************************\n");
        }

        private static void ReadAndTestRandomRowFromFile(MLContext mlContext)
        {
            if (File.Exists(_testDataPath))
            {
                using (StreamReader reader = new StreamReader(_testDataPath))
                {
                    string strFile = reader.ReadToEnd();
                    Random rand = new Random();
                    List<string> strArray = new List<string>(strFile.Split("\n"));
                    List<string> randomTaxiTripSample = new List<string>(strArray[rand.Next(1, strArray.Count)].Split(","));

                    ITransformer loadedModel;
                    using (var stream = new FileStream(_modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
                    {
                        loadedModel = mlContext.Model.Load(stream);
                    }

                    var predictionFunction = loadedModel.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(mlContext);
                    var taxiTripSample = new TaxiTrip()
                    {
                        VendorId = randomTaxiTripSample[0],
                        RateCode = randomTaxiTripSample[1],
                        PassengerCount = float.Parse(randomTaxiTripSample[2]),
                        TripTime = float.Parse(randomTaxiTripSample[3]),
                        TripDistance = float.Parse(randomTaxiTripSample[4]),
                        PaymentType = randomTaxiTripSample[5],
                        FareAmount = float.Parse(randomTaxiTripSample[6])
                    };

                    var prediction = predictionFunction.Predict(taxiTripSample);

                    WriteLine($"**********************************************************************");
                    WriteLine($"VendorId:       {taxiTripSample.VendorId}");
                    WriteLine($"RateCode:       {taxiTripSample.RateCode}");
                    WriteLine($"PassengerCount: {taxiTripSample.PassengerCount}");
                    WriteLine($"TripTime:       {taxiTripSample.TripTime}");
                    WriteLine($"TripDistance:   {taxiTripSample.TripDistance}");
                    WriteLine($"PaymentType:    {taxiTripSample.PaymentType}");
                    WriteLine($"Predicted fare: {prediction.FareAmount:0.####}, actual fare: {taxiTripSample.FareAmount}");
                    WriteLine($"**********************************************************************\n");
                }
            }
            else
            {
                WriteLine($"{_testDataPath} not found.");
            }
        }

        private static void TestRandomPrediction(MLContext mlContext)
        {
            Random rand = new Random();

            //load the model
            ITransformer loadedModel;
            using (var stream = new FileStream(_modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                loadedModel = mlContext.Model.Load(stream);
            }

            //Prediction test
            // Create prediction function and make prediction.
            var predictionFunction = loadedModel.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(mlContext);
	        //
            //Sample: 
            //vendor_id,rate_code,passenger_count,trip_time_in_secs,trip_distance,payment_type,fare_amount
            //VTS,1,1,1140,3.75,CRD,15.5

            var taxiTripSample = new TaxiTrip()
            {
                RateCode = "1",                                 // 1 - metered, 2 - fixed
                PassengerCount = rand.Next(1, 6),               // 1 - 6 passengers
                TripTime = rand.Next(6, 1100) * 10,             // nearest 10 secs.
                TripDistance = ((float)rand.NextDouble() * 100.0f) + 0.1f,
                FareAmount = 0 // To predict. Actual/Observed = 15.5
            };
	        //
            // Pick a vendor ID
            if (rand.Next() % 2 == 0)
                taxiTripSample.VendorId = "VTS";
            else
                taxiTripSample.VendorId = "CMT";

            if (rand.Next() % 2 == 0)
                taxiTripSample.PaymentType = "CSH";
            else
                taxiTripSample.PaymentType = "CRD";

            var prediction = predictionFunction.Predict(taxiTripSample);

            WriteLine($"**********************************************************************");
            WriteLine($"VendorId:       {taxiTripSample.VendorId}");
            WriteLine($"RateCode:       {taxiTripSample.RateCode}");
            WriteLine($"PassengerCount: {taxiTripSample.PassengerCount}");
            WriteLine($"TripTime:       {taxiTripSample.TripTime}");
            WriteLine($"TripDistance:   {taxiTripSample.TripDistance}");
            WriteLine($"PaymentType:    {taxiTripSample.PaymentType}");
            WriteLine($"Predicted fare: {prediction.FareAmount:0.####}, actual fare: 15.5");
            WriteLine($"**********************************************************************\n");

        }

        private static void SaveModelAsFile(MLContext mlContext, ITransformer model)
        {
            if (!Directory.Exists(Path.Combine(Environment.CurrentDirectory, "Models")))
                Directory.CreateDirectory(Path.Combine(Environment.CurrentDirectory, "Models"));

            using (var fileStream = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
                mlContext.Model.Save(model, fileStream);

            WriteLine("The model is saved to {0}\n", _modelPath);
        }
    }
}
