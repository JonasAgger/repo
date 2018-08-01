using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection.Emit;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using ML_Test;

namespace TitanicSurvivors
{
    class Program
    {
        private const string Path = "../../Data/train.csv";
        private const string PathTest = "../../Data/test.csv";

        private const string PathTestDataTransformed = "../../Data/testTransformed.csv";
        private const string PathFullDataTransformed = "../../Data/trainTransformed.csv";

        private const string PathSplitDataTrain = "../../Data/splitTrain.csv";
        private const string PathSplitDataTest = "../../Data/splitTest.csv";

        private const string PathPredictions = "../../Data/predictons.csv";

        static void Main(string[] args)
        {
            ManageData();
            ManageTestData();

            ML_Model<Passenger, PassengerPrediction> model =
                ModelFactory<Passenger, PassengerPrediction>.CreateModel(PathFullDataTransformed);
            /*
            var testData = new TextLoader(PathFullDataTransformed).CreateFrom<Passenger>(separator: ',');

            var evaluator = new Evaluator();;

            RegressionMetrics metrics = evaluator.Evaluate(model.Model, testData);

            Console.WriteLine($"Accuracy is : {metrics}");

            Console.ReadLine();
            */
            var fullTestData = File.ReadAllLines(PathTestDataTransformed);

            var predictions = new List<string>();

            predictions.Add("PassengerId,Survived");

            foreach (var passengerData in fullTestData)
            {
                var splitPassengerData = passengerData.Split(',');

                var passenger = new Passenger()
                {
                    PassengerId = float.Parse(splitPassengerData[0]),
                    Pclass = float.Parse(splitPassengerData[1]),
                    Sex = float.Parse(splitPassengerData[2]),
                    Age = float.Parse(splitPassengerData[3])
                };

                string survived = model.Model.Predict(passenger).PredictedLabels ? "1" : "0";

                predictions.Add(string.Format("{0},{1}", passenger.PassengerId, survived));
            }

            File.WriteAllLines(PathPredictions, predictions);
        }

        private static void ManageData()
        {
            var passengers = File.ReadAllLines(Path).Skip(1).ToList();

            var cleanedUp = new List<string>();

            var newfile = new List<string>();
            var newfile2 = new List<string>();

            // Cleanup

            Regex reg = new Regex("(\")(.*?)(\")");

            foreach (var passenger in passengers)
            {
                string Result = reg.Replace(passenger, "");

                var line = Result.Split(',');


                if (line[0] == String.Empty) line[0] = "0";
                if (line[1] == String.Empty)
                {
                    continue;
                }

                if (line[2] == String.Empty) line[2] = "2";
                if (line[4] == String.Empty) line[4] = new Random().NextDouble() > 0.5 ? "0" : "1";
                if (line[4] == "male") line[4] = "0";
                if (line[4] == "female") line[4] = "1";
                if (line[5] == String.Empty) line[5] = "32";
                if (line[6] == String.Empty) line[6] = "1";
                if (line[7] == String.Empty) line[7] = "1";


                string str = line[0] + ',' + line[1] + ',' + line[2] + ',' + line[4] + ',' + line[5] + ',' + line[6] +
                             ',' + line[7];
                cleanedUp.Add(str);
            }


            int i = 0;

            for (; i < Convert.ToInt32(cleanedUp.Count * 0.75); i++)
            {
                newfile.Add(cleanedUp.ElementAt(i));
            }

            for (; i < cleanedUp.Count; i++)
            {
                newfile2.Add(cleanedUp.ElementAt(i));
            }

            File.WriteAllLines(PathFullDataTransformed, cleanedUp);

            File.WriteAllLines(PathSplitDataTrain, newfile);
            File.WriteAllLines(PathSplitDataTest, newfile2);
        }

        private static void ManageTestData()
        {
            var passengers = File.ReadAllLines(PathTest).Skip(1).ToList();

            var cleanedUp = new List<string>();

            // Cleanup

            Regex reg = new Regex("(\")(.*?)(\")");

            foreach (var passenger in passengers)
            {
                string result = reg.Replace(passenger, "");

                var line = result.Split(',');


                if (line[0] == String.Empty) continue;
                if (line[1] == String.Empty) line[1] = "2";
                if (line[3] == String.Empty) line[3] = new Random().NextDouble() > 0.5 ? "0" : "1";
                if (line[3] == "male") line[3] = "0";
                if (line[3] == "female") line[3] = "1";
                if (line[4] == String.Empty) line[4] = "32";
                if (line[5] == String.Empty) line[5] = "1";
                if (line[6] == String.Empty) line[6] = "1";


                string str = line[0] + ',' + line[1] + ',' + line[3] + ',' + line[4] + ',' + line[5] + ',' + line[6];
                cleanedUp.Add(str);
            }


            File.WriteAllLines(PathTestDataTransformed, cleanedUp);

        }
    }
}
