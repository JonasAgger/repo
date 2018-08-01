using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net.Mime;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading.Tasks;
using ToyNeuralNet;


namespace NNToyMNIST
{
    class Program
    {
        private const string path = "./train.csv";
        private const string pathTest = "./test.csv";

        static void Main(string[] args)
        {
            var rnd = new Random();


            Console.WriteLine(
                "Starting Toy Neural Network, doing the MNIST DataSet" + 
                "\n\n" + 
                "Loading Data..\n"
                );

            
            
            var rawData = File.ReadAllLines(path).Skip(1).SelectMany(s =>
            {
                var rawDataString = s.Split(',');
                
                var label = new double[10] {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
                label[int.Parse(rawDataString[0])] = 1.0;

                var data = new double[784];

                for (int i = 0; i < 784; i++)
                    data[i] = (double.Parse(rawDataString[i + 1]) / byte.MaxValue);


                return new [] { new TrainingData(data, label) };
            });



            Console.WriteLine(
                "Shuffling data and creating dataset.\n"
                );

            /*

            var intermediaryDataSet = rawData.OrderBy(d => rnd.Next()).ToList();
            var shuffledDataSet = new List<TrainingData>();
            
            for (int i = 0; i < 10; i++)
            {
                for (int j = 0; j < intermediaryDataSet.Count; j++)
                {
                    shuffledDataSet.Add(intermediaryDataSet[j]);
                }

                intermediaryDataSet = intermediaryDataSet.OrderBy(d => rnd.Next()).ToList();
            }

            
            Console.WriteLine(
                "Training on the Data...\n"
            );


            // Creating Neural Network

            var nn = new NeuralNetwork(784, 400, 10, new Sigmoid(), 0.025);
            var pnn = nn.Copy();



            var stopwatch = new Stopwatch();

            stopwatch.Start();

            for (int i = 0; i < shuffledDataSet.Count; i++)
            {
                if (i % (shuffledDataSet.Count / 100) == 0)
                    Console.WriteLine("{0}% done", i / (shuffledDataSet.Count / 100));

                var trainData = shuffledDataSet[i];
                nn.Train(trainData.Data, trainData.Label);
            }

            stopwatch.Stop();

            Console.WriteLine(
                "Done Training!\n" +
                "Training took: {0}", stopwatch.Elapsed.TotalSeconds

            );

           
            stopwatch.Reset();
            stopwatch.Start();

            Parallel.For(0, shuffledDataSet.Count, i =>
            {
                var trainData = shuffledDataSet[i];
                pnn.Train(trainData.Data, trainData.Label);
            });

            stopwatch.Stop();

            Console.WriteLine(
                "Done Training!\n" +
                "Training took: {0}", stopwatch.Elapsed.TotalSeconds

            );

            Console.WriteLine(
                "Testing Network!"
            );
            

            

            var pred1 = shuffledDataSet[rnd.Next(shuffledDataSet.Count)];
            var pred2 = shuffledDataSet[rnd.Next(shuffledDataSet.Count)];
            var pred3 = shuffledDataSet[rnd.Next(shuffledDataSet.Count)];
            var pred4 = shuffledDataSet[rnd.Next(shuffledDataSet.Count)];
            var pred5 = shuffledDataSet[rnd.Next(shuffledDataSet.Count)];

            var nnpred1 = nn.Predict(pred1.Data);
            var nnpred2 = nn.Predict(pred2.Data);
            var nnpred3 = nn.Predict(pred3.Data);
            var nnpred4 = nn.Predict(pred4.Data);
            var nnpred5 = nn.Predict(pred5.Data);



            double accuracy = 0.0;
            double paccuracy = 0.0;

            for (int i = 0; i < intermediaryDataSet.Count; i++)
            {
                var trainData = intermediaryDataSet[i];

                int label = OneHotNumber(trainData.Label);

                int predLabel = OneHotNumber( nn.Predict( trainData.Data ) );
                int predLabelParallel = OneHotNumber(pnn.Predict(trainData.Data));

                if (label == predLabel)
                    accuracy += 1.0;
                if (label == predLabelParallel)
                    paccuracy += 1.0;
            }

            accuracy /= (double)intermediaryDataSet.Count;
            paccuracy /= (double)intermediaryDataSet.Count;


            Console.WriteLine("Total accuracy on same dataset -- {0:P}", accuracy);
            Console.WriteLine("Total accuracy on same dataset -- Parallel -- {0:P}", paccuracy);


            Console.WriteLine("Prediction {0} -- Label is: {1} -- Neural Network predicted: {2}", 1, OneHotNumber(pred1.Label), OneHotNumber(nnpred1));
            Console.WriteLine("Prediction {0} -- Label is: {1} -- Neural Network predicted: {2}", 2, OneHotNumber(pred2.Label), OneHotNumber(nnpred2));
            Console.WriteLine("Prediction {0} -- Label is: {1} -- Neural Network predicted: {2}", 3, OneHotNumber(pred3.Label), OneHotNumber(nnpred3));
            Console.WriteLine("Prediction {0} -- Label is: {1} -- Neural Network predicted: {2}", 4, OneHotNumber(pred4.Label), OneHotNumber(nnpred4));
            Console.WriteLine("Prediction {0} -- Label is: {1} -- Neural Network predicted: {2}", 5, OneHotNumber(pred5.Label), OneHotNumber(nnpred5));

            */
            //using (var stream = File.Open("model", FileMode.OpenOrCreate))
            //{
            //    var binaryFormatter = new BinaryFormatter();
            //    binaryFormatter.Serialize(stream, nn);
            //}

            


            // -- LOADING FROM FILE
            NeuralNetwork nn;

            using (var stream = File.Open("model", FileMode.Open))
            {
                var binaryFormatter = new BinaryFormatter();
                nn = (NeuralNetwork)binaryFormatter.Deserialize(stream);
            }
            

            Console.WriteLine(
                "Loading TestData and Making Predictions!"
            );


            var image = File.ReadAllBytes("test.bmp");

            image = image.Skip(image.Length - (784 * 3)).ToArray();

            double[] imageArray = new double[784];

            for (int i = 0; i < 784; i++)
            {
                imageArray[i] = (255 - image[i * 3]) / 255.0;
            }

            Console.WriteLine("Predicted: {0}", OneHotNumber(nn.Predict(imageArray)));


            double accuracy = 0.0;

            var intermediaryDataSet = rawData.ToList();

            for (int i = 0; i < intermediaryDataSet.Count; i++)
            {
                var trainData = intermediaryDataSet[i];

                int label = OneHotNumber(trainData.Label);

                int predLabel = OneHotNumber(nn.Predict(trainData.Data));

                if (label == predLabel)
                    accuracy += 1.0;
            }

            accuracy /= (double)intermediaryDataSet.Count;

            Console.WriteLine("Accu: {0:P}", accuracy);

            /*
            var rawTestData = File.ReadAllLines(pathTest).Skip(1).SelectMany(s =>
            {
                var rawDataString = s.Split(',');

                var data = new double[784];

                for (int i = 0; i < 784; i++)
                    data[i] = (double.Parse(rawDataString[i]) / byte.MaxValue);


                return new[] { new TrainingData(data, new double[1]) };
            }).ToList();


            var linesToWrite = new List<string>();

            linesToWrite.Add("ImageId,Label");

            for (int i = 0; i < rawTestData.Count; i++)
            {
                var line = (i + 1).ToString() + "," + OneHotNumber(nn.Predict(rawTestData[i].Data)).ToString();
                linesToWrite.Add(line);
            }


            File.WriteAllLines("testSubmission.csv", linesToWrite);
            
            */
        }




        private static int OneHotNumber(double[] vector)
        {
            // Vector arranged such that the position is the label.
            // Meaning if vector[0] == 1.0, then the value found was "0"
            var maxVal = vector[0];
            int bestPos = 0;
            for (int i = 1; i < vector.Length; i++)
            {
                if (maxVal < vector[i])
                {
                    bestPos = i;
                    maxVal = vector[i];
                }
            }

            return bestPos;
        }


    }
}
