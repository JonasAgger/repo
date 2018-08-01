using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Single;
using ToyNeuralNet;

namespace NNToy
{
    class Program
    {
        static void Main(string[] args)
        {
            var rnd = new Random();
            var training_data = new List<TrainingData>();
            
            // Training data is an object, with a data segment and a label segment, both as double arrays

            // XOR dataset. input is binary 1 and 0. Output is an onehot vector, here signaling [ 0 , 1 ]. 
            training_data.Add(new TrainingData(new double[] { 0.0, 0.0 }, new double[] { 1.0, 0.0 }));
            training_data.Add(new TrainingData(new double[] { 1.0, 0.0 }, new double[] { 0.0, 1.0 }));
            training_data.Add(new TrainingData(new double[] { 0.0, 1.0 }, new double[] { 0.0, 1.0 }));
            training_data.Add(new TrainingData(new double[] { 1.0, 1.0 }, new double[] { 1.0, 0.0 }));


            // Neural Net, setup
            var nn = new NeuralNetwork(2,4,2, learningRate:0.1);


           // Training
            for (int i = 0; i < 1000000; i++)
            {
                var data = training_data[rnd.Next(training_data.Count)]; // Get a random data point
                nn.Train(data.Data, data.Label); // Train on that point
            }

            // Predicting
            Console.WriteLine("NN is predicting '0,0' to be {0}", nn.Predict(new double[] { 0.0, 0.0 })[0]);
            Console.WriteLine("NN is predicting '1,1' to be {0}", nn.Predict(new double[] { 1.0, 1.0 })[0]);
            Console.WriteLine("NN is predicting '0,1' to be {0}", nn.Predict(new double[] { 0.0, 1.0 })[0]);
            Console.WriteLine("NN is predicting '1,0' to be {0}", nn.Predict(new double[] { 1.0, 0.0 })[0]);

        }


    }
}
