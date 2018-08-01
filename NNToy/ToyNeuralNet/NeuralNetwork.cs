using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;

namespace ToyNeuralNet
{
    public interface IActivationFunction
    {
        double Activasion(double input);
        double DActivasion(double input);
    }
    [Serializable]
    public class Sigmoid : IActivationFunction
    {
        public double Activasion(double input)
        {
            return 1.0 / (1.0 + Math.Exp(-input));
        }

        public double DActivasion(double input)
        {
            return input * (1 - input);
        }
    }
    [Serializable]
    public class Tahn : IActivationFunction
    {
        public double Activasion(double input)
        {
            return Math.Tanh(input);
        }

        public double DActivasion(double input)
        {
            return 1 - (input * input);
        }
    }

    [Serializable]
    public class NeuralNetwork
    {
        private int _inputNodes;
        private int _hiddenNodes;
        private int _outputNodes;

        private double _learningRate;

        private Matrix<double> _weightsHiddenOut;
        private object _wHiddenOutLock = new object();
        private Matrix<double> _weightsInHidden;
        private object _wInHiddenLock = new object();
        private Matrix<double> _biasHidden;
        //private object _wBiasHiddenLock = new object();
        private Matrix<double> _biasOut;
        //private object _wBiasOutLock = new object();

        private IActivationFunction _activation;

        public NeuralNetwork(int inputNodes, int hiddenNodes, int outputNodes, IActivationFunction activation = null, double learningRate = 0.01)
        {
            _hiddenNodes = hiddenNodes;
            _outputNodes = outputNodes;
            _activation = activation;
            _inputNodes = inputNodes;

            var rnd = new Random();

            _weightsInHidden = Matrix<double>.Build.Dense(hiddenNodes, inputNodes, (i, j) => rnd.NextDouble() * 2 - 1);
            _weightsHiddenOut = Matrix<double>.Build.Dense(outputNodes, hiddenNodes, (i, j) => rnd.NextDouble() * 2 - 1);

            _biasHidden = Matrix<double>.Build.Dense(hiddenNodes, 1, (i, j) => rnd.NextDouble() * 2 - 1);
            _biasOut = Matrix<double>.Build.Dense(outputNodes, 1, (i, j) => rnd.NextDouble() * 2 - 1);

            _activation = activation ?? new Sigmoid();

            _learningRate = learningRate;
        }

        public double[] Predict(double[] input)
        {
            // Making matrix from input
            var inputs = Matrix<double>.Build.Dense(input.Length, 1, (i, j) => input[i]);
            // Adding Weights between Input and Hidden layer
            var hidden = _weightsInHidden.Multiply(inputs);
            // Adding Hidden layer Bias
            hidden.Add(_biasHidden, hidden);
            // Activation Function
            hidden.MapInplace(val => _activation.Activasion(val));

            // Adding Weights between Hidden and Output layer
            var output = _weightsHiddenOut.Multiply(hidden);
            // Adding Output layer Bias
            output.Add(_biasOut, output);
            // Activation Function
            output.MapInplace(val => _activation.Activasion(val));

            return output.ToRowMajorArray();
        }

        public void Train(double[] input, double[] labelArray)
        {
            // Making prediction
            var inputs = Matrix<double>.Build.Dense(input.Length, 1, (i, j) => input[i]);

            var hidden = _weightsInHidden.Multiply(inputs);

            hidden.Add(_biasHidden, hidden);

            hidden.MapInplace(val => _activation.Activasion(val));

            var output = _weightsHiddenOut.Multiply(hidden);

            output.Add(_biasOut, output);

            output.MapInplace(val => _activation.Activasion(val));


            // Building target matrix
            var target = Matrix<double>.Build.Dense(labelArray.Length, 1, (i, j) => labelArray[i]);

            // Calculate error
            // Error => Target - output
            var errors = target.Subtract(output);

            // Gradient = outputs * (1 - outputs);
            // Calculate gradient

            var gradiants = output.Map(val => _activation.DActivasion(val));
            InPlaceMultiply(ref gradiants, ref errors);
            gradiants.Multiply(_learningRate, gradiants);

            // Calculate deltas
            var hidden_t = hidden.Transpose();
            var weight_hidden_out_deltas = gradiants.Multiply(hidden_t);

            //lock (_wHiddenOutLock)
            //{
                // Adjust weights by deltas
                _weightsHiddenOut.Add(weight_hidden_out_deltas, _weightsHiddenOut);
                // Adjust bias by deltas, which is gradiant
                _biasOut.Add(gradiants, _biasOut);
            //}

            // Calculate hidden layer errors
            var weights_hidden_out_transposed = _weightsHiddenOut.Transpose();
            var hidden_errors = weights_hidden_out_transposed.Multiply(errors);

            // Calculate hidden gradiant
            var hidden_gradiant = hidden.Map(val => _activation.DActivasion(val));
            InPlaceMultiply(ref hidden_gradiant, ref hidden_errors);
            hidden_gradiant.Multiply(_learningRate, hidden_gradiant);

            // Calculate input => hidden deltas
            var inputs_t = inputs.Transpose();
            var weights_input_hidden_deltas = hidden_gradiant.Multiply(inputs_t);

            //lock (_wInHiddenLock)
            //{
                _weightsInHidden.Add(weights_input_hidden_deltas, _weightsInHidden);
                _biasHidden.Add(hidden_gradiant, _biasHidden);
            //}
        }

        public void PTrain(double[] input, double[] labelArray)
        {
            // Making prediction
            var inputs = Matrix<double>.Build.Dense(input.Length, 1, (i, j) => input[i]);

            var hidden = _weightsInHidden.Multiply(inputs);

            hidden.Add(_biasHidden, hidden);

            hidden.MapInplace(val => _activation.Activasion(val));

            var output = _weightsHiddenOut.Multiply(hidden);

            output.Add(_biasOut, output);

            output.MapInplace(val => _activation.Activasion(val));


            // Building target matrix
            var target = Matrix<double>.Build.Dense(labelArray.Length, 1, (i, j) => labelArray[i]);

            // Calculate error
            // Error => Target - output
            var errors = target.Subtract(output);

            // Gradient = outputs * (1 - outputs);
            // Calculate gradient

            var gradiants = output.Map(val => _activation.DActivasion(val));
            InPlaceMultiply(ref gradiants, ref errors);
            gradiants.Multiply(_learningRate, gradiants);

            // Calculate deltas
            var hidden_t = hidden.Transpose();
            var weight_hidden_out_deltas = gradiants.Multiply(hidden_t);

            lock (_wHiddenOutLock)
            {
                // Adjust weights by deltas
                _weightsHiddenOut.Add(weight_hidden_out_deltas, _weightsHiddenOut);
                // Adjust bias by deltas, which is gradiant
                _biasOut.Add(gradiants, _biasOut);
            }

            // Calculate hidden layer errors
            var weights_hidden_out_transposed = _weightsHiddenOut.Transpose();
            var hidden_errors = weights_hidden_out_transposed.Multiply(errors);

            // Calculate hidden gradiant
            var hidden_gradiant = hidden.Map(val => _activation.DActivasion(val));
            InPlaceMultiply(ref hidden_gradiant, ref hidden_errors);
            hidden_gradiant.Multiply(_learningRate, hidden_gradiant);

            // Calculate input => hidden deltas
            var inputs_t = inputs.Transpose();
            var weights_input_hidden_deltas = hidden_gradiant.Multiply(inputs_t);

            lock (_wInHiddenLock)
            {
                _weightsInHidden.Add(weights_input_hidden_deltas, _weightsInHidden);
                _biasHidden.Add(hidden_gradiant, _biasHidden);
            }
        }


        public NeuralNetwork Copy()
        {
            using (var ms = new MemoryStream())
            {
                var formatter = new BinaryFormatter();
                formatter.Serialize(ms, this);
                ms.Position = 0;

                return (NeuralNetwork)formatter.Deserialize(ms);
            }
        }


        public void Mutate(Func<double, double> f)
        {
            _weightsInHidden.MapInplace(f);
            _weightsHiddenOut.MapInplace(f);

            _biasHidden.MapInplace(f);
            _biasOut.MapInplace(f);
        }

        private void InPlaceMultiply(ref Matrix<double> input, ref Matrix<double> scalar)
        {
            if (input.RowCount != scalar.RowCount)
                throw new ArgumentException(
                    $"InPlaceMultiply's input size dident match the scalars input size\nSizes are input:{input.RowCount} -- scalar:{scalar.RowCount}");

            for (int i = 0; i < input.RowCount; i++)
                input[i, 0] *= scalar[i, 0];
        }
    }
}
