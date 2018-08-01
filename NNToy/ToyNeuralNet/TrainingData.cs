using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ToyNeuralNet
{
    [Serializable]
    public class TrainingData
    {
        public double[] Data;
        public double[] Label;

        public TrainingData(double[] data, double[] label)
        {
            Data = data;
            Label = label;
        }
    }
}
