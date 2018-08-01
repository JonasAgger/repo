using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML;

namespace ML_Test
{

    public class ML_Model<T, F> where T : class where F : class, new()
    { 
        public PredictionModel<T, F> Model;
    }
}
