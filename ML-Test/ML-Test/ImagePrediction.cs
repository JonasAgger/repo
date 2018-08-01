using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Runtime.Api;

namespace ML_Test
{
    public class ImagePrediction
    {
        [ColumnName("PredictedLabel")]
        public float PredictedLabels;
    }
}
