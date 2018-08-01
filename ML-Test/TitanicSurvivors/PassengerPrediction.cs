using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Runtime.Api;

namespace TitanicSurvivors
{
    class PassengerPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool PredictedLabels;
    }
}
