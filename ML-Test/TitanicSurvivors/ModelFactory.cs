using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography.X509Certificates;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Learners;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace ML_Test
{
    static class ModelFactory<T, F> where T : class where F : class, new()
    {
        private static PredictionModel<T,F> Create(string path)
        {
            var pipeline = new LearningPipeline();

            pipeline.Add(new TextLoader(path).CreateFrom<T>(separator: ','));

            pipeline.Add(new ColumnCopier(("Survived", "Label")));

            pipeline.Add(new Dictionarizer("Label"));

            pipeline.Add(new ColumnConcatenator("Features", "Pclass", "Sex", "Age"));

            pipeline.Add(new LogisticRegressionClassifier());

            pipeline.Add(new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" });

            PredictionModel<T, F> model = pipeline.Train<T, F>();

            return model;
        }

        public static ML_Model<T,F> CreateModel(string path)
        {
            return new ML_Model<T, F> {Model = Create(path)};
        }
    }
}
