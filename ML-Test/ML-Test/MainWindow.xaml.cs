using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.EntryPoints;
using Microsoft.ML.Runtime.Internal.IO;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace ML_Test
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public ML_Model<ImageData, ImagePrediction> ImgModel;
        private IEnumerator<string> _imgEnum;

        private int picture = 1;
        private byte[] data;
        private const string Path = "../../Data/train.csv";

        private const string PathFullDataTransformed = "../../Data/trainTransformed.csv";

        private const string PathSplitDataTrain = "../../Data/splitTrain.csv";
        private const string PathSplitDataTest = "../../Data/splitTest.csv";

        private const string PathAlarmData = "../../Data/alarmClock.npy";
        public MainWindow()
        {
            InitializeComponent();
        }

        private void Button_OnClick(object sender, RoutedEventArgs e)
        {

            ShowPicture(data.Skip(784*picture++).Take(784).ToArray());
            /*
            _imgEnum.MoveNext();

            var firstImg = _imgEnum.Current;

            ShowPicture(firstImg);

            Label2.Content = "Prediction: " + ImgModel.Model.Predict(ConvertToImgData(firstImg.Split(','))).PredictedLabels;
            Label.Content = "Label: " + firstImg[0];
            */
        }


        private void MainWindow_OnLoaded(object sender, RoutedEventArgs e)
        {

            data = File.ReadAllBytes(PathAlarmData).Skip(80).ToArray();

            var image = File.ReadAllBytes("../../Data/test.bmp");

            image = image.Skip(image.Length - (784 * 3)).ToArray();

            double[] imageArray = new double[784];
            byte[] byteArray = new byte[784];
            for (int i = 0; i < 784; i++)
            {
                byteArray[i] = (byte)(255 - image[i * 3]);
                imageArray[i] = (255 - image[i * 3]) / 255.0;
            }


            Label.Content = "Size: " + data.Length/784;

            //ShowPicture(data.Take(784).ToArray());

            ShowPicture(byteArray);

            /*
            var images = File.ReadAllLines(Path);

            images = images.Skip(1).ToArray();

            var newfile = new List<string>();
            var newfile2 = new List<string>();

            int i = 0;

            for (; i < Convert.ToInt32(images.Length * 0.75); i++)
            {
                newfile.Add(images[i]);
            }

            for (; i < images.Length; i++)
            {
                newfile2.Add(images[i]);
            }

            File.WriteAllLines(PathFullDataTransformed, images);

            File.WriteAllLines(PathSplitDataTrain, newfile);
            File.WriteAllLines(PathSplitDataTest, newfile2);


            ImgModel = ModelFactory<ImageData, ImagePrediction>.CreateImgModel(PathFullDataTransformed);


            // Load testData, get results

            /*
            var testData = new TextLoader("../../Data/test2.csv").CreateFrom<ImageData>(separator: ',');

            var evaluator = new ClassificationEvaluator();

            ClassificationMetrics metrics = evaluator.Evaluate(model, testData);

            Label.Content = $"Accuracy is : {metrics.AccuracyMicro}";
            */

            /*
            var images2 = File.ReadLines(Path);

            _imgEnum = images2.GetEnumerator();

            _imgEnum.MoveNext();
            _imgEnum.MoveNext();

            var firstImg = _imgEnum.Current;

            ShowPicture(firstImg);

            Label2.Content = "Prediction: " + ImgModel.Model.Predict(ConvertToImgData(firstImg.Split(','))).PredictedLabels;
            Label.Content = "Label: " + firstImg[0];
            */
        }

        private void ShowPicture(string imgData)
        {
            var firstImg = imgData.Split(',');

            PixelFormat pf = PixelFormats.Gray8;

            int width = 28;
            int height = 28;

            byte[] rawImage = new byte[height * width];

            for (int j = 1; j < firstImg.Length; j++)
                rawImage[j - 1] = byte.Parse(firstImg[j]);

            BitmapSource bitmap = BitmapSource.Create(width, height, 28, 28, pf, null, rawImage, 28);

            Image.Width = 28;

            Image.Source = bitmap;
        }

        private void ShowPicture(byte[] imgData)
        {
            PixelFormat pf = PixelFormats.Gray8;

            int width = 28;
            int height = 28;

            BitmapSource bitmap = BitmapSource.Create(width, height, 28, 28, pf, null, imgData, 28);

            Image.Width = 28;

            Image.Source = bitmap;
        }

        private ImageData ConvertToImgData(string[] inputData, bool hasLabel = true)
        {
            var length = inputData.Length - (hasLabel ? 1 : 0);

            var imgData = new ImageData();

            imgData.Pixels = new float[length];

            for (int i = 0; i < length; i++)
            {
                imgData.Pixels[i] = float.Parse(inputData[i + (hasLabel ? 1 : 0)]);
            }

            return imgData;
        }
    }
}
