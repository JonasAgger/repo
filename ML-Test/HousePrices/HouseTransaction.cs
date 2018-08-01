using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Runtime.Api;

namespace HousePrices
{
    class HouseTransaction
    {
        [Column("0")]
        [ColumnName("Number")]
        public float Number;

        [Column("1-784")]
        [VectorType(784)]
        [ColumnName("Pixels")]
        public float[] Pixels;
    }
}
