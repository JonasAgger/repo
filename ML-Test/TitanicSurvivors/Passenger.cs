using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Runtime.Api;

namespace TitanicSurvivors
{
    class Passenger
    {
        // PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
        [Column("0")]
        [ColumnName("PassengerId")]
        public float PassengerId;

        [Column("1")]
        [ColumnName("Survived")]
        public bool Survived;

        [Column("2")]
        [ColumnName("Pclass")]
        public float Pclass;

        [Column("3")]
        [ColumnName("Sex")]
        public float Sex;

        [Column("4")]
        [ColumnName("Age")]
        public float Age;
    }
}
