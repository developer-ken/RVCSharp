using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RVCSharp.ML.F0Predictor
{
    public interface IF0Predictor
    {
        public float[] ComputeF0(float[] wav, int length);
    }
}
