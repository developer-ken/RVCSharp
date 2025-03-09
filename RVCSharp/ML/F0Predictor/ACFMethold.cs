using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RVCSharp.ML.F0Predictor
{
    public class ACFMethold : IF0Predictor
    {
        private int HopLength, SampleRate;
        public ACFMethold(int hopLength, int samplingRate)
        {
            HopLength = hopLength;
            SampleRate = samplingRate;
        }

        public float[] ComputeF0(float[] wav, int length)
        {
            HopLength = (int)Math.Floor(((double)wav.Length / (double)length));
            return ComputeF0(wav).Take(length).ToArray();
        }

        public float[] ComputeF0(float[] audioData)
        {
            int frameLength = HopLength;// (int)(FramePeriod / 1000.0 * SampleRate);
            int numFrames = audioData.Length / frameLength;
            float[] f0 = new float[numFrames];

            for (int i = 0; i < numFrames; i++)
            {
                float[] frame = audioData.Skip(i * frameLength).Take((int)(frameLength * 1.5)).ToArray();
                f0[i] = ComputeF0ForFrame(frame);
            }
            return f0;
        }

        private float ComputeF0ForFrame(float[] frame)
        {
            int n = frame.Length;
            float[] autocorrelation = new float[n];

            // 计算自相关函数
            for (int lag = 0; lag < n; lag++)
            {
                for (int i = 0; i < n - lag; i++)
                {
                    autocorrelation[lag] += (float)(frame[i] * frame[i + lag]);
                }
            }

            // 忽略零延迟的峰值，寻找第一个非零延迟的峰值
            int peakIndex = 1;
            float maxVal = autocorrelation[1];
            for (; (autocorrelation.Length > peakIndex) && ((maxVal = autocorrelation[peakIndex]) > 0); peakIndex++) ;
            for (int lag = peakIndex; lag < n; lag++)
            {
                if (autocorrelation[lag] > maxVal)
                {
                    maxVal = autocorrelation[lag];
                    peakIndex = lag;
                }
            }

            // 计算基频
            float f0 = SampleRate / (float)peakIndex;

            return f0;
        }
    }
}
