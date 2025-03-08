using System;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NAudio.Wave;
using MathNet.Numerics;
using NAudio.Wave.SampleProviders;
using RVCSharp.AudioProcess;
using RVCSharp.ML.F0Predictor;

namespace RVCSharp.ML
{
    public class OnnxRVC
    {
        private InferenceSession _model;
        private ContentVec _vecModel;
        private int _samplingRate;
        private int _hopSize;

        public OnnxRVC(string modelPath, int samplerate = 40000, int hopsize = 512, string vecPath = "pretrained/vec-768-layer-12.onnx", MLExecutionProvider device = MLExecutionProvider.CPU)
        {
            var options = new SessionOptions();
            switch (device)
            {
                case MLExecutionProvider.CUDA:
                    options.AppendExecutionProvider_CUDA();
                    break;
                case MLExecutionProvider.DirectML:
                    options.AppendExecutionProvider_DML();
                    break;
                case MLExecutionProvider.CPU:
                default:
                    options.AppendExecutionProvider_CPU();
                    break;
            }
            _model = new InferenceSession(modelPath, options);
            _samplingRate = samplerate;
            _hopSize = hopsize;
            _vecModel = new ContentVec(vecPath, device);
        }

        public short[] Forward(DenseTensor<float> hubert, DenseTensor<long> hubertLength, DenseTensor<long> pitch, DenseTensor<float> pitchf, DenseTensor<long> ds, DenseTensor<float> rnd)
        {
            var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_model.InputMetadata.Keys.ElementAt(0), hubert),
            NamedOnnxValue.CreateFromTensor(_model.InputMetadata.Keys.ElementAt(1), hubertLength),
            NamedOnnxValue.CreateFromTensor(_model.InputMetadata.Keys.ElementAt(2), pitch),
            NamedOnnxValue.CreateFromTensor(_model.InputMetadata.Keys.ElementAt(3), pitchf),
            NamedOnnxValue.CreateFromTensor(_model.InputMetadata.Keys.ElementAt(4), ds),
            NamedOnnxValue.CreateFromTensor(_model.InputMetadata.Keys.ElementAt(5), rnd)
        };
            var results = _model.Run(inputs);
            var output = results.First().AsTensor<float>();
            return output.Select(x => (short)(x * 32767)).ToArray();
        }

        public float[] Inference(string rawPath, int sid, IF0Predictor f0Predictor, int f0UpKey = 0)
        {
            var rawwav = AudioProc.LoadWav(rawPath, _samplingRate);
            var seg = AudioProc.Segment(rawwav, _samplingRate * 10, 0.1);
            for (int i = 0; i < seg.Length; i++)
            {
                DateTime start = DateTime.Now;
                Console.Write($"Processing segment ({i}/{seg.Length})... ");
                var orgLength = seg[i].Length;
                var wav16k = AudioProc.Resample(seg[i], _samplingRate, 16000);
                seg[i] = Inference(wav16k, sid, f0Predictor, f0UpKey).Take(orgLength).ToArray();
                float secs = (float)(DateTime.Now - start).TotalSeconds;
                Console.WriteLine($"{secs.ToString("0.00")}s - " + ((secs < 10) ? "realtime" : "slow"));
            }
            return AudioProc.MergeSegments(seg, 0.1);
        }

        public float[] Inference(float[] wav16k, int sid, IF0Predictor f0Predictor, int f0UpKey = 0)
        {
            const int f0Min = 50;
            const int f0Max = 1100;
            double f0MelMin = 1127 * Math.Log(1 + f0Min / 700.0);
            double f0MelMax = 1127 * Math.Log(1 + f0Max / 700.0);

            if (wav16k.Length / 16000 > 30.0)
            {
                throw new Exception("Segment your waves down to less than 30s before inference.");
            }

            var hubert = _vecModel.Forward(wav16k);
            hubert = ContentVec.Transpose(Repeat(hubert, 2), 0, 2, 1).ToDenseTensor();

            var hubertLength = new DenseTensor<long>(new[] { 1 }) { [0] = hubert.Dimensions[1] };

            var orgscale = wav16k.Max() - wav16k.Min();
            var pitchf = f0Predictor.ComputeF0(wav16k, (int)hubertLength[0]);
            pitchf = pitchf.Select(x => (float)(x * Math.Pow(2, f0UpKey / 12.0))).ToArray();
            var f0Mel = pitchf.Select(x => 1127 * Math.Log(1 + x / 700.0)).ToArray();
            f0Mel = f0Mel.Select(x => x > 0 ? (x - f0MelMin) * 254 / (f0MelMax - f0MelMin) + 1 : 1).ToArray();
            f0Mel = f0Mel.Select(x => x > 255 ? 255 : x).ToArray();
            var pitch = f0Mel.Select(x => (long)Math.Round(x)).ToArray();

            var pitchfTensor = new DenseTensor<float>(new[] { 1, pitchf.Length });
            for (int i = 0; i < pitchf.Length; i++)
            {
                pitchfTensor[0, i] = pitchf[i];
            }

            var pitchTensor = new DenseTensor<long>(new[] { 1, pitch.Length });
            for (int i = 0; i < pitch.Length; i++)
            {
                pitchTensor[0, i] = pitch[i];
            }

            var ds = new DenseTensor<long>(new[] { 1 }) { [0] = sid };

            var rnd = new DenseTensor<float>(new[] { 1, 192, (int)hubertLength[0] });
            var random = new Random();
            for (int i = 0; i < 192 * hubertLength[0]; i++)
            {
                rnd[0, (int)(i / hubertLength[0]), (int)(i % hubertLength[0])] = (float)random.NextDouble();
            }

            var outWav = Forward(hubert, hubertLength, pitchTensor, pitchfTensor, ds, rnd);
            outWav = Pad(outWav, 2 * _hopSize);
            return AudioProc.Normalize(outWav, orgscale);
        }

        private DenseTensor<T> Repeat<T>(DenseTensor<T> tensor, int times)
        {
            var dimensions = tensor.Dimensions.ToArray();
            dimensions[2] *= times;
            var result = new DenseTensor<T>(dimensions);
            for (int i = 0; i < tensor.Dimensions[0]; i++)
            {
                for (int j = 0; j < tensor.Dimensions[1]; j++)
                {
                    for (int k = 0; k < tensor.Dimensions[2]; k++)
                    {
                        for (int t = 0; t < times; t++)
                        {
                            result[i, j, k * times + t] = tensor[i, j, k];
                        }
                    }
                }
            }
            return result;
        }

        private short[] Pad(short[] input, int padSize)
        {
            var result = new short[input.Length + padSize];
            Array.Copy(input, result, input.Length);
            return result;
        }
    }
}