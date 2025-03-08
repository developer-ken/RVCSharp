using System;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NAudio.Wave;
using MathNet.Numerics;
using NAudio.Wave.SampleProviders;
using RVCSharp.AudioProcess;

namespace RVCSharp.ML
{
    public class ContentVec
    {
        private InferenceSession _model;

        public ContentVec(string vecPath = "pretrained/vec-768-layer-12.onnx", MLExecutionProvider device = MLExecutionProvider.CPU)
        {
            Console.WriteLine($"Load model(s) from {vecPath}");
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
            _model = new InferenceSession(vecPath, options);
        }

        public DenseTensor<float> Forward(float[] wav)
        {
            var feats = wav;
            if (feats.Length == 2)
            {
                feats = feats.Select(x => x / 2).ToArray();
            }
            var inputTensor = new DenseTensor<float>(new[] { 1, 1, feats.Length });
            for (int i = 0; i < feats.Length; i++)
            {
                inputTensor[0, 0, i] = feats[i];
            }
            var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_model.InputMetadata.Keys.First(), inputTensor)
        };
            using var results = _model.Run(inputs);
            var logits = results.First().AsTensor<float>();
            return Transpose<float>(logits.ToDenseTensor(), 0, 2, 1);
        }

        public static DenseTensor<T> Transpose<T>(DenseTensor<T> tensor, params int[] perm)
        {
            var dimensions = perm.Select(p => tensor.Dimensions[p]).ToArray();
            var result = new DenseTensor<T>(dimensions);
            var indices = new int[perm.Length];
            for (int i = 0; i < tensor.Length; i++)
            {
                var flatIndex = i;
                for (int j = perm.Length - 1; j >= 0; j--)
                {
                    indices[j] = flatIndex % tensor.Dimensions[j];
                    flatIndex /= tensor.Dimensions[j];
                }
                var transposedIndex = 0;
                for (int j = 0; j < perm.Length; j++)
                {
                    transposedIndex = transposedIndex * dimensions[j] + indices[perm[j]];
                }
                result.SetValue(transposedIndex, tensor.GetValue(i));
            }
            return result;
        }
    }

    public enum MLExecutionProvider
    {
        CPU, CUDA, DirectML
    }
}