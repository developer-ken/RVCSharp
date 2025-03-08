using RVCSharp.AudioProcess;
using RVCSharp.ML;
using RVCSharp.ML.F0Predictor;

namespace RVCSharpDemo
{
    internal class Program
    {
        static void Main(string[] args)
        {
            OnnxRVC rvc = new OnnxRVC("D:\\mldys_eggv3_sing_r2.onnx", 48000, 512, "D:\\vec-768-layer-12.onnx", MLExecutionProvider.CPU);
            IF0Predictor f0 = new ACFMethold(512, 16000);
            var data = rvc.Inference(@"D:\Downloads\tmpcrc4vt1c.wav", 0, f0);
            AudioProc.SaveWav(data, "D:\\Downloads\\e2e22.wav", 44100, 48000);
        }
    }
}
