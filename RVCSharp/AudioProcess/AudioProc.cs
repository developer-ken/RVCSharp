using NAudio.Wave.SampleProviders;
using NAudio.Wave;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RVCSharp.AudioProcess
{
    public class AudioProc
    {
        public static float[] LoadWav(string path, int sampleRate)
        {
            using var reader = new AudioFileReader(path);
            var resampler = new WdlResamplingSampleProvider(reader, sampleRate);
            var samples = new float[reader.Length / sizeof(float)];
            resampler.Read(samples, 0, samples.Length);
            return samples;
        }

        public static void SaveWav(float[] samples, string path, int target_samplerate = 44100, int samplerate = 48000)
        {
            using var writer = new WaveFileWriter(path, new WaveFormat(samplerate, 16, 1));
            var resampler = new WdlResamplingSampleProvider(new ArraySampleProvider(samples, samplerate), target_samplerate);
            var buffer = new float[1024];
            while (resampler.Read(buffer, 0, buffer.Length) > 0)
            {
                for (int i = 0; i < buffer.Length; i++)
                {
                    writer.WriteSample(buffer[i]);
                }
            }
            writer.Close();
        }

        public static float[] Normalize(short[] input, float pp = 1)
        {
            float[] output = new float[input.Length];
            short max = input.Max(), min = input.Min();
            for (int i = 0; i < input.Length; i++)
            {
                output[i] = ((float)input[i]) * pp / (max - min);
            }
            return output;
        }

        public static float[] Resample(float[] input, int originalSampleRate, int targetSampleRate)
        {
            var resampler = new WdlResamplingSampleProvider(new ArraySampleProvider(input, originalSampleRate), targetSampleRate);
            var samples = new float[((long)input.Length * (long)targetSampleRate) / originalSampleRate];
            resampler.Read(samples, 0, samples.Length);
            return samples;
        }

        public static float[][] Segment(float[] input, int unitsize, double overlap = 0.05)
        {
            int newusize = (int)Math.Ceiling(unitsize * (1 - overlap));
            int numFrames = input.Length / newusize;
            float[][] frames = new float[numFrames][];
            for (int i = 0; i < numFrames; i++)
            {
                frames[i] = input.Skip(i * newusize).Take(unitsize).ToArray();
            }
            return frames;
        }

        public static float[] MergeSegments(float[][] segments, double overlap = 0.05)
        {
            int shorterlen = (int)Math.Ceiling(segments[0].Length * (1 - overlap));
            int overlapped = segments[0].Length - shorterlen;
            int numFrames = segments.Length;
            float[] result = new float[shorterlen * numFrames];
            int resultp = 0;
            foreach (var segment in segments)
            {
                if (resultp == 0)  //第一片段，直接拼接
                {
                    for (int i = 0; i < segment.Length; i++)
                    {
                        result[i] = segment[i];
                    }
                    resultp += shorterlen;
                }
                else
                {   //非第一片段，先计算重叠位置的平滑过渡
                    for (int i = resultp; i < resultp + overlapped; i++)
                    {
                        float ratio = (i - resultp) / overlapped;
                        result[i] = ((result[i] * (1 - ratio)) + (segment[i - resultp] * ratio));
                    }
                    resultp += overlapped;

                    //然后拼接后续部分
                    for (int i = resultp; i < resultp + shorterlen; i++)
                    {
                        result[i] = segment[i + overlapped];
                    }
                }
            }
            return result;
        }

        public static float[] BitAverage(float[] former, float[] next, bool grayout = true)
        {
            float[] output = new float[former.Length];
            if (former.Length != next.Length) throw new ArgumentException("Two inputs must have same length.");
            if (!grayout)
                for (int i = 0; i < former.Length; i++)
                {
                    output[i] = 0.5f * (former[i] + next[i]);
                }
            else for (int i = 0; i < former.Length; i++)
                {
                    output[i] = 0.5f * (
                        (((former.Length - i) / former.Length) * former[i]) +
                        ((i / former.Length) * next[i]));
                }
            return output;
        }
    }
}
