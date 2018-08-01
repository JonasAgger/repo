using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Concentus.Enums;
using Concentus.Structs;
using NAudio.Wave;
using VoiceChat;

namespace Server_Service2
{
    class NetworkAudioSender : IDisposable
    {
        private readonly IAudioClient audioSender;
        private readonly WaveInEvent waveIn;
        private readonly OpusEncoder encoder = OpusEncoder.Create(48000, 2, OpusApplication.OPUS_APPLICATION_VOIP);

        public NetworkAudioSender(int inputDeviceNumber, IAudioClient audioSender)
        {
            this.audioSender = audioSender;
            waveIn = new WaveInEvent
            {
                BufferMilliseconds = 100,
                DeviceNumber = inputDeviceNumber,
                WaveFormat = new WaveFormat(48000, 16, 2)
            };
            waveIn.DataAvailable += OnAudioCaptured;
            waveIn.StartRecording();
        }

        void OnAudioCaptured(object sender, WaveInEventArgs e)
        {
            short[] audioData = new short[e.BytesRecorded];

            audioData[0] = BitConverter.ToInt16(e.Buffer, 0);
            for (int i = 2; i < e.BytesRecorded-1; i+=2)
            {
                audioData[i / 2] = BitConverter.ToInt16(e.Buffer, i);
            }

            byte[] encoded = new byte[1000];

            var len = encoder.Encode(audioData, 0, 960, encoded, 0, encoded.Length);

            //Array.Copy(encoded, encoded, len);
            audioSender.Send(encoded);
        }

        public void Dispose()
        {
            waveIn.DataAvailable -= OnAudioCaptured;
            waveIn.StopRecording();
            waveIn.Dispose();
            waveIn?.Dispose();
        }
    }

    class NetworkAudioPlayer : IDisposable
    {
        private readonly IAudioClient receiver;
        private readonly IWavePlayer waveOut;
        private readonly BufferedWaveProvider waveProvider;
        private readonly OpusDecoder decoder;

        public NetworkAudioPlayer(IAudioClient receiver)
        {
            this.receiver = receiver;
            decoder = OpusDecoder.Create(48000, 2);
            receiver.SetReceived(OnDataReceived);

            waveOut = new WaveOut();
            waveProvider = new BufferedWaveProvider(new WaveFormat(48000, 16, 2));
            waveOut.Init(waveProvider);
            waveOut.Play();
        }

        private void OnDataReceived(byte[] audioData)
        {
            int frameSize = OpusPacketInfo.GetNumSamples(decoder, audioData, 0, audioData.Length); // must be same as framesize used in input, you can use OpusPacketInfo.GetNumSamples() to determine this dynamically
            short[] outputBuffer = new short[frameSize];

            int thisFrameSize = decoder.Decode(audioData, 0, audioData.Length, outputBuffer, 0, frameSize);

            byte[] decoded = new byte[thisFrameSize * 2];

            for (int i = 0; i < thisFrameSize; i++)
            {
                var data = BitConverter.GetBytes(outputBuffer[i]);
                decoded[i] = data[0];
                decoded[i+1] = data[1];
            }

            waveProvider.AddSamples(decoded, 0, decoded.Length);
        }
        /*
        void OnDataReceived(byte[] compressed)
        {
            byte[] decoded = codec.Decode(compressed, 0, compressed.Length);
            waveProvider.AddSamples(decoded, 0, decoded.Length);
        }
        */
        public void Dispose()
        {
            waveOut?.Dispose();
        }
    }
}
