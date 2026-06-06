package com.whisperonnx.asr;

/**
 * Wraps raw PCM 16-bit mono audio in a WAV RIFF header.
 * OpenAI /v1/audio/transcriptions requires a file with extension;
 * WAV with PCM payload qualifies.
 */
public class WavUtils {

    /**
     * @param pcmData       Raw PCM 16-bit mono bytes
     * @param sampleRate    Sample rate in Hz (16000 for Whisper)
     * @param channels      Number of channels (1 for mono)
     * @param bitsPerSample Bits per sample (16)
     * @return Complete WAV file bytes (44-byte header + PCM data)
     */
    public static byte[] pcmToWav(byte[] pcmData, int sampleRate, int channels, int bitsPerSample) {
        int headerSize = 44;
        int dataSize = pcmData.length;
        byte[] wav = new byte[headerSize + dataSize];

        // RIFF chunk descriptor
        writeFourCC(wav, 0, "RIFF");
        writeIntLE(wav, 4, 36 + dataSize);       // ChunkSize
        writeFourCC(wav, 8, "WAVE");

        // fmt sub-chunk
        writeFourCC(wav, 12, "fmt ");
        writeIntLE(wav, 16, 16);                 // Subchunk1Size (PCM)
        writeShortLE(wav, 20, (short) 1);        // AudioFormat (1 = PCM)
        writeShortLE(wav, 22, (short) channels); // NumChannels
        writeIntLE(wav, 24, sampleRate);         // SampleRate
        writeIntLE(wav, 28, sampleRate * channels * bitsPerSample / 8); // ByteRate
        writeShortLE(wav, 32, (short) (channels * bitsPerSample / 8));  // BlockAlign
        writeShortLE(wav, 34, (short) bitsPerSample);                   // BitsPerSample

        // data sub-chunk
        writeFourCC(wav, 36, "data");
        writeIntLE(wav, 40, dataSize);            // Subchunk2Size

        // Copy PCM payload
        System.arraycopy(pcmData, 0, wav, headerSize, dataSize);

        return wav;
    }

    private static void writeFourCC(byte[] dest, int offset, String fourCC) {
        for (int i = 0; i < 4; i++) {
            dest[offset + i] = (byte) fourCC.charAt(i);
        }
    }

    private static void writeIntLE(byte[] dest, int offset, int value) {
        dest[offset]     = (byte) (value & 0xFF);
        dest[offset + 1] = (byte) ((value >> 8) & 0xFF);
        dest[offset + 2] = (byte) ((value >> 16) & 0xFF);
        dest[offset + 3] = (byte) ((value >> 24) & 0xFF);
    }

    private static void writeShortLE(byte[] dest, int offset, short value) {
        dest[offset]     = (byte) (value & 0xFF);
        dest[offset + 1] = (byte) ((value >> 8) & 0xFF);
    }
}
