package com.whisperonnx.asr;

import static android.content.Intent.FLAG_ACTIVITY_NEW_TASK;

import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.util.Log;

import androidx.preference.PreferenceManager;

import com.whisperonnx.SetupActivity;
import com.whisperonnx.voice_translation.neural_networks.NeuralNetworkApi;
import com.whisperonnx.voice_translation.neural_networks.voice.Recognizer;
import com.whisperonnx.voice_translation.neural_networks.voice.RecognizerListener;

import java.io.File;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

public class Whisper {

    public interface WhisperListener {
        void onUpdateReceived(String message);
        void onResultReceived(WhisperResult result);
    }

    private static final String TAG = "Whisper";
    public static final String MSG_PROCESSING = "Processing...";
    public static final String MSG_PROCESSING_DONE = "Processing done...!";

    private final AtomicBoolean mInProgress = new AtomicBoolean(false);

    private Recognizer.Action mAction;
    private String mLangCode = "";
    private WhisperListener mUpdateListener;

    private final Lock taskLock = new ReentrantLock();
    private final Condition hasTask = taskLock.newCondition();
    private volatile boolean taskAvailable = false;
    private Recognizer recognizer = null;
    private Context mContext;
    private long startTime;
    private OpenAITranscriber openaiTranscriber;
    private String mBackendType;

    public Whisper(Context context) {
        mContext = context;

        SharedPreferences sp = PreferenceManager.getDefaultSharedPreferences(context);
        mBackendType = sp.getString(OpenAITranscriber.PREF_BACKEND_TYPE, OpenAITranscriber.BACKEND_LOCAL);

        if (mBackendType.equals(OpenAITranscriber.BACKEND_OPENAI)) {
            // Remote backend: no model files needed, just start process loop
            Thread threadProcessRecordBuffer = new Thread(this::processRecordBufferLoop);
            threadProcessRecordBuffer.start();
            return;
        }

        // Existing local model check (unchanged)
        File sdcardDataFolder = mContext.getExternalFilesDir(null);

        if (sdcardDataFolder != null && !sdcardDataFolder.exists() && !sdcardDataFolder.mkdirs()) {
            Log.e(TAG, "Failed to make directory: " + sdcardDataFolder);
            return;
        }

        File[] files = sdcardDataFolder.listFiles();

        int fileCount = 0;
        for (File file : files) {
            if (file.isFile()) {
                fileCount++;
            }
        }
        if (fileCount != 6) { //install model
            Intent intent = new Intent(mContext, SetupActivity.class);
            intent.addFlags(FLAG_ACTIVITY_NEW_TASK);
            mContext.startActivity(intent);
        } else { // Start thread for RecordBuffer transcription
            Thread threadProcessRecordBuffer = new Thread(this::processRecordBufferLoop);
            threadProcessRecordBuffer.start();
        }

    }

    public void setListener(WhisperListener listener) {
        this.mUpdateListener = listener;
    }

    public void loadModel() {
        if (mBackendType.equals(OpenAITranscriber.BACKEND_OPENAI)) {
            // No model to load for remote backend
            return;
        }
        recognizer = new Recognizer(mContext, false, new NeuralNetworkApi.InitListener() {
            @Override
            public void onInitializationFinished() {
                Log.d(TAG, "Recognizer initialized");
            }

            @Override
            public void onError(int[] reasons, long value) {
                Log.d(TAG, "Recognizer init error");
            }
        });


        recognizer.addCallback(new RecognizerListener() {
            @Override
            public void onSpeechRecognizedResult(String text, String languageCode, double confidenceScore, boolean isFinal) {
                Log.d(TAG, languageCode + " " + text);
                WhisperResult whisperResult = new WhisperResult(text,languageCode, mAction);

                sendResult(whisperResult);

                long timeTaken = System.currentTimeMillis() - startTime;
                Log.d(TAG, "Time Taken for transcription: " + timeTaken + "ms");
                sendUpdate(MSG_PROCESSING_DONE);
            }

            @Override
            public void onError(int[] reasons, long value) {
                Log.d(TAG, "ERROR during recognition");
            }
        });
    }

    public void unloadModel() {
        if (recognizer != null) {
            recognizer.destroy();
        }
        if (openaiTranscriber != null) {
            openaiTranscriber.shutdown();
            openaiTranscriber = null;
        }
    }

    public void setAction(Recognizer.Action action) {
        this.mAction = action;
    }

    public void setLanguage(String language){
        this.mLangCode = language;
    }

    public void start() {
        if (!mInProgress.compareAndSet(false, true)) {
            Log.d(TAG, "Execution is already in progress...");
            return;
        }
        taskLock.lock();
        try {
            taskAvailable = true;
            hasTask.signal();
        } finally {
            taskLock.unlock();
        }
    }

    public void stop() {
        mInProgress.set(false);
    }

    public boolean isInProgress() {
        return mInProgress.get();
    }

    private void processRecordBufferLoop() {
        while (!Thread.currentThread().isInterrupted()) {
            taskLock.lock();
            try {
                while (!taskAvailable) {
                    hasTask.await();
                }
                processRecordBuffer();
                taskAvailable = false;
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            } finally {
                taskLock.unlock();
            }
        }
    }

    private void processRecordBuffer() {
        try {
            if (RecordBuffer.getOutputBuffer() == null) {
                sendUpdate("No audio data available");
                return;
            }

            // Refresh backend type (user may have toggled)
            SharedPreferences sp = PreferenceManager.getDefaultSharedPreferences(mContext);
            mBackendType = sp.getString(OpenAITranscriber.PREF_BACKEND_TYPE, OpenAITranscriber.BACKEND_LOCAL);

            if (mBackendType.equals(OpenAITranscriber.BACKEND_OPENAI)) {
                Log.d(TAG, "Routing to REMOTE (OpenAI) backend");
                processRecordBufferRemote();
            } else {
                Log.d(TAG, "Routing to LOCAL (ONNX) backend");
                processRecordBufferLocal();
            }
        } catch (Exception e) {
            Log.e(TAG, "Error during transcription", e);
            sendUpdate("Transcription failed: " + e.getMessage());
        } finally {
            mInProgress.set(false);
        }
    }

    private void processRecordBufferLocal() {
        startTime = System.currentTimeMillis();
        sendUpdate(MSG_PROCESSING);
        recognizer.recognize(RecordBuffer.getSamples(), 1, mLangCode, mAction);
    }

    private void processRecordBufferRemote() {
        startTime = System.currentTimeMillis();
        sendUpdate(MSG_PROCESSING);

        if (openaiTranscriber == null) {
            openaiTranscriber = new OpenAITranscriber();
        }

        byte[] pcmData = RecordBuffer.getOutputBuffer();
        if (pcmData.length < 6400) {
            sendUpdate("Recording too short");
            return;
        }

        openaiTranscriber.transcribe(mContext, pcmData, new OpenAITranscriber.Callback() {
            @Override
            public void onSuccess(String transcription) {
                long timeTaken = System.currentTimeMillis() - startTime;
                Log.d(TAG, "OpenAI time taken: " + timeTaken + "ms");

                String detectedLang = detectLanguageFromText(transcription);
                WhisperResult result = new WhisperResult(transcription, detectedLang, mAction);
                sendResult(result);
                sendUpdate(MSG_PROCESSING_DONE);
            }

            @Override
            public void onError(String errorMessage) {
                Log.e(TAG, "OpenAI transcription error: " + errorMessage);
                sendUpdate(errorMessage);
            }
        });
    }

    /**
     * Simple heuristic: if text contains CJK characters, guess "zh".
     * Otherwise default to "en". Used for Chinese postprocessing
     * (switching between traditional/simplified).
     */
    private String detectLanguageFromText(String text) {
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if (Character.UnicodeBlock.of(c) == Character.UnicodeBlock.CJK_UNIFIED_IDEOGRAPHS
                || Character.UnicodeBlock.of(c) == Character.UnicodeBlock.CJK_UNIFIED_IDEOGRAPHS_EXTENSION_A
                || Character.UnicodeBlock.of(c) == Character.UnicodeBlock.CJK_UNIFIED_IDEOGRAPHS_EXTENSION_B) {
                return "zh";
            }
        }
        return "en";
    }

    private void sendUpdate(String message) {
        if (mUpdateListener != null) {
            mUpdateListener.onUpdateReceived(message);
        }
    }

    private void sendResult(WhisperResult whisperResult) {
        if (mUpdateListener != null) {
            mUpdateListener.onResultReceived(whisperResult);
        }
    }

}
