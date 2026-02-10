package com.medgemma.edge.inference

import android.content.Context
import android.util.Log
import java.io.File

/**
 * Kotlin wrapper for llama.cpp-based MedGemma GGUF inference.
 *
 * Loads the native JNI library (libmedgemma-jni.so) which includes
 * llama.cpp compiled for Android. Provides model loading, text generation,
 * and benchmarking.
 */
class MedGemmaInference(private val context: Context) {

    companion object {
        private const val TAG = "MedGemma"
        private const val MODEL_FILENAME = "medgemma-4b-q4_k_s-final.gguf"

        @Volatile
        private var libraryLoaded = false

        /**
         * Search common locations on device storage for the GGUF model.
         */
        fun findModelPath(context: Context): String? {
            val searchPaths = listOf(
                File(context.getExternalFilesDir(null), "models/$MODEL_FILENAME"),
                File("/storage/emulated/0/MedGemmaEdge/$MODEL_FILENAME"),
                File("/storage/emulated/0/Download/$MODEL_FILENAME"),
            )
            return searchPaths.firstOrNull { it.exists() }?.absolutePath
        }
    }

    // ── Public state ────────────────────────────────────────────────────────
    var isLoaded: Boolean = false
        private set
    var modelSizeMb: Float = 0f
        private set

    // ── JNI declarations ────────────────────────────────────────────────────
    private external fun nativeInit(nativeLibDir: String)
    private external fun nativeLoadModel(modelPath: String): Int
    private external fun nativeGenerate(prompt: String, maxTokens: Int): String
    private external fun nativeGetPartialResult(): String
    private external fun nativeStopGeneration()
    private external fun nativeBench(pp: Int, tg: Int, reps: Int): String
    private external fun nativeSystemInfo(): String
    private external fun nativeUnload()

    // ── Public API ──────────────────────────────────────────────────────────

    /**
     * Initialize the native library. Called automatically by loadModel().
     * Safe to call multiple times.
     */
    fun initialize() {
        if (!libraryLoaded) {
            try {
                System.loadLibrary("medgemma-jni")
                nativeInit(context.applicationInfo.nativeLibraryDir)
                libraryLoaded = true
                Log.i(TAG, "Native library loaded. System info:\n${nativeSystemInfo()}")
            } catch (e: UnsatisfiedLinkError) {
                throw RuntimeException(
                    "Failed to load medgemma-jni native library. " +
                    "Make sure you cloned llama.cpp and rebuilt the project.", e
                )
            }
        }
    }

    /**
     * Load a GGUF model from the given path.
     * @return load time in milliseconds
     */
    fun loadModel(modelPath: String): Long {
        val startTime = System.currentTimeMillis()

        val file = File(modelPath)
        if (!file.exists()) {
            throw RuntimeException("Model file not found: $modelPath")
        }
        modelSizeMb = file.length() / (1024f * 1024f)

        initialize()

        val result = nativeLoadModel(modelPath)
        if (result != 0) {
            throw RuntimeException("Failed to load model (error code: $result)")
        }

        isLoaded = true
        val loadTime = System.currentTimeMillis() - startTime
        Log.i(TAG, "Model loaded in ${loadTime}ms, size: ${"%.1f".format(modelSizeMb)}MB")
        return loadTime
    }

    /**
     * Generate text from a prompt (blocking, writes to streaming buffer).
     * Poll getPartialResult() from another coroutine for streaming updates.
     * @param prompt The input prompt text
     * @param maxTokens Maximum tokens to generate
     * @return Final generated text with stats header
     */
    fun generate(prompt: String, maxTokens: Int = 128): String {
        check(isLoaded) { "Model not loaded. Call loadModel() first." }
        return nativeGenerate(prompt, maxTokens)
    }

    /**
     * Get current streaming generation state.
     * @return StreamingState with tokens generated, speed, generating flag, and partial text
     */
    fun getPartialResult(): StreamingState {
        val raw = nativeGetPartialResult()
        // Format from C++: "tokens|speed|is_generating|text"
        val parts = raw.split("|", limit = 4)
        return if (parts.size >= 4) {
            StreamingState(
                tokensGenerated = parts[0].toIntOrNull() ?: 0,
                tokPerSec = parts[1].toFloatOrNull() ?: 0f,
                isGenerating = parts[2] == "1",
                text = parts[3]
            )
        } else {
            StreamingState(0, 0f, false, raw)
        }
    }

    /**
     * Signal the native generation loop to stop early.
     */
    fun stopGeneration() {
        nativeStopGeneration()
    }

    data class StreamingState(
        val tokensGenerated: Int,
        val tokPerSec: Float,
        val isGenerating: Boolean,
        val text: String
    )

    /**
     * Run prompt-processing and token-generation benchmarks.
     * @param pp Number of prompt tokens to benchmark
     * @param tg Number of generation tokens to benchmark
     * @param reps Number of repetitions
     * @return Formatted benchmark results string
     */
    fun benchmark(pp: Int = 512, tg: Int = 128, reps: Int = 3): String {
        check(isLoaded) { "Model not loaded. Call loadModel() first." }
        return nativeBench(pp, tg, reps)
    }

    /**
     * Get llama.cpp system info (CPU features, etc.)
     */
    fun systemInfo(): String {
        initialize()
        return nativeSystemInfo()
    }

    /**
     * Unload the model and free native resources.
     */
    fun release() {
        if (isLoaded) {
            nativeUnload()
            isLoaded = false
            Log.i(TAG, "Model released")
        }
    }
}
