package com.medgemma.edge.inference

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Debug
import android.util.Log
import java.io.File
import java.nio.FloatBuffer

/**
 * BiomedCLIP ONNX INT8 inference engine.
 *
 * Loads biomedclip_vision_int8.onnx from device storage and runs
 * image → 512-dim embedding inference.
 */
class BiomedClipInference(private val context: Context) {

    companion object {
        private const val TAG = "BiomedCLIP"
        private const val MODEL_FILENAME = "biomedclip_vision_int8.onnx"
        private const val IMAGE_SIZE = 224
        private const val EMBEDDING_DIM = 512

        // ImageNet normalization (used by BiomedCLIP)
        private val MEAN = floatArrayOf(0.48145466f, 0.4578275f, 0.40821073f)
        private val STD = floatArrayOf(0.26862954f, 0.26130258f, 0.27577711f)

        /**
         * Find model file in common locations on device storage.
         */
        fun findModelPath(context: Context): String? {
            val searchPaths = listOf(
                // Primary: app-specific external storage
                File(context.getExternalFilesDir(null), "models/$MODEL_FILENAME"),
                // Secondary: shared storage
                File("/storage/emulated/0/MedGemmaEdge/$MODEL_FILENAME"),
                File("/storage/emulated/0/Download/$MODEL_FILENAME"),
            )
            return searchPaths.firstOrNull { it.exists() }?.absolutePath
        }
    }

    private var ortEnv: OrtEnvironment? = null
    private var session: OrtSession? = null

    // Benchmark data
    var modelSizeMb: Float = 0f
        private set
    var loadTimeMs: Long = 0
        private set
    var isLoaded: Boolean = false
        private set

    /**
     * Load the ONNX model from the given path.
     * Returns load time in milliseconds.
     */
    fun loadModel(modelPath: String): Long {
        Log.d(TAG, "Loading model from: $modelPath")

        val startTime = System.currentTimeMillis()

        val modelFile = File(modelPath)
        modelSizeMb = modelFile.length() / (1024f * 1024f)

        ortEnv = OrtEnvironment.getEnvironment()

        val sessionOptions = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(4)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            // Try NNAPI for hardware acceleration (fails gracefully on emulator)
            try {
                addNnapi()
                Log.d(TAG, "NNAPI enabled")
            } catch (e: Exception) {
                Log.d(TAG, "NNAPI not available, using CPU: ${e.message}")
            }
        }

        session = ortEnv!!.createSession(modelPath, sessionOptions)

        loadTimeMs = System.currentTimeMillis() - startTime
        isLoaded = true

        val inputInfo = session!!.inputInfo
        val outputInfo = session!!.outputInfo
        Log.d(TAG, "Model loaded. Input: ${inputInfo.keys}, Output: ${outputInfo.keys}")
        Log.d(TAG, "Size: ${modelSizeMb}MB, Load time: ${loadTimeMs}ms")

        return loadTimeMs
    }

    /**
     * Run inference on an image URI.
     * Returns InferenceResult with embedding and timing.
     */
    fun runInference(imageUri: Uri): InferenceResult {
        check(isLoaded) { "Model not loaded. Call loadModel() first." }

        // Decode and preprocess image
        val bitmap = decodeBitmap(imageUri)
        val inputTensor = preprocessImage(bitmap)

        // Get memory before inference
        val memBefore = Debug.getNativeHeapAllocatedSize()

        // Run inference
        val startTime = System.nanoTime()

        val inputName = session!!.inputInfo.keys.first()
        val shape = longArrayOf(1, 3, IMAGE_SIZE.toLong(), IMAGE_SIZE.toLong())
        val onnxTensor = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(inputTensor), shape)

        val results = session!!.run(mapOf(inputName to onnxTensor))
        val output = results[0].value as Array<FloatArray>
        val embedding = output[0]

        val inferenceTimeMs = (System.nanoTime() - startTime) / 1_000_000f
        val memAfter = Debug.getNativeHeapAllocatedSize()
        val memUsedMb = (memAfter - memBefore) / (1024f * 1024f)

        onnxTensor.close()
        results.close()

        Log.d(TAG, "Inference: ${inferenceTimeMs}ms, Embedding dim: ${embedding.size}")

        return InferenceResult(
            embedding = embedding,
            inferenceTimeMs = inferenceTimeMs,
            memoryUsedMb = memUsedMb,
            embeddingDim = embedding.size
        )
    }

    /**
     * Run inference on a random input (for benchmarking without real images).
     */
    fun runBenchmark(runs: Int = 10): BenchmarkResult {
        check(isLoaded) { "Model not loaded. Call loadModel() first." }

        val inputName = session!!.inputInfo.keys.first()
        val shape = longArrayOf(1, 3, IMAGE_SIZE.toLong(), IMAGE_SIZE.toLong())

        // Random input
        val randomInput = FloatArray(3 * IMAGE_SIZE * IMAGE_SIZE) { (Math.random() * 2 - 1).toFloat() }

        // Warmup
        val warmupTensor = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(randomInput), shape)
        session!!.run(mapOf(inputName to warmupTensor)).close()
        warmupTensor.close()

        // Benchmark
        val times = mutableListOf<Float>()
        for (i in 0 until runs) {
            val tensor = OnnxTensor.createTensor(ortEnv, FloatBuffer.wrap(randomInput), shape)
            val start = System.nanoTime()
            session!!.run(mapOf(inputName to tensor)).close()
            times.add((System.nanoTime() - start) / 1_000_000f)
            tensor.close()
        }

        return BenchmarkResult(
            runs = runs,
            avgTimeMs = times.average().toFloat(),
            minTimeMs = times.min(),
            maxTimeMs = times.max(),
            modelSizeMb = modelSizeMb
        )
    }

    /**
     * Decode bitmap from URI, resize to 224x224.
     */
    private fun decodeBitmap(uri: Uri): Bitmap {
        val inputStream = context.contentResolver.openInputStream(uri)
        val bitmap = BitmapFactory.decodeStream(inputStream)
        inputStream?.close()
        return Bitmap.createScaledBitmap(bitmap, IMAGE_SIZE, IMAGE_SIZE, true)
    }

    /**
     * Preprocess image to NCHW float tensor with ImageNet normalization.
     */
    private fun preprocessImage(bitmap: Bitmap): FloatArray {
        val pixels = IntArray(IMAGE_SIZE * IMAGE_SIZE)
        bitmap.getPixels(pixels, 0, IMAGE_SIZE, 0, 0, IMAGE_SIZE, IMAGE_SIZE)

        val floatArray = FloatArray(3 * IMAGE_SIZE * IMAGE_SIZE)

        for (i in pixels.indices) {
            val pixel = pixels[i]
            val r = ((pixel shr 16 and 0xFF) / 255f - MEAN[0]) / STD[0]
            val g = ((pixel shr 8 and 0xFF) / 255f - MEAN[1]) / STD[1]
            val b = ((pixel and 0xFF) / 255f - MEAN[2]) / STD[2]

            // NCHW format: channel first
            floatArray[i] = r                                          // R channel
            floatArray[IMAGE_SIZE * IMAGE_SIZE + i] = g                // G channel
            floatArray[2 * IMAGE_SIZE * IMAGE_SIZE + i] = b            // B channel
        }

        return floatArray
    }

    fun release() {
        session?.close()
        ortEnv?.close()
        session = null
        ortEnv = null
        isLoaded = false
    }

    /**
     * Result of a single inference run.
     */
    data class InferenceResult(
        val embedding: FloatArray,
        val inferenceTimeMs: Float,
        val memoryUsedMb: Float,
        val embeddingDim: Int
    )

    /**
     * Result of a benchmark run.
     */
    data class BenchmarkResult(
        val runs: Int,
        val avgTimeMs: Float,
        val minTimeMs: Float,
        val maxTimeMs: Float,
        val modelSizeMb: Float
    )
}
