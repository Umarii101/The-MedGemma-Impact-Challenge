package com.medgemma.edge

import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.Settings
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.medgemma.edge.inference.BiomedClipInference
import com.medgemma.edge.inference.MedGemmaInference
import com.medgemma.edge.ui.theme.MedGemmaEdgeTheme
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            MedGemmaEdgeTheme {
                MedGemmaEdgeApp()
            }
        }
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MedGemmaEdgeApp() {
    val context = LocalContext.current
    var selectedTab by remember { mutableIntStateOf(0) }
    val tabs = listOf("BiomedCLIP", "MedGemma")

    // Track storage permission state; re-check when app resumes
    var hasStoragePermission by remember {
        mutableStateOf(
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                Environment.isExternalStorageManager()
            } else true
        )
    }

    // Re-check permission when lifecycle resumes (user may grant in Settings and come back)
    val lifecycleOwner = androidx.lifecycle.compose.LocalLifecycleOwner.current
    DisposableEffect(lifecycleOwner) {
        val observer = androidx.lifecycle.LifecycleEventObserver { _, event ->
            if (event == androidx.lifecycle.Lifecycle.Event.ON_RESUME) {
                hasStoragePermission = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                    Environment.isExternalStorageManager()
                } else true
            }
        }
        lifecycleOwner.lifecycle.addObserver(observer)
        onDispose { lifecycleOwner.lifecycle.removeObserver(observer) }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("MedGemma Edge", fontWeight = FontWeight.Bold) },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer
                )
            )
        }
    ) { innerPadding ->
        Column(modifier = Modifier.padding(innerPadding)) {
            // Permission banner (Android 11+)
            if (!hasStoragePermission) {
                Card(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(12.dp),
                    colors = CardDefaults.cardColors(
                        containerColor = MaterialTheme.colorScheme.errorContainer
                    )
                ) {
                    Column(modifier = Modifier.padding(16.dp)) {
                        Text(
                            "Storage Permission Required",
                            fontWeight = FontWeight.Bold,
                            color = MaterialTheme.colorScheme.onErrorContainer
                        )
                        Spacer(Modifier.height(4.dp))
                        Text(
                            "This app needs \"All Files Access\" to read model files (.onnx, .gguf) " +
                                "from /sdcard/MedGemmaEdge/. Tap below to grant in Settings.",
                            fontSize = 13.sp,
                            color = MaterialTheme.colorScheme.onErrorContainer
                        )
                        Spacer(Modifier.height(8.dp))
                        Button(
                            onClick = {
                                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                                    val intent = Intent(
                                        Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION,
                                        Uri.parse("package:${context.packageName}")
                                    )
                                    context.startActivity(intent)
                                }
                            },
                            colors = ButtonDefaults.buttonColors(
                                containerColor = MaterialTheme.colorScheme.error,
                                contentColor = MaterialTheme.colorScheme.onError
                            )
                        ) {
                            Text("Grant Storage Permission")
                        }
                    }
                }
            }

            TabRow(selectedTabIndex = selectedTab) {
                tabs.forEachIndexed { index, title ->
                    Tab(
                        selected = selectedTab == index,
                        onClick = { selectedTab = index },
                        text = { Text(title) }
                    )
                }
            }

            when (selectedTab) {
                0 -> BiomedClipScreen()
                1 -> MedGemmaScreen()
            }
        }
    }
}

// ============================================================
// BiomedCLIP Tab
// ============================================================
@Composable
fun BiomedClipScreen() {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    // State
    var modelPath by remember { mutableStateOf<String?>(null) }
    var isLoading by remember { mutableStateOf(false) }
    var statusText by remember { mutableStateOf("Model not loaded") }
    var resultsText by remember { mutableStateOf("") }
    var selectedImageUri by remember { mutableStateOf<Uri?>(null) }

    // Inference engine
    val inference = remember { BiomedClipInference(context) }

    // Auto-detect model path on first composition
    LaunchedEffect(Unit) {
        modelPath = BiomedClipInference.findModelPath(context)
        statusText = if (modelPath != null) {
            "Model found: ${modelPath!!.substringAfterLast("/")}"
        } else {
            "Model not found. Place biomedclip_vision_int8.onnx in:\n" +
                "  /sdcard/MedGemmaEdge/\n" +
                "  /sdcard/Download/\n" +
                "  App storage/models/"
        }
    }

    // Image picker launcher
    val imagePickerLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri -> selectedImageUri = uri }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
            .verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        // Status card
        CopyableCard(title = "Status", text = statusText)

        // Load + Select Image row
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Button(
                onClick = {
                    if (modelPath == null) {
                        Toast.makeText(context, "Model file not found", Toast.LENGTH_SHORT).show()
                        return@Button
                    }
                    scope.launch {
                        isLoading = true
                        statusText = "Loading model..."
                        try {
                            val loadTime = withContext(Dispatchers.IO) {
                                inference.loadModel(modelPath!!)
                            }
                            statusText = "Model loaded\n" +
                                "  Size: ${"%.1f".format(inference.modelSizeMb)} MB\n" +
                                "  Load time: ${loadTime} ms"
                        } catch (e: Exception) {
                            statusText = "Load failed: ${e.message}"
                        }
                        isLoading = false
                    }
                },
                enabled = !isLoading && modelPath != null && !inference.isLoaded,
                modifier = Modifier.weight(1f)
            ) { Text("Load Model") }

            Button(
                onClick = { imagePickerLauncher.launch("image/*") },
                enabled = inference.isLoaded && !isLoading,
                modifier = Modifier.weight(1f)
            ) { Text("Select Image") }
        }

        // Inference + Benchmark row
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Button(
                onClick = {
                    if (selectedImageUri == null) {
                        Toast.makeText(context, "Select an image first", Toast.LENGTH_SHORT).show()
                        return@Button
                    }
                    scope.launch {
                        isLoading = true
                        resultsText = "Running inference..."
                        try {
                            val result = withContext(Dispatchers.IO) {
                                inference.runInference(selectedImageUri!!)
                            }
                            val embPreview = result.embedding.take(8)
                                .joinToString(", ") { "%.4f".format(it) }
                            val norm = Math.sqrt(
                                result.embedding.sumOf { it.toDouble() * it.toDouble() }
                            )
                            resultsText = buildString {
                                appendLine("=== INFERENCE RESULT ===")
                                appendLine("Time: ${"%.2f".format(result.inferenceTimeMs)} ms")
                                appendLine("Memory: ${"%.1f".format(result.memoryUsedMb)} MB")
                                appendLine("Embedding dim: ${result.embeddingDim}")
                                appendLine()
                                appendLine("First 8 values:")
                                appendLine("[$embPreview, ...]")
                                appendLine()
                                appendLine("Min: ${"%.4f".format(result.embedding.min())}")
                                appendLine("Max: ${"%.4f".format(result.embedding.max())}")
                                append("Norm: ${"%.4f".format(norm)}")
                            }
                        } catch (e: Exception) {
                            resultsText = "Inference failed: ${e.message}"
                        }
                        isLoading = false
                    }
                },
                enabled = inference.isLoaded && !isLoading && selectedImageUri != null,
                modifier = Modifier.weight(1f)
            ) { Text("Run Inference") }

            Button(
                onClick = {
                    scope.launch {
                        isLoading = true
                        resultsText = "Running benchmark (10 runs)..."
                        try {
                            val result = withContext(Dispatchers.IO) {
                                inference.runBenchmark(runs = 10)
                            }
                            resultsText = buildString {
                                appendLine("=== BENCHMARK ===")
                                appendLine("Runs: ${result.runs}")
                                appendLine("Avg: ${"%.2f".format(result.avgTimeMs)} ms")
                                appendLine("Min: ${"%.2f".format(result.minTimeMs)} ms")
                                appendLine("Max: ${"%.2f".format(result.maxTimeMs)} ms")
                                appendLine("Model: ${"%.1f".format(result.modelSizeMb)} MB")
                                appendLine()
                                appendLine("Device: ${android.os.Build.MODEL}")
                                append("SoC: ${android.os.Build.HARDWARE}")
                            }
                        } catch (e: Exception) {
                            resultsText = "Benchmark failed: ${e.message}"
                        }
                        isLoading = false
                    }
                },
                enabled = inference.isLoaded && !isLoading,
                modifier = Modifier.weight(1f)
            ) { Text("Benchmark x10") }
        }

        // Progress indicator
        if (isLoading) {
            LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
        }

        // Selected image indicator
        if (selectedImageUri != null) {
            Text(
                "Image: ${selectedImageUri.toString().substringAfterLast("/")}",
                fontSize = 12.sp,
                color = MaterialTheme.colorScheme.outline
            )
        }

        // Results card
        if (resultsText.isNotEmpty()) {
            CopyableCard(
                title = "Results",
                text = resultsText,
                containerColor = MaterialTheme.colorScheme.surfaceVariant
            )
        }
    }

    // Release resources on disposal
    DisposableEffect(Unit) {
        onDispose { inference.release() }
    }
}

// ============================================================
// MedGemma Tab  (streaming generation + stop button)
// ============================================================
@Composable
fun MedGemmaScreen() {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()

    // State
    var modelPath by remember { mutableStateOf<String?>(null) }
    var isLoading by remember { mutableStateOf(false) }
    var isGenerating by remember { mutableStateOf(false) }
    var statusText by remember { mutableStateOf("Model not loaded") }
    var resultsText by remember { mutableStateOf("") }
    var streamingHeader by remember { mutableStateOf("") }  // "12 tok | 8.3 tok/s"
    var promptText by remember {
        mutableStateOf("You are a medical AI assistant. Briefly describe the key radiological features of pneumonia on a chest X-ray.")
    }

    // Inference engine
    val inference = remember { MedGemmaInference(context) }

    // Auto-detect model path
    LaunchedEffect(Unit) {
        modelPath = MedGemmaInference.findModelPath(context)
        statusText = if (modelPath != null) {
            "Model found: ${modelPath!!.substringAfterLast("/")}"
        } else {
            "Model not found. Place medgemma-4b-q4_k_s-final.gguf in:\n" +
                "  /sdcard/MedGemmaEdge/\n" +
                "  /sdcard/Download/\n" +
                "  App storage/models/"
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
            .verticalScroll(rememberScrollState()),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        // Status
        CopyableCard(title = "Status", text = statusText)

        // Load Model
        Button(
            onClick = {
                if (modelPath == null) {
                    Toast.makeText(context, "Model file not found", Toast.LENGTH_SHORT).show()
                    return@Button
                }
                scope.launch {
                    isLoading = true
                    statusText = "Loading 2.2GB model into RAM (15-30s)..."
                    try {
                        val loadTime = withContext(Dispatchers.IO) {
                            inference.loadModel(modelPath!!)
                        }
                        val sysInfo = try {
                            withContext(Dispatchers.IO) { inference.systemInfo() }
                        } catch (_: Exception) { "unavailable" }

                        // Get available RAM
                        val activityManager = context.getSystemService(android.app.ActivityManager::class.java)
                        val memInfo = android.app.ActivityManager.MemoryInfo()
                        activityManager.getMemoryInfo(memInfo)
                        val availMb = memInfo.availMem / (1024 * 1024)
                        val totalMb = memInfo.totalMem / (1024 * 1024)

                        statusText = "Model loaded\n" +
                            "  Size: ${"%.1f".format(inference.modelSizeMb)} MB\n" +
                            "  Load time: ${loadTime} ms\n" +
                            "  Context: 1024 | Threads: 4\n" +
                            "  RAM: ${availMb}/${totalMb} MB free\n" +
                            "  CPU: $sysInfo"
                    } catch (e: Exception) {
                        statusText = "Load failed: ${e.message}"
                    }
                    isLoading = false
                }
            },
            enabled = !isLoading && !isGenerating && modelPath != null && !inference.isLoaded,
            modifier = Modifier.fillMaxWidth()
        ) { Text("Load Model") }

        // Prompt input
        OutlinedTextField(
            value = promptText,
            onValueChange = { promptText = it },
            label = { Text("Prompt") },
            modifier = Modifier.fillMaxWidth(),
            minLines = 3,
            maxLines = 6,
            enabled = inference.isLoaded && !isLoading && !isGenerating
        )

        // Generate + Stop + Benchmark row
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Button(
                onClick = {
                    scope.launch {
                        isLoading = true
                        isGenerating = true
                        resultsText = "Processing prompt..."
                        streamingHeader = ""

                        // Launch polling coroutine for streaming updates
                        val pollingJob = launch {
                            kotlinx.coroutines.delay(500) // initial delay for prompt eval
                            while (isGenerating) {
                                try {
                                    val state = withContext(Dispatchers.IO) {
                                        inference.getPartialResult()
                                    }
                                    if (state.tokensGenerated > 0) {
                                        streamingHeader = "${state.tokensGenerated} tok | ${"%.1f".format(state.tokPerSec)} tok/s"
                                        resultsText = state.text
                                    }
                                } catch (_: Exception) {}
                                kotlinx.coroutines.delay(200)
                            }
                        }

                        try {
                            val finalText = withContext(Dispatchers.IO) {
                                inference.generate(promptText, maxTokens = 256)
                            }
                            // finalText has stats header from C++
                            resultsText = finalText
                            streamingHeader = ""
                        } catch (e: Exception) {
                            resultsText = "Generation failed: ${e.message}"
                        }

                        isGenerating = false
                        isLoading = false
                        pollingJob.cancel()
                    }
                },
                enabled = inference.isLoaded && !isLoading && !isGenerating,
                modifier = Modifier.weight(1f)
            ) { Text("Generate") }

            // Stop button (only active while generating)
            Button(
                onClick = {
                    inference.stopGeneration()
                    Toast.makeText(context, "Stopping...", Toast.LENGTH_SHORT).show()
                },
                enabled = isGenerating,
                colors = ButtonDefaults.buttonColors(
                    containerColor = MaterialTheme.colorScheme.error,
                    contentColor = MaterialTheme.colorScheme.onError
                ),
                modifier = Modifier.weight(0.6f)
            ) { Text("Stop") }

            Button(
                onClick = {
                    scope.launch {
                        isLoading = true
                        resultsText = "Running benchmark (pp=512, tg=128, 3 reps)..."
                        try {
                            val result = withContext(Dispatchers.IO) {
                                inference.benchmark(pp = 512, tg = 128, reps = 3)
                            }
                            resultsText = result
                        } catch (e: Exception) {
                            resultsText = "Benchmark failed: ${e.message}"
                        }
                        isLoading = false
                    }
                },
                enabled = inference.isLoaded && !isLoading && !isGenerating,
                modifier = Modifier.weight(1f)
            ) { Text("Benchmark") }
        }

        // Progress + streaming speed
        if (isLoading) {
            LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
        }
        if (streamingHeader.isNotEmpty()) {
            Text(
                streamingHeader,
                fontSize = 13.sp,
                fontWeight = FontWeight.Bold,
                color = MaterialTheme.colorScheme.primary
            )
        }

        // Results
        if (resultsText.isNotEmpty()) {
            CopyableCard(
                title = if (isGenerating) "Generating..." else "Results",
                text = resultsText,
                containerColor = MaterialTheme.colorScheme.surfaceVariant
            )
        }
    }

    DisposableEffect(Unit) {
        onDispose { inference.release() }
    }
}

// ============================================================
// Reusable: Card with selectable text + Copy button
// ============================================================
@Composable
fun CopyableCard(
    title: String,
    text: String,
    modifier: Modifier = Modifier,
    containerColor: androidx.compose.ui.graphics.Color = MaterialTheme.colorScheme.surfaceContainerHighest
) {
    val clipboardManager = LocalClipboardManager.current
    val context = LocalContext.current

    Card(
        modifier = modifier.fillMaxWidth(),
        colors = CardDefaults.cardColors(containerColor = containerColor)
    ) {
        Column(modifier = Modifier.padding(12.dp)) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(title, fontWeight = FontWeight.Bold, fontSize = 16.sp)
                TextButton(
                    onClick = {
                        clipboardManager.setText(AnnotatedString(text))
                        Toast.makeText(context, "Copied to clipboard", Toast.LENGTH_SHORT).show()
                    },
                    contentPadding = PaddingValues(horizontal = 8.dp, vertical = 0.dp)
                ) {
                    Text("Copy", fontSize = 12.sp)
                }
            }
            Spacer(Modifier.height(4.dp))
            SelectionContainer {
                Text(text, fontSize = 13.sp, fontFamily = FontFamily.Monospace)
            }
        }
    }
}