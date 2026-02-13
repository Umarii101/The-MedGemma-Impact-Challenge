# MedLens — Offline Medical AI Chat Assistant

> Production Android app combining BiomedCLIP vision analysis with MedGemma clinical reasoning, running entirely on-device.

## Overview

MedLens is the production evolution of the [Inference Test App/](../Inference%20Test%20App/) proof-of-concept. It provides a unified **chat interface** where healthcare workers can:

1. **Capture** a medical image (camera or gallery)
2. **Add context** ("Patient has had this rash for 3 days")
3. **Receive** an AI-generated clinical assessment — fully offline

The entire pipeline runs on-device with no internet required, protecting patient privacy.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     MedLens Android App                          │
│              Jetpack Compose · Single-Activity MVVM              │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────┐    ┌──────────────────┐    ┌──────────────────┐ │
│  │ Camera /   │───→│  BiomedCLIP      │───→│ MedicalClassifier│ │
│  │ Gallery    │    │  Vision Encoder   │    │ (Zero-Shot)      │ │
│  │ (CameraX)  │    │  ONNX INT8, 84MB │    │ 30 conditions    │ │
│  └────────────┘    │  512-dim embed    │    │ cosine similarity│ │
│                    │  ~126ms inference │    └────────┬─────────┘ │
│                    └──────────────────┘             │           │
│                                                      ▼           │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    MedGemma 4B-IT                           │ │
│  │              llama.cpp · Q4_K_S GGUF · 2.2 GB              │ │
│  │         Streaming generation · 7.8 tok/s · JNI bridge      │ │
│  │                                                              │ │
│  │  Input: System prompt + classification findings + user msg  │ │
│  │  Output: Streaming clinical assessment in chat bubbles      │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                    Chat Interface                           │ │
│  │  • Message bubbles with image previews                     │ │
│  │  • Classification badge (top findings)                     │ │
│  │  • Streaming text with animated dots                       │ │
│  │  • Suggestion chips for common queries                     │ │
│  │  • Camera / gallery / text input bar                       │ │
│  │  • Medical disclaimer                                      │ │
│  └────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

### Pipeline Flow

1. User picks/captures an image → `BiomedClipInference` extracts a 512-dim embedding (~126ms)
2. `MedicalClassifier` computes cosine similarity against 30 pre-computed reference embeddings → top-5 conditions with confidence scores
3. Classification results are formatted as text context and sent to `MedGemmaInference`
4. MedGemma generates a streaming clinical assessment using the Gemma 3 chat template
5. Tokens stream into the chat UI via polling (200ms interval)

### Zero-Shot Classification

The app ships with pre-computed BiomedCLIP text embeddings for **30 medical conditions** across 4 categories:

| Category | Example Conditions |
|----------|-------------------|
| Chest X-Ray | Pneumonia, Pleural Effusion, Cardiomegaly, Pneumothorax, Atelectasis |
| Dermatology | Melanoma, Psoriasis, Eczema, Basal Cell Carcinoma |
| Ophthalmology | Diabetic Retinopathy, Glaucoma, Macular Degeneration |
| General Pathology | Fracture, Mass/Tumor, Edema, Inflammation, Hemorrhage |

These embeddings were generated offline using `quantization/scripts/generate_expanded_embeddings.py` with the BiomedCLIP text encoder. At runtime, only cosine similarity is computed — no text encoder needed on device.

## On-Device Performance

**Device**: Realme GT Neo 6 (Snapdragon 8s Gen 3, 12 GB RAM, Android 14)

| Component | Metric | Value |
|-----------|--------|-------|
| BiomedCLIP INT8 | Model size | 83.6 MB |
| | Inference time | ~126 ms |
| | Embedding dim | 512 |
| MedGemma Q4_K_S | Model size | 2.2 GB |
| | Prompt processing | 32.8 tok/s |
| | Token generation | 7.8 tok/s |
| | Context window | 512 tokens |
| | Load time | 5–9 seconds |
| MedicalClassifier | Labels | 30 conditions |
| | Classify time | <1 ms |
| **End-to-end** | Image → first token | ~6–8 seconds |
| | Full response (~150 tokens) | ~25 seconds |

## Source Files

```
Medlens/
├── app/src/main/
│   ├── java/com/medgemma/edge/
│   │   ├── MainActivity.kt              # Entry point, permission handling, screen routing
│   │   ├── ChatViewModel.kt             # Central ViewModel: model loading, analysis pipeline,
│   │   │                                #   streaming generation, chat state management
│   │   ├── inference/
│   │   │   ├── BiomedClipInference.kt   # ONNX Runtime wrapper for BiomedCLIP vision encoder
│   │   │   ├── MedGemmaInference.kt     # JNI wrapper for llama.cpp + Gemma 3 chat template
│   │   │   └── MedicalClassifier.kt     # Zero-shot classifier (cosine similarity vs embeddings)
│   │   └── ui/
│   │       ├── ChatScreen.kt            # Chat UI: bubbles, input bar, suggestion chips, disclaimer
│   │       ├── CameraCapture.kt         # Full-screen CameraX with front/back toggle
│   │       └── theme/                   # Material 3 theming
│   │
│   ├── cpp/
│   │   ├── CMakeLists.txt               # llama.cpp build: -O3, ARM dotprod+i8mm, static linking
│   │   └── medgemma_jni.cpp             # C++ JNI bridge: init, load, generate, stream, stop
│   │
│   ├── assets/
│   │   ├── text_embeddings.json         # Pre-computed BiomedCLIP embeddings (30 labels, 512-dim)
│   │   └── label_categories.json        # Label → category mapping
│   │
│   ├── AndroidManifest.xml              # Permissions: Camera, Storage; largeHeap=true
│   └── res/                             # Drawables, strings (app_name="MedLens"), themes
│
├── app/build.gradle.kts                 # Dependencies: CameraX 1.3.4, ONNX Runtime 1.19.0,
│                                        #   Compose BOM 2024.09.00, Gson, Coil, Navigation
├── settings.gradle.kts
├── gradle.properties
└── .gitignore
```

## Build & Run
  I have provided the app apk in \Medlens\APK\app-debug.apk

### Prerequisites

- **Android Studio** Ladybug (2024.2+) with:
  - NDK 28.2.13676358
  - CMake 3.22.1
- **llama.cpp** source (cloned separately — not included in repo)

### Step 1: Clone llama.cpp

The CMakeLists.txt expects `llama_cpp_repo/` five directories up from `app/src/main/cpp/`. From the repo root:

```bash
# Clone llama.cpp adjacent to the medlens folder (at repo root level)
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git llama_cpp_repo
```

This places `llama_cpp_repo/` at the same level as `Medlens/`, `Inference Test App/`, etc.

### Step 2: Build

```bash
# Open Medlens/ in Android Studio → Build → assembleDebug
# Or from command line:
cd medlens
./gradlew assembleDebug
```

### Step 3: Deploy Models to Phone

Models are not included in the APK. Push them to the device:

```bash
# Create model directory
adb shell mkdir -p /storage/emulated/0/MedGemmaEdge/

# Push quantized models (download from Kaggle — see root README)
adb push biomedclip_vision_int8.onnx /storage/emulated/0/MedGemmaEdge/
adb push medgemma-4b-q4_k_s-final.gguf /storage/emulated/0/MedGemmaEdge/
```

### Step 4: Install & Launch

```bash
adb install app/build/outputs/apk/debug/app-debug.apk
```

1. Grant storage permission when prompted (required for model loading)
2. Models auto-load on launch (~5–9 seconds)
3. Use camera or gallery button to capture/select a medical image
4. Optionally type context before sending
5. View streaming AI assessment in the chat

## Key Technical Decisions

| Decision | Rationale |
|----------|-----------|
| **llama.cpp static linking** | Single `.so` with all llama/ggml code. No runtime dependency issues. |
| **`use_mmap=false`** | Sequential RAM load avoids page-fault thrashing (5× faster inference). |
| **`-O3` on ALL targets** | Gradle's `assembleDebug` defaults to `-O0`. Three-mechanism CMake approach ensures every ggml/llama source file gets `-O3`. See [DEPLOYMENT_TECHNICAL_REPORT.md](../Inference%20Test%20App/DEPLOYMENT_TECHNICAL_REPORT.md). |
| **ARM `armv8.2-a+dotprod+i8mm+fp16` globally** | Enables NEON dot-product and int8 matrix-multiply intrinsics for all quantized matmul operations. |
| **Zero-shot classification via text embeddings** | Avoids shipping a text encoder on device. Pre-compute once offline, classify with cosine similarity at runtime. |
| **Single ViewModel** | `ChatViewModel` orchestrates all models, manages chat state, and handles streaming — keeps the UI layer thin. |
| **Gemma 3 chat template** | Proper `<start_of_turn>user/model` formatting with `parse_special=true` in the tokenizer. System prompt injected as first user turn prefix. |

## Evolution from PoC

The [Inference Test App/](../Inference%20Test%20App/) proof-of-concept had a 2-tab layout with raw embeddings display and basic text generation. MedLens evolved this into:

- **Unified chat interface** replacing separate tabs
- **Combined BiomedCLIP → Classifier → MedGemma pipeline** (PoC had models running independently)
- **CameraX integration** with front/back camera toggle
- **Zero-shot medical classification** with 30-condition label set
- **Streaming generation** with animated chat bubbles
- **Proper Gemma 3 chat template** (PoC used raw tokenization)
- **Context-aware prompting** — classification findings embedded in the LLM prompt

See [Inference Test App/ROADMAP.md](../Inference%20Test%20App/ROADMAP.md) for the original planned evolution and what was achieved.

## ⚠️ Medical Disclaimer

MedLens is for **assistive and research purposes only**. It is not FDA approved and not a substitute for professional medical judgment. All outputs require validation by licensed healthcare providers.

---

*Built for the Kaggle MedGemma Impact Challenge — Edge AI*
