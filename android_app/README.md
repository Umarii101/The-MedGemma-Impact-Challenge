# MedGemma Edge - Android Application

> Android demo app for on-device clinical AI inference

## 🚧 Status: In Development

This folder will contain the Android application that demonstrates:

1. **BiomedCLIP inference** — Medical image embedding via ONNX Runtime Mobile
2. **MedGemma inference** — Clinical text generation via llama.cpp

## Planned Architecture

```
android_app/
├── app/
│   ├── src/main/
│   │   ├── java/com/medgemma/edge/
│   │   │   ├── MainActivity.kt
│   │   │   ├── inference/
│   │   │   │   ├── BiomedClipInference.kt
│   │   │   │   └── MedGemmaInference.kt
│   │   │   ├── ui/
│   │   │   │   ├── CameraFragment.kt
│   │   │   │   └── ResultsFragment.kt
│   │   │   └── utils/
│   │   ├── assets/
│   │   │   ├── biomedclip_vision_int8.onnx
│   │   │   └── medgemma-4b-q4_k_s.gguf
│   │   └── res/
│   └── build.gradle
├── build.gradle
└── settings.gradle
```

## Dependencies

```gradle
// ONNX Runtime for BiomedCLIP
implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.16.0'

// llama.cpp for MedGemma (via JNI bindings)
implementation 'com.github.aspect-build:llama-cpp-android:...'
```

## Target Device

**Realme GT Neo 6** (Snapdragon 8s Gen 3)
- 8-12 GB RAM
- Hexagon NPU
- Android 14+

## Model Files

Models are not included in the repository (too large for Git).

Download from:
- `biomedclip_vision_int8.onnx` — 84 MB
- `medgemma-4b-q4_k_s.gguf` — 2.2 GB

Place in `app/src/main/assets/` before building.

## Build Instructions

```bash
# Open in Android Studio
# Sync Gradle
# Build > Make Project
# Run on device or emulator
```

## Minimum Requirements

- Android SDK 24+ (Android 7.0)
- 8 GB device RAM
- 3 GB storage for models
- Camera permission (for X-ray capture)
