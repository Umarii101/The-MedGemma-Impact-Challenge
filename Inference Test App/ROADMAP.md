# Roadmap — MedLens Android App

> **Note:** This roadmap was written for the **proof-of-concept** test app (`android_app/`).
> Many Phase 2 and Phase 4 goals have been **achieved** in the production app — see [`Medlens/README.md`](../Medlens/README.md).

> Current status: **Proof-of-concept test app** — superseded by the production MedLens app.

## Current State (v0.1 — Test App)

- [x] BiomedCLIP INT8 ONNX inference — 126 ms avg, 83.6 MB
- [x] MedGemma Q4_K_S generation — 7.8 tok/s, 2.2 GB
- [x] Streaming text output with live tok/s display
- [x] Stop generation support
- [x] Permission handling for external storage
- [x] Two-tab UI (BiomedCLIP image analysis, MedGemma text generation)

## Phase 2 — Combined Pipeline  ✅ *Achieved in MedLens*

The core use case: **health worker photographs a medical image → BiomedCLIP extracts visual features → MedGemma generates clinical assessment**.

### 2.1 Unified Workflow  ✅
- ~~Single screen: capture/pick image → auto-analyze → show clinical summary~~ **Done** — unified chat interface
- ~~Pass BiomedCLIP embedding as context to MedGemma prompt~~ **Done** — zero-shot classification via cosine similarity (30 conditions, 512-dim embeddings)
- ~~Template: "Given this medical image analysis [embedding summary], provide clinical assessment for: [user question]"~~ **Done** — MedicalClassifier builds structured prompts from classification results

### 2.2 Prompt Engineering  ✅
- ~~System prompt optimized for clinical decision support~~ **Done**
- ~~Structured output format (Findings, Assessment, Recommendations)~~ **Done**
- ~~Confidence indicators and limitation disclaimers~~ **Done** — confidence scores + mandatory disclaimers

### 2.3 Chat Interface  ✅
- ~~Multi-turn conversation with KV cache persistence~~ **Done**
- ~~Context window management (sliding window or context shifting)~~ **Done** — 2048 token context
- ~~Conversation history with copy/share support~~ **Done** — scrollable chat with copy

## Phase 3 — Performance Optimization

### 3.1 GPU Acceleration (Vulkan/OpenCL)
- Adreno 735 GPU is currently **idle** — all inference is CPU-only
- llama.cpp supports `GGML_VULKAN` for Vulkan compute shaders
- Also has `GGML_OPENCL` with Adreno-optimized kernels (`GGML_OPENCL_USE_ADRENO_KERNELS`)
- Expected improvement: 2–4× on token generation

### 3.2 KleidIAI ARM Kernels
- Currently disabled due to spaces-in-path build issue
- KleidIAI provides Arm-optimized matmul kernels specifically for quantized models
- Fix: build in a path without spaces, or patch FetchContent
- Expected improvement: 10–30% on quantized matmul

### 3.3 Smaller Quantizations
- Q3_K_S or IQ4_XS could reduce model from 2.2 GB to ~1.6–1.8 GB
- Lower memory pressure → more RAM for KV cache → larger context
- Trade-off: slight quality degradation on medical reasoning

### 3.4 Context & Memory
- Current: 512-token context (minimal for demo)
- Target: 2048–4096 tokens for multi-turn clinical conversations
- Requires careful KV cache memory budgeting (~200 MB per 1024 tokens)

### 3.5 Thread Tuning
- Current: 4 threads (big cores only)
- Test: 2 threads (X4 + 1×A720) for lower power at similar speed
- Test: 6 threads (all big + some little) for max throughput
- Profile thermal throttling under sustained generation

## Phase 4 — Production Polish  (partial ✅ in MedLens)

### 4.1 UI/UX
- ~~Material Design 3 theming~~ ✅ **Done** — Material 3 + Compose
- ~~Camera integration (CameraX) for direct image capture~~ ✅ **Done** — CameraX 1.3.4
- ~~Offline-first architecture with graceful degradation~~ ✅ **Done** — fully offline
- Accessibility: large text, high contrast, screen reader support — *future*

### 4.2 Model Management
- Download models from within the app (avoid manual adb push) — *future*
- Model integrity verification (SHA256 checksums) — *future*
- Storage space check before download — *future*
- Model versioning and update support — *future*

### 4.3 Security & Privacy  ✅
- ~~All processing on-device (no network calls)~~ ✅ **Done**
- ~~No patient data leaves the device~~ ✅ **Done**
- Optional: encrypted model storage — *future*
- Audit logging for clinical use — *future*

### 4.4 NNAPI / QNN Delegation
- BiomedCLIP: ONNX Runtime NNAPI execution provider for Hexagon NPU
- Expected: 2–5× speedup on image embedding extraction
- Requires NNAPI-compatible operator coverage testing

## Performance Targets

| Metric | Current (v0.1) | Target (v1.0) |
|--------|---------------|---------------|
| BiomedCLIP inference | 126 ms | <50 ms (NNAPI) |
| MedGemma prompt processing | 32.8 tok/s | 50+ tok/s (GPU) |
| MedGemma generation | 7.8 tok/s | 15+ tok/s (GPU) |
| Model load time | 5–9 s | <5 s |
| Context window | 512 tokens | 2048+ tokens |
| APK size | 57 MB | <40 MB (single ABI) |
| End-to-end (image → assessment) | N/A | <20 s |

## Known Limitations

1. **Memory**: 2.2 GB model + KV cache + OS leaves ~2 GB free on 12 GB device. Devices with 8 GB RAM may struggle.
2. **No GPU offload**: Adreno 735 GPU is unused. Vulkan backend would help significantly.
3. ~~**No multi-turn**: Each generation is independent — no conversation memory yet.~~ ✅ **Fixed in MedLens**
4. **Model download**: Users must manually push models via adb or file manager.
5. **Single device tested**: Only validated on Realme GT Neo 6 (Snapdragon 8s Gen 3).

---

> **See [`Medlens/README.md`](../Medlens/README.md) for the production app that builds on this PoC.**
