# Edge Deployment

Quantized models optimized for mobile/edge deployment on Android devices.

## 📥 Model Downloads

The Models that I have quantized are too large for Git. Download from Kaggle:

**[📁 Download Models (Kaggle)](https://www.kaggle.com)**
- [MedGemma 4B Q4_K_S GGUF](https://www.kaggle.com/models/muhammadumar2001/medgemma-4b-q4-k-s-gguf)
- [BiomedCLIP Vision INT8 ONNX](https://www.kaggle.com/models/muhammadumar2001/biomedclip-vision-int8-onnx)


| File | Size | Description |
|------|------|-------------|
| `biomedclip_vision_int8.onnx` | 84 MB | Production model |
| `biomedclip_vision.onnx` | 329 MB | FP32 baseline (optional) |
| `medgemma-4b-q4_k_s-final.gguf` | 2.2 GB | Quantized LLM |

After download, place files in:
```
edge_deployment/models/biomedclip/biomedclip_vision_int8.onnx
edge_deployment/models/biomedclip/biomedclip_vision.onnx
edge_deployment/models/medgemma/medgemma-4b-q4_k_s-final.gguf
```

## Models

| Model | Format | Size | Quantization | Target |
|-------|--------|------|--------------|--------|
| **BiomedCLIP Vision** | ONNX INT8 | 84 MB | Dynamic INT8 | Image embeddings |
| **BiomedCLIP Vision** | ONNX FP32 | 329 MB | None (baseline) | Reference |
| **MedGemma 4B** | GGUF Q4_K_S | 2.2 GB | 4-bit k-quant | Text generation |

## Quick Start

```bash
# Verify models are in place
python tests/test_biomedclip.py
python tests/test_medgemma.py

# Run all tests
python tests/run_all_tests.py
```

## Model Details

### BiomedCLIP Vision (INT8)
- **Source**: microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
- **Architecture**: ViT-B/16
- **Input**: 224x224 RGB image (NCHW format)
- **Output**: 512-dim embedding vector
- **Accuracy**: 99.95% cosine similarity vs FP32
- **Inference**: ~126 ms on-device CPU (Snapdragon 8s Gen 3)

### MedGemma 4B (Q4_K_S)
- **Source**: google/medgemma-4b-it
- **Quantization**: Q4_K_S (4-bit k-quant small)
- **Original Size**: 8.6 GB → 2.2 GB (74% reduction)
- **Context**: 512 tokens (configurable up to 2048)
- **Speed**: 7.8 tok/s generation, 32.8 tok/s prompt processing (CPU, Snapdragon 8s Gen 3)

## Android Integration

### BiomedCLIP with ONNX Runtime
```gradle
implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.19.0'
```

### MedGemma with llama.cpp
Compiled from source as a static library via CMake `add_subdirectory()` with a C++ JNI bridge. See [Medlens/README.md](../Medlens/README.md) for build instructions and [Inference Test App/DEPLOYMENT_TECHNICAL_REPORT.md](../Inference%20Test%20App/DEPLOYMENT_TECHNICAL_REPORT.md) for the optimization story.

## File Structure

```
edge_deployment/
├── models/
│   ├── biomedclip/
│   │   ├── biomedclip_vision_int8.onnx  # Production (84 MB)
│   │   └── biomedclip_vision.onnx       # Baseline (329 MB)
│   └── medgemma/
│       └── medgemma-4b-q4_k_s-final.gguf
└── README.md
```

## Hardware Requirements

**Target Device**: Realme GT Neo 6 (Snapdragon 8s Gen 3)
- CPU: 1×Cortex-X4 (3.0 GHz) + 4×Cortex-A720 (2.8 GHz) + 3×Cortex-A520 (2.0 GHz)
- GPU: Adreno 735
- NPU: Hexagon NPU
- RAM: 12 GB LPDDR5X

**Measured On-Device Performance**:

| Model | Metric | Value |
|-------|--------|-------|
| BiomedCLIP INT8 | Inference | ~126 ms (CPU) |
| MedGemma Q4_K_S | Prompt processing | 32.8 tok/s (CPU) |
| MedGemma Q4_K_S | Token generation | 7.8 tok/s (CPU) |
| MedGemma Q4_K_S | Model load | 5–9 seconds |

**Future Targets** (GPU/NPU acceleration not yet implemented):
- BiomedCLIP: ~30–50 ms with NNAPI
- MedGemma: 15–30 tok/s with Vulkan/OpenCL GPU offload
