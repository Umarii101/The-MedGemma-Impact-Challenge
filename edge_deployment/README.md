# Edge Deployment

Quantized models optimized for mobile/edge deployment on Android devices.

## 📥 Model Downloads

Models are too large for Git. Download from Google Drive:

**[📁 Download Models (Google Drive)](https://drive.google.com/file/d/1JZmLMVmimPnL3tiSe0GkNxYENRP49qAP/view?usp=sharing)**

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
- **Inference**: ~100ms CPU, ~30-50ms with NNAPI

### MedGemma 4B (Q4_K_S)
- **Source**: google/medgemma-4b-it
- **Quantization**: Q4_K_S (4-bit k-quant small)
- **Original Size**: 8.6 GB → 2.2 GB (74% reduction)
- **Context**: 2048 tokens
- **Speed**: 9+ tok/s CPU, 15-30+ tok/s on mobile NPU

## Android Integration

### BiomedCLIP with ONNX Runtime
```gradle
implementation 'com.microsoft.onnxruntime:onnxruntime-android:1.16.0'
```

### MedGemma with llama.cpp
Use [llama.cpp Android bindings](https://github.com/ggerganov/llama.cpp) or MLC-LLM.

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
- CPU: 4x Cortex-A720 + 4x Cortex-A520
- GPU: Adreno 735
- NPU: Hexagon NPU
- RAM: 8-12 GB

**Expected Performance**:
- BiomedCLIP: 30-50ms inference (NNAPI)
- MedGemma: 15-30 tok/s (NPU-accelerated)
