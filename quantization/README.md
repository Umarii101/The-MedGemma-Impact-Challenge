# Quantization Scripts & Methodology

This folder contains the scripts used to quantize HAI-DEF models for edge deployment.

## Overview

| Model | Script | Input | Output | Method |
|-------|--------|-------|--------|--------|
| BiomedCLIP | `quantize_int8.py` | PyTorch model | ONNX INT8 | Dynamic quantization |
| BiomedCLIP | `quantize_onnx_int8.py` | ONNX FP32 | ONNX INT8 | ONNX Runtime quantization |
| MedGemma | `quantize_gguf.py` | GGUF FP16 | GGUF Q4_K_S | llama.cpp quantization |

## BiomedCLIP INT8 Quantization

### Why INT8?
- **4x smaller** than FP32 (329 MB → 84 MB)
- **Near-lossless**: 99.95% cosine similarity vs FP32
- **Hardware accelerated** via NNAPI on Android

### Process

```bash
# Step 1: Export to ONNX FP32
python quantize_int8.py --export-only

# Step 2: Quantize to INT8
python quantize_onnx_int8.py
```

### Key Code

```python
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="biomedclip_vision.onnx",
    model_output="biomedclip_vision_int8.onnx",
    weight_type=QuantType.QUInt8,
    optimize_model=True
)
```

### Results

| Metric | FP32 | INT8 | Delta |
|--------|------|------|-------|
| Size | 329 MB | 84 MB | -74% |
| Inference (CPU) | ~120ms | ~100ms | -17% |
| Cosine Similarity | 1.0 | 0.9995 | -0.05% |

## MedGemma Q4_K_S Quantization

### Why Q4_K_S?
- **74% size reduction** (8.6 GB → 2.2 GB)
- **Optimized for mobile**: Fits in smartphone RAM
- **Quality preserved**: K-quant maintains important weights

### Process

```bash
# Step 1: Convert HuggingFace to GGUF FP16
python llama.cpp/convert_hf_to_gguf.py \
    --outfile medgemma-4b-f16.gguf \
    --outtype f16 \
    google/medgemma-4b-it

# Step 2: Quantize to Q4_K_S
./llama.cpp/llama-quantize \
    medgemma-4b-f16.gguf \
    medgemma-4b-q4_k_s.gguf \
    Q4_K_S
```

### Quantization Options Tested

| Method | Size | Quality | Notes |
|--------|------|---------|-------|
| Q4_0 | 2.0 GB | Lower | Fastest, some quality loss |
| **Q4_K_S** | 2.2 GB | Good | ✅ Best balance |
| Q4_K_M | 2.4 GB | Better | Slightly larger |
| Q5_K_S | 2.7 GB | High | May not fit all devices |
| Q8_0 | 4.3 GB | Highest | Too large for mobile |

### Results

| Metric | FP16 | Q4_K_S | Delta |
|--------|------|--------|-------|
| Size | 8.6 GB | 2.2 GB | -74% |
| Load Time | ~30s | ~2s | -93% |
| Speed (CPU) | 3 tok/s | 9 tok/s | +200% |

## Validation

Both quantized models are validated by test scripts:

```bash
cd .. 
python tests/run_all_tests.py
```

Expected output:
```
[PASS] BiomedCLIP INT8 - 99.95% accuracy
[PASS] MedGemma Q4_K_S - 9+ tok/s
ALL TESTS PASSED
```

## File Locations

After quantization, models should be placed in:

```
edge_deployment/
├── models/
│   ├── biomedclip/
│   │   ├── biomedclip_vision_int8.onnx  # Production
│   │   └── biomedclip_vision.onnx       # Baseline (optional)
│   └── medgemma/
│       └── medgemma-4b-q4_k_s.gguf
```

## Dependencies

```bash
pip install onnx onnxruntime llama-cpp-python gguf
```

For llama.cpp quantization, clone and build:
```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make
```
