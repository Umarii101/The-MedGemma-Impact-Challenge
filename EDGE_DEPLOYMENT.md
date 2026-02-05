# Edge AI Deployment for Healthcare

> Bringing MedGemma and BiomedCLIP to mobile devices for offline clinical decision support in low-resource settings.

## 🎯 The Challenge

Healthcare workers in rural clinics, field hospitals, and developing regions often lack reliable internet access. They need AI-assisted tools that:

- **Run offline** on available hardware
- **Protect patient privacy** with on-device processing
- **Deliver fast responses** for time-critical decisions
- **Fit resource constraints** of mobile devices

## 🏆 Our Solution

We adapted Google's HAI-DEF models for edge deployment on Android devices:

| Model | Original | Quantized | Reduction | Use Case |
|-------|----------|-----------|-----------|----------|
| **MedGemma 4B** | 8.6 GB | 2.2 GB | **74%** | Clinical text reasoning |
| **BiomedCLIP** | 329 MB | 84 MB | **74%** | Medical image embeddings |

**Target Device**: Snapdragon 8s Gen 3 (Realme GT Neo 6)
- 8-12 GB RAM
- Hexagon NPU for accelerated inference
- Representative of mid-range smartphones in emerging markets

## 📊 Quantization Results

### MedGemma 4B-IT → Q4_K_S GGUF

| Metric | Value |
|--------|-------|
| Original Size | 8.6 GB (FP16) |
| Quantized Size | 2.2 GB |
| Quantization | Q4_K_S (4-bit k-quant) |
| Inference Speed (CPU) | 9+ tok/s |
| Context Length | 2048 tokens |

**Why Q4_K_S?**
- Best balance of size vs quality for mobile
- Maintains medical reasoning capability
- Compatible with llama.cpp ecosystem

### BiomedCLIP Vision → ONNX INT8

| Metric | Value |
|--------|-------|
| Original Size | 329 MB (FP32) |
| Quantized Size | 84 MB |
| Quantization | Dynamic INT8 |
| Accuracy vs FP32 | **99.95%** cosine similarity |
| Inference Speed (CPU) | ~100ms |

**Why INT8 ONNX?**
- ONNX Runtime Mobile has excellent Android support
- NNAPI delegation for hardware acceleration
- Near-lossless accuracy preservation

## 🔬 Technical Approach

### MedGemma Quantization

```
medgemma-4b-it (HF Safetensors)
    ↓ llama.cpp conversion
medgemma-4b-f16.gguf (8.6 GB)
    ↓ Q4_K_S quantization
medgemma-4b-q4_k_s.gguf (2.2 GB)
```

Key decisions:
1. Used llama.cpp's `convert_hf_to_gguf.py` for Gemma 3 architecture
2. Selected Q4_K_S over Q4_0 for better quality on medical text
3. Validated with clinical prompts before/after quantization

### BiomedCLIP Quantization

```
BiomedCLIP (PyTorch)
    ↓ ONNX export (opset 17)
biomedclip_vision.onnx (329 MB)
    ↓ Dynamic INT8 quantization
biomedclip_vision_int8.onnx (84 MB)
```

Key decisions:
1. Vision encoder only (text encoder handled by MedGemma)
2. Dynamic quantization preserves accuracy without calibration data
3. Validated with synthetic X-ray images

## 📱 Android Integration

### BiomedCLIP with ONNX Runtime Mobile

```kotlin
// Load quantized model
val session = OrtEnvironment.getEnvironment()
    .createSession(assets.open("biomedclip_vision_int8.onnx"))

// Enable NNAPI for hardware acceleration
val options = OrtSession.SessionOptions()
options.addNnapi()
```

### MedGemma with llama.cpp

```kotlin
// Using llama.cpp Android bindings
val model = LlamaModel.load("medgemma-4b-q4_k_s.gguf")
model.setContextSize(2048)
model.setThreads(4)  // Optimize for device
```

## ✅ Validation

All models validated with automated tests:

```bash
python tests/run_all_tests.py
```

```
[PASS] BiomedCLIP INT8 - 99.95% accuracy, 84 MB, ~100ms
[PASS] MedGemma Q4_K_S - 2.2 GB, 9+ tok/s
ALL TESTS PASSED
```

## 🌍 Real-World Impact

### Use Case: Rural Health Clinic

**Scenario**: A community health worker in a remote village encounters a patient with respiratory symptoms and a concerning chest X-ray.

**Without Edge AI**:
- No internet → No AI assistance
- Must travel hours to reach specialist
- Delayed diagnosis risks patient outcome

**With Edge AI**:
1. Capture X-ray image with smartphone
2. BiomedCLIP extracts visual features (100ms)
3. MedGemma analyzes symptoms + image context (15-30s)
4. Receive structured assessment with recommendations
5. Make informed decision on referral urgency

**All processing happens on-device** — no internet required, no patient data leaves the device.

## 📁 Repository Structure

```
edge_deployment/
├── models/
│   ├── biomedclip/
│   │   ├── biomedclip_vision_int8.onnx  # 84 MB (production)
│   │   └── biomedclip_vision.onnx       # 329 MB (baseline)
│   └── medgemma/
│       └── medgemma-4b-q4_k_s.gguf      # 2.2 GB
├── scripts/
└── README.md

quantization/
├── scripts/
│   ├── quantize_int8.py        # BiomedCLIP INT8
│   ├── quantize_onnx_int8.py   # ONNX quantization
│   └── quantize_gguf.py        # MedGemma GGUF
├── results/
└── README.md

benchmarks/
├── desktop_benchmarks.md
└── device_benchmarks.md        # On-device measurements

tests/
├── test_biomedclip.py
├── test_medgemma.py
└── run_all_tests.py
```

## 🔗 Links

- **Model Source**: [Google HAI-DEF](https://huggingface.co/google/medgemma-4b-it)
- **BiomedCLIP**: [microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
- **Quantization Tools**: llama.cpp, ONNX Runtime

---

*Built for the Kaggle MedGemma Impact Challenge — Edge AI Prize*
