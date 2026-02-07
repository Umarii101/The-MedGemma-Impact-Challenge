# Benchmark Results

Performance measurements for quantized edge deployment models.

## Desktop Benchmarks (Development Machine)

**Hardware**: Windows PC with NVIDIA GPU
**Runtime**: ONNX Runtime 1.23.2, llama-cpp-python 0.3.16

### BiomedCLIP INT8

| Metric | FP32 Baseline | INT8 Quantized | Change |
|--------|---------------|----------------|--------|
| Model Size | 329 MB | 84 MB | **-74%** |
| Load Time | ~500ms | ~200ms | -60% |
| Inference (CPU) | 117ms | 98ms | -16% |
| Inference (GPU) | 15ms | 12ms | -20% |
| Memory Usage | ~400 MB | ~120 MB | -70% |
| Output Accuracy | 1.000 | 0.9995 | **-0.05%** |

*Cosine similarity of 0.9995 indicates near-lossless quantization*

### MedGemma Q4_K_S

| Metric | FP16 (via transformers) | Q4_K_S GGUF | Change |
|--------|-------------------------|-------------|--------|
| Model Size | 8.6 GB | 2.2 GB | **-74%** |
| Load Time | ~30s | 2.2s | -93% |
| Speed (CPU, 4 threads) | ~3 tok/s | 9.0 tok/s | **+200%** |
| RAM Usage | ~10 GB | ~3 GB | -70% |
| Context Length | 131K | 2048 | Configured |

*Speed improvement due to optimized GGUF inference in llama.cpp*

## Target Device Specifications

**Device**: Realme GT Neo 6
**SoC**: Snapdragon 8s Gen 3

| Component | Specification |
|-----------|---------------|
| CPU | 4x Cortex-A720 @ 3.0GHz + 4x Cortex-A520 @ 2.0GHz |
| GPU | Adreno 735 |
| NPU | Hexagon NPU |
| RAM | 12 GB LPDDR5X |
| Storage | UFS 4.0 |

### Expected Mobile Performance

Based on similar Snapdragon 8 Gen series devices:

| Model | Expected Performance | Notes |
|-------|---------------------|-------|
| BiomedCLIP INT8 | 30-50ms | With NNAPI acceleration |
| BiomedCLIP INT8 | 80-100ms | CPU only fallback |
| MedGemma Q4_K_S | 15-30 tok/s | With NPU offload |
| MedGemma Q4_K_S | 8-12 tok/s | CPU only |

### Memory Budget

| Component | Allocation |
|-----------|------------|
| BiomedCLIP INT8 | ~150 MB |
| MedGemma Q4_K_S | ~2.5 GB |
| App + UI | ~200 MB |
| System Reserve | ~2 GB |
| **Total Required** | **~5 GB** |

*Fits comfortably in 12GB device RAM*

## Latency Breakdown (Estimated)

**End-to-end clinical analysis with image:**

| Step | Time |
|------|------|
| Image capture & preprocessing | 50ms |
| BiomedCLIP inference | 50ms |
| Embedding formatting | 10ms |
| MedGemma prompt construction | 5ms |
| MedGemma inference (150 tokens) | 5-10s |
| Output parsing & display | 50ms |
| **Total** | **~6-11 seconds** |

## Validation Commands

Run benchmarks locally:

```bash
# Full test suite
python tests/run_all_tests.py

# Individual model tests
python tests/test_biomedclip.py
python tests/test_medgemma.py
```

## Notes

1. **CPU vs GPU/NPU**: Desktop benchmarks use CPU for fair comparison with mobile
2. **Context length**: Limited to 2048 tokens for mobile RAM constraints
3. **Batch size**: Single image/query (typical mobile use case)
4. **Warm-up**: All benchmarks exclude first inference (model loading)

---

*Benchmarks last updated: February 2026*
