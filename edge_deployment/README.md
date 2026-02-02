# Edge Deployment - MedGemma

## Quantized Model
- **Model**: medgemma-4b-q4_k_s-final.gguf (2.21GB)
- **Quantization**: Q4_K_S (4-bit k-quant)
- **Original Size**: 8.6GB to 2.21GB (74% reduction)
- **Target**: Realme GT Neo 6 (Snapdragon 8s Gen 3)
- **Performance**: ~15-30+ tok/s expected on mobile NPU

## Scripts
- simple_test.py - Test inference with Gemma 3 chat template
- quantize_gguf.py - Create quantized GGUF from F16 base

## Model NOT included in Git
The .gguf model file is excluded from Git (too large: 2.21GB).
Download or recreate using the quantization scripts.

