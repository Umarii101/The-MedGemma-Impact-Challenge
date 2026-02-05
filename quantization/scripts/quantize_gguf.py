"""
Quantize F16 GGUF to Q4_K_S using llama-cpp-python
"""
import ctypes
from llama_cpp import llama_model_quantize, llama_model_quantize_default_params

# GGML quantization types
# From llama.cpp: https://github.com/ggerganov/llama.cpp/blob/master/ggml/include/ggml.h
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q2_K = 10
GGML_TYPE_Q3_K = 11
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q5_K = 13
GGML_TYPE_Q6_K = 14
GGML_TYPE_IQ2_XXS = 16
GGML_TYPE_IQ2_XS = 17
GGML_TYPE_IQ3_XXS = 18
GGML_TYPE_IQ1_S = 19
GGML_TYPE_IQ4_NL = 20

# llama.cpp file types (for quantization output)
LLAMA_FTYPE_Q4_K_S = 14  # Q4_K_S: 4-bit K-quant, small variant

input_path = "quantization/outputs/medgemma-4b-f16-v3.gguf"
output_path = "quantization/outputs/medgemma-4b-q4_k_s-final.gguf"

print(f"Input:  {input_path}")
print(f"Output: {output_path}")
print(f"Target: Q4_K_S (type {LLAMA_FTYPE_Q4_K_S})")
print()

# Get default quantization params
params = llama_model_quantize_default_params()
params.nthread = 8  # Use 8 threads
params.ftype = LLAMA_FTYPE_Q4_K_S

print("Starting quantization...")
print("This may take 5-10 minutes...")
print()

try:
    result = llama_model_quantize(
        input_path.encode('utf-8'),
        output_path.encode('utf-8'),
        ctypes.byref(params)
    )
    
    if result == 0:
        print(f"\n✅ Quantization successful!")
        print(f"Output: {output_path}")
    else:
        print(f"\n❌ Quantization failed with code: {result}")
except Exception as e:
    print(f"\n❌ Error: {e}")
