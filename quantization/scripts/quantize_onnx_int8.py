"""
Quantize BiomedCLIP ONNX model to INT8 using ONNX Runtime quantization.
This should reduce the model from ~329 MB to ~82 MB.
"""
import os
import sys
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

def main():
    print("=" * 60)
    print("BiomedCLIP ONNX INT8 Quantization")
    print("=" * 60)
    
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    onnx_path = os.path.join(base_dir, "deployment", "onnx", "biomedclip_vision.onnx")
    int8_path = os.path.join(base_dir, "deployment", "onnx", "biomedclip_vision_int8.onnx")
    
    print(f"\n[1/3] Loading FP32 ONNX model...")
    fp32_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"  Size: {fp32_size:.2f} MB")
    
    print(f"\n[2/3] Applying dynamic INT8 quantization...")
    
    # Dynamic INT8 quantization - quantizes weights to INT8
    # and computes activations in INT8 dynamically
    quantize_dynamic(
        model_input=onnx_path,
        model_output=int8_path,
        weight_type=QuantType.QUInt8,  # INT8 weights
        per_channel=True,  # Better accuracy with per-channel quantization
        reduce_range=False,  # Full INT8 range
        extra_options={
            'ActivationSymmetric': False,
            'WeightSymmetric': True,
        }
    )
    
    int8_size = os.path.getsize(int8_path) / (1024 * 1024)
    print(f"  ✓ INT8 model: {int8_size:.2f} MB")
    print(f"  Size reduction: {(1 - int8_size/fp32_size)*100:.1f}%")
    
    # Verify
    print(f"\n[3/3] Verifying accuracy...")
    
    # Load both models
    session_fp32 = ort.InferenceSession(onnx_path)
    session_int8 = ort.InferenceSession(int8_path)
    
    input_name = session_fp32.get_inputs()[0].name
    output_name = session_fp32.get_outputs()[0].name
    
    # Test input
    np.random.seed(42)
    test_input = np.random.randn(1, 3, 224, 224).astype(np.float32)
    
    # Inference
    import time
    
    # FP32
    start = time.time()
    for _ in range(10):
        output_fp32 = session_fp32.run([output_name], {input_name: test_input})[0]
    fp32_time = (time.time() - start) / 10 * 1000
    
    # INT8
    start = time.time()
    for _ in range(10):
        output_int8 = session_int8.run([output_name], {input_name: test_input})[0]
    int8_time = (time.time() - start) / 10 * 1000
    
    # Compare
    output_fp32_flat = output_fp32.flatten()
    output_int8_flat = output_int8.flatten()
    
    cosine_sim = np.dot(output_fp32_flat, output_int8_flat) / (
        np.linalg.norm(output_fp32_flat) * np.linalg.norm(output_int8_flat)
    )
    max_diff = np.max(np.abs(output_fp32_flat - output_int8_flat))
    mean_diff = np.mean(np.abs(output_fp32_flat - output_int8_flat))
    
    print(f"\n  Accuracy Metrics:")
    print(f"    Cosine Similarity: {cosine_sim:.8f}")
    print(f"    Max Difference: {max_diff:.6f}")
    print(f"    Mean Difference: {mean_diff:.6f}")
    
    print(f"\n  Inference Time (avg 10 runs):")
    print(f"    FP32: {fp32_time:.2f} ms")
    print(f"    INT8: {int8_time:.2f} ms")
    print(f"    Speedup: {fp32_time/int8_time:.2f}x")
    
    # Summary
    print("\n" + "=" * 60)
    print("QUANTIZATION COMPLETE")
    print("=" * 60)
    
    print(f"\nModel Sizes:")
    print(f"  FP32: {fp32_size:.2f} MB")
    print(f"  INT8: {int8_size:.2f} MB")
    print(f"  Reduction: {(1 - int8_size/fp32_size)*100:.1f}%")
    
    if cosine_sim > 0.99:
        print(f"\n✓ Excellent accuracy preserved (cosine > 0.99)")
    elif cosine_sim > 0.95:
        print(f"\n⚠ Good accuracy (cosine > 0.95)")
    else:
        print(f"\n⚠ Some accuracy loss (cosine = {cosine_sim:.4f})")
    
    print(f"\nFile: {int8_path}")

if __name__ == "__main__":
    main()
