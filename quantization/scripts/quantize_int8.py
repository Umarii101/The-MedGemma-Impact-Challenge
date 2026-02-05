"""
Export BiomedCLIP with Dynamic INT8 Quantization.
This reduces the model from ~329 MB to ~82 MB while keeping good accuracy.
"""
import os
import sys
import time
import torch
import numpy as np

def main():
    print("=" * 60)
    print("BiomedCLIP Dynamic INT8 Quantization")
    print("=" * 60)
    
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "deployment", "mobile", "biomedclip_vision.pt")
    int8_path = os.path.join(base_dir, "deployment", "mobile", "biomedclip_vision_int8.pt")
    int8_lite_path = os.path.join(base_dir, "deployment", "mobile", "biomedclip_vision_int8.ptl")
    
    print(f"\n[1/4] Loading FP32 TorchScript model...")
    model_fp32 = torch.jit.load(model_path, map_location='cpu')
    model_fp32.eval()
    
    fp32_size = os.path.getsize(model_path) / (1024 * 1024)
    print(f"  ✓ Loaded: {fp32_size:.2f} MB")
    
    print(f"\n[2/4] Applying dynamic INT8 quantization...")
    
    # Dynamic quantization converts weights to INT8 and quantizes activations dynamically
    # This is the simplest and most compatible quantization method
    model_int8 = torch.quantization.quantize_dynamic(
        model_fp32,
        {torch.nn.Linear},  # Quantize Linear layers (main memory consumer in ViT)
        dtype=torch.qint8
    )
    print("  ✓ Quantization applied")
    
    # Save quantized model
    print(f"\n[3/4] Saving quantized models...")
    
    # Regular save (for debugging/testing)
    torch.jit.save(model_int8, int8_path)
    int8_size = os.path.getsize(int8_path) / (1024 * 1024)
    print(f"  ✓ TorchScript INT8: {int8_size:.2f} MB")
    
    # Optimize for mobile
    from torch.utils.mobile_optimizer import optimize_for_mobile
    model_int8_mobile = optimize_for_mobile(model_int8)
    model_int8_mobile._save_for_lite_interpreter(int8_lite_path)
    int8_lite_size = os.path.getsize(int8_lite_path) / (1024 * 1024)
    print(f"  ✓ PyTorch Lite INT8: {int8_lite_size:.2f} MB")
    
    # Verify accuracy
    print(f"\n[4/4] Verifying accuracy...")
    
    # Test input
    np.random.seed(42)
    test_input = torch.randn(1, 3, 224, 224)
    
    # FP32 inference
    with torch.no_grad():
        start = time.time()
        for _ in range(10):
            output_fp32 = model_fp32(test_input)
        fp32_time = (time.time() - start) / 10 * 1000
    
    # INT8 inference
    with torch.no_grad():
        start = time.time()
        for _ in range(10):
            output_int8 = model_int8(test_input)
        int8_time = (time.time() - start) / 10 * 1000
    
    # Compare outputs
    output_fp32_np = output_fp32.numpy().flatten()
    output_int8_np = output_int8.numpy().flatten()
    
    cosine_sim = np.dot(output_fp32_np, output_int8_np) / (
        np.linalg.norm(output_fp32_np) * np.linalg.norm(output_int8_np)
    )
    max_diff = np.max(np.abs(output_fp32_np - output_int8_np))
    mean_diff = np.mean(np.abs(output_fp32_np - output_int8_np))
    
    print(f"\n  Cosine Similarity: {cosine_sim:.8f}")
    print(f"  Max Difference: {max_diff:.6f}")
    print(f"  Mean Difference: {mean_diff:.6f}")
    
    print(f"\n  Inference Time (avg 10 runs):")
    print(f"    FP32: {fp32_time:.2f} ms")
    print(f"    INT8: {int8_time:.2f} ms")
    print(f"    Speedup: {fp32_time/int8_time:.2f}x")
    
    # Summary
    print("\n" + "=" * 60)
    print("QUANTIZATION COMPLETE")
    print("=" * 60)
    
    print(f"\nModel Sizes:")
    print(f"  FP32 TorchScript: {fp32_size:.2f} MB")
    print(f"  INT8 TorchScript: {int8_size:.2f} MB")
    print(f"  INT8 PyTorch Lite: {int8_lite_size:.2f} MB")
    
    print(f"\nSize Reduction: {(1 - int8_lite_size/fp32_size)*100:.1f}%")
    
    print(f"\nQuality:")
    if cosine_sim > 0.99:
        print(f"  ✓ Excellent accuracy (cosine > 0.99)")
    elif cosine_sim > 0.95:
        print(f"  ⚠ Good accuracy (cosine > 0.95)")
    else:
        print(f"  ✗ Accuracy loss detected")
    
    print(f"\nFiles:")
    print(f"  {int8_path}")
    print(f"  {int8_lite_path}")
    
    print(f"\nRecommendation for Realme GT Neo 6:")
    print(f"  Use: biomedclip_vision_int8.ptl ({int8_lite_size:.1f} MB)")
    print(f"  Runtime: PyTorch Mobile (LibTorch)")
    print(f"  Expected: ~{int8_time:.0f}ms per image on CPU, faster on GPU")

if __name__ == "__main__":
    main()
