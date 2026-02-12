"""
BiomedCLIP Zero-Shot Classification Evaluation
===============================================
Tests the BiomedCLIP INT8 ONNX + pre-computed text embeddings pipeline
on labeled medical images to measure classification accuracy.

This is the same pipeline that runs on-device in the MedLens Android app:
  Image → BiomedCLIP (512-dim embedding) → cosine similarity vs 30 labels → top-K predictions

Usage:
    python evaluation/biomedclip_classification_eval.py

Requires:
    pip install onnxruntime numpy Pillow
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Paths (relative to repo root)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_INT8 = REPO_ROOT / "edge_deployment" / "models" / "biomedclip" / "biomedclip_vision_int8.onnx"
MODEL_FP32 = REPO_ROOT / "edge_deployment" / "models" / "biomedclip" / "biomedclip_vision.onnx"
EMBEDDINGS_PATH = REPO_ROOT / "Medlens" / "app" / "src" / "main" / "assets" / "text_embeddings.json"
CATEGORIES_PATH = REPO_ROOT / "Medlens" / "app" / "src" / "main" / "assets" / "label_categories.json"
TEST_DATA_DIR = REPO_ROOT / "evaluation" / "test_data" / "xray"

# BiomedCLIP preprocessing constants (CLIP-standard, matches Android app)
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
IMAGE_SIZE = 224

# Ground-truth mapping: folder name → set of acceptable top-K label matches
GROUND_TRUTH = {
    "covid19": {
        "exact": ["COVID-19 infection in chest x-ray"],
        "acceptable": [
            "COVID-19 infection in chest x-ray",
            "viral pneumonia in chest x-ray",
            "pneumonia in chest x-ray",
            "lung opacity in chest x-ray",
        ],
    },
    "normal": {
        "exact": ["normal chest x-ray"],
        "acceptable": [
            "normal chest x-ray",
            "normal healthy medical image",
        ],
    },
    "pneumonia": {
        "exact": ["pneumonia in chest x-ray"],
        "acceptable": [
            "pneumonia in chest x-ray",
            "bacterial pneumonia in chest x-ray",
            "viral pneumonia in chest x-ray",
            "lung opacity in chest x-ray",
        ],
    },
}


def preprocess_image(image_path: str) -> np.ndarray:
    """Preprocess image exactly as BiomedClipInference.kt does on Android."""
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0  # [H, W, 3] in [0, 1]
    arr = (arr - MEAN) / STD  # normalize
    arr = arr.transpose(2, 0, 1)  # HWC → CHW
    arr = np.expand_dims(arr, 0)  # → [1, 3, 224, 224]
    return arr


def load_embeddings(path: Path) -> Tuple[List[str], np.ndarray]:
    """Load pre-computed text embeddings. Returns (labels, embeddings[N, 512])."""
    with open(path, "r") as f:
        data = json.load(f)
    labels = list(data.keys())
    embeddings = np.array([data[l] for l in labels], dtype=np.float32)
    return labels, embeddings


def load_categories(path: Path) -> Dict[str, str]:
    """Load label → category mapping."""
    with open(path, "r") as f:
        return json.load(f)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between vector a [1, D] and matrix b [N, D] → [N]."""
    a_norm = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    return (a_norm @ b_norm.T).squeeze()


def classify_image(
    session, image_path: str, labels: List[str], text_embeddings: np.ndarray
) -> List[Tuple[str, float]]:
    """Run BiomedCLIP + zero-shot classification on a single image."""
    input_tensor = preprocess_image(image_path)
    embedding = session.run(None, {"image": input_tensor})[0]  # [1, 512]
    similarities = cosine_similarity(embedding, text_embeddings)
    ranked = sorted(zip(labels, similarities.tolist()), key=lambda x: -x[1])
    return ranked


def discover_test_images(test_dir: Path) -> List[Tuple[str, str]]:
    """Discover test images organized as test_dir/<class>/<file>."""
    images = []
    for class_dir in sorted(test_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        for img_file in sorted(class_dir.iterdir()):
            if img_file.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
                images.append((str(img_file), class_name))
    return images


def run_evaluation():
    """Main evaluation pipeline."""
    import onnxruntime as ort

    print("=" * 80)
    print("  BiomedCLIP Zero-Shot Classification Evaluation")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Load resources
    # ------------------------------------------------------------------
    print(f"\n[1/4] Loading text embeddings from {EMBEDDINGS_PATH.name}...")
    labels, text_embeddings = load_embeddings(EMBEDDINGS_PATH)
    categories = load_categories(CATEGORIES_PATH)
    print(f"       {len(labels)} labels, {text_embeddings.shape[1]}-dim embeddings")

    print(f"\n[2/4] Loading BiomedCLIP INT8 ONNX model...")
    print(f"       Model: {MODEL_INT8.name} ({MODEL_INT8.stat().st_size / 1e6:.1f} MB)")
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_int8 = ort.InferenceSession(str(MODEL_INT8), sess_options)
    print("       ✓ INT8 model loaded")

    # Also load FP32 for comparison if available
    session_fp32 = None
    if MODEL_FP32.exists():
        print(f"\n       Loading FP32 baseline model for comparison...")
        session_fp32 = ort.InferenceSession(str(MODEL_FP32), sess_options)
        print("       ✓ FP32 model loaded")

    # ------------------------------------------------------------------
    # Discover test images
    # ------------------------------------------------------------------
    print(f"\n[3/4] Discovering test images in {TEST_DATA_DIR}...")
    test_images = discover_test_images(TEST_DATA_DIR)
    print(f"       Found {len(test_images)} images across {len(set(c for _, c in test_images))} classes")

    if not test_images:
        print("\n  ✗ No test images found. Place images in evaluation/test_data/xray/<class>/")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Run classification
    # ------------------------------------------------------------------
    print(f"\n[4/4] Running zero-shot classification...\n")
    print("-" * 80)

    results = []
    total_time_ms = 0

    for img_path, true_class in test_images:
        img_name = Path(img_path).name

        # Time the inference
        start = time.perf_counter()
        ranked = classify_image(session_int8, img_path, labels, text_embeddings)
        elapsed_ms = (time.perf_counter() - start) * 1000
        total_time_ms += elapsed_ms

        top5 = ranked[:5]
        gt = GROUND_TRUTH.get(true_class, {"exact": [], "acceptable": []})

        # Check accuracy
        top1_label = top5[0][0]
        top1_correct = top1_label in gt["exact"]
        top3_hit = any(lbl in gt["acceptable"] for lbl, _ in top5[:3])
        top5_hit = any(lbl in gt["acceptable"] for lbl, _ in top5[:5])

        results.append({
            "image": img_name,
            "true_class": true_class,
            "top1_label": top1_label,
            "top1_score": top5[0][1],
            "top1_correct": top1_correct,
            "top3_hit": top3_hit,
            "top5_hit": top5_hit,
            "inference_ms": elapsed_ms,
            "top5": top5,
        })

        # Print per-image results
        status = "✓" if top1_correct else ("~" if top3_hit else "✗")
        print(f"  {status} [{true_class:>10s}] {img_name}")
        print(f"    Inference: {elapsed_ms:.1f} ms")
        for i, (lbl, score) in enumerate(top5):
            marker = " ◄" if lbl in gt["acceptable"] else ""
            cat = categories.get(lbl, "?")
            print(f"    #{i+1}: {score:.4f}  {lbl} [{cat}]{marker}")
        print()

    # ------------------------------------------------------------------
    # INT8 vs FP32 embedding fidelity (if FP32 available)
    # ------------------------------------------------------------------
    fidelity_results = []
    if session_fp32:
        print("-" * 80)
        print("  INT8 vs FP32 Quantization Fidelity")
        print("-" * 80)
        for img_path, true_class in test_images:
            input_tensor = preprocess_image(img_path)
            emb_int8 = session_int8.run(None, {"image": input_tensor})[0]
            emb_fp32 = session_fp32.run(None, {"image": input_tensor})[0]

            cos_sim = float(cosine_similarity(emb_int8, emb_fp32))
            max_diff = float(np.abs(emb_int8 - emb_fp32).max())
            mean_diff = float(np.abs(emb_int8 - emb_fp32).mean())

            # Check if INT8 and FP32 agree on top-1 classification
            sims_int8 = cosine_similarity(emb_int8, text_embeddings)
            sims_fp32 = cosine_similarity(emb_fp32, text_embeddings)
            top1_int8 = labels[int(np.argmax(sims_int8))]
            top1_fp32 = labels[int(np.argmax(sims_fp32))]

            fidelity_results.append({
                "image": Path(img_path).name,
                "cosine_sim": cos_sim,
                "max_diff": max_diff,
                "mean_diff": mean_diff,
                "top1_agree": top1_int8 == top1_fp32,
            })

            agree = "✓" if top1_int8 == top1_fp32 else "✗"
            print(f"  {agree} {Path(img_path).name:>40s}  cos={cos_sim:.6f}  "
                  f"max_diff={max_diff:.6f}  top1_agree={top1_int8 == top1_fp32}")
        print()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("=" * 80)
    print("  SUMMARY")
    print("=" * 80)

    n = len(results)
    top1_acc = sum(1 for r in results if r["top1_correct"]) / n * 100
    top3_acc = sum(1 for r in results if r["top3_hit"]) / n * 100
    top5_acc = sum(1 for r in results if r["top5_hit"]) / n * 100
    avg_ms = total_time_ms / n

    print(f"\n  Test images:        {n}")
    print(f"  Top-1 exact acc:    {top1_acc:.1f}%  ({sum(1 for r in results if r['top1_correct'])}/{n})")
    print(f"  Top-3 clinical acc: {top3_acc:.1f}%  ({sum(1 for r in results if r['top3_hit'])}/{n})")
    print(f"  Top-5 clinical acc: {top5_acc:.1f}%  ({sum(1 for r in results if r['top5_hit'])}/{n})")
    print(f"  Avg inference:      {avg_ms:.1f} ms/image")

    if fidelity_results:
        avg_cos = np.mean([r["cosine_sim"] for r in fidelity_results])
        agree_pct = sum(1 for r in fidelity_results if r["top1_agree"]) / len(fidelity_results) * 100
        print(f"\n  INT8/FP32 cosine:   {avg_cos:.6f}")
        print(f"  INT8/FP32 top-1 agreement: {agree_pct:.0f}%")

    print(f"\n  Model: BiomedCLIP ViT-B/16 INT8 ONNX ({MODEL_INT8.stat().st_size / 1e6:.1f} MB)")
    print(f"  Labels: {len(labels)} conditions across 4 categories")
    print(f"  Embedding dim: {text_embeddings.shape[1]}")
    print()

    # ------------------------------------------------------------------
    # Save results as JSON
    # ------------------------------------------------------------------
    output_path = REPO_ROOT / "evaluation" / "results" / "biomedclip_classification_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "model": "BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 (INT8 ONNX)",
        "model_size_mb": round(MODEL_INT8.stat().st_size / 1e6, 1),
        "num_labels": len(labels),
        "embedding_dim": int(text_embeddings.shape[1]),
        "num_test_images": n,
        "top1_exact_accuracy": round(top1_acc, 1),
        "top3_clinical_accuracy": round(top3_acc, 1),
        "top5_clinical_accuracy": round(top5_acc, 1),
        "avg_inference_ms": round(avg_ms, 1),
        "per_image_results": [
            {
                "image": r["image"],
                "true_class": r["true_class"],
                "top1_label": r["top1_label"],
                "top1_score": round(r["top1_score"], 4),
                "top1_correct": r["top1_correct"],
                "top3_hit": r["top3_hit"],
                "top5_hit": r["top5_hit"],
                "inference_ms": round(r["inference_ms"], 1),
                "top5_predictions": [
                    {"label": lbl, "score": round(s, 4)} for lbl, s in r["top5"]
                ],
            }
            for r in results
        ],
    }

    if fidelity_results:
        output["quantization_fidelity"] = {
            "avg_cosine_similarity": round(float(avg_cos), 6),
            "top1_agreement_pct": round(agree_pct, 1),
            "per_image": [
                {
                    "image": r["image"],
                    "cosine_similarity": round(r["cosine_sim"], 6),
                    "max_abs_diff": round(r["max_diff"], 6),
                    "top1_agree": r["top1_agree"],
                }
                for r in fidelity_results
            ],
        }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Results saved to: {output_path.relative_to(REPO_ROOT)}")
    print()


if __name__ == "__main__":
    run_evaluation()
