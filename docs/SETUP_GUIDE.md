# Setup and Testing Guide

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone or download the project
cd medgemma-backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify CUDA Setup

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA Version: {torch.version.cuda}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"}')"
```

Expected output (RTX 3080):
```
CUDA Available: True
CUDA Version: 11.8 (or higher)
GPU: NVIDIA GeForce RTX 3080
```

### 3. Test Installation

```bash
# Quick test
python -c "from pipelines import MultimodalPipeline; print('✓ Imports successful')"
```

### 4. Run Demo

```bash
# Run main demo script
python main.py
```

## 📦 Detailed Setup

### Prerequisites

1. **Operating System**
   - Linux (Ubuntu 20.04+ recommended)
   - Windows 10/11 with WSL2 (or native with CUDA support)
   - macOS (CPU only, not recommended)

2. **GPU Drivers**
   - NVIDIA GPU with CUDA support
   - CUDA 11.8 or higher
   - cuDNN 8.x
   - Latest NVIDIA drivers

3. **Python**
   - Python 3.10 or 3.11 (recommended)
   - pip 23.0+

### Installation Steps

#### Step 1: Install CUDA (if not already installed)

**Ubuntu/Linux**:
```bash
# Check if CUDA is installed
nvcc --version

# If not installed, follow NVIDIA's official guide:
# https://developer.nvidia.com/cuda-downloads
```

**Windows**:
- Download CUDA Toolkit from NVIDIA
- Install with default settings
- Verify installation: `nvcc --version`

#### Step 2: Install Python Dependencies

```bash
# Core ML frameworks
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Transformers and accelerate
pip install transformers accelerate

# Image processing
pip install timm pillow opencv-python

# Scientific computing
pip install numpy scipy scikit-learn

# Validation
pip install pydantic

# Optional: BitsAndBytes for 8-bit quantization
pip install bitsandbytes sentencepiece
```

#### Step 3: Download Models (First Run)

Models will auto-download on first use. Expect ~15-20GB total:

- MedGemma 7B: ~13GB
- CLIP ViT-Large: ~1.7GB
- DINOv2: ~300MB

**Pre-download (optional)**:
```python
from transformers import AutoModel, AutoTokenizer

# Pre-download MedGemma
tokenizer = AutoTokenizer.from_pretrained("google/medgemma-7b")
model = AutoModelForCausalLM.from_pretrained("google/medgemma-7b")

# Pre-download CLIP
from transformers import CLIPModel
clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
```

## 🧪 Testing

### Unit Tests (Example Structure)

```python
# test_clinical_text.py
import pytest
from pipelines import ClinicalTextPipeline

def test_basic_summarization():
    """Test basic clinical text summarization"""
    pipeline = ClinicalTextPipeline(use_8bit=True)
    
    note = "Patient presents with fever and cough for 3 days."
    
    result = pipeline.analyze(
        clinical_note=note,
        patient_age=45,
        extract_entities=False,
        calculate_risk=False,
        generate_recommendations=False
    )
    
    assert result.summary is not None
    assert len(result.summary) > 0
    assert result.confidence >= 0.0
    assert result.confidence <= 1.0
    
    pipeline.cleanup()

def test_entity_extraction():
    """Test clinical entity extraction"""
    pipeline = ClinicalTextPipeline(use_8bit=True)
    
    note = """
    Patient has diabetes and hypertension.
    Currently taking metformin and lisinopril.
    Presents with headache and nausea.
    """
    
    result = pipeline.analyze(
        clinical_note=note,
        extract_entities=True
    )
    
    # Should extract some entities
    total_entities = (
        len(result.extracted_symptoms) +
        len(result.extracted_conditions) +
        len(result.medications_mentioned)
    )
    
    assert total_entities > 0
    
    pipeline.cleanup()

# Run tests
pytest test_clinical_text.py -v
```

### Integration Tests

```python
# test_integration.py
from pipelines import MultimodalPipeline
from PIL import Image
import numpy as np

def test_full_pipeline():
    """Test complete multimodal pipeline"""
    pipeline = MultimodalPipeline()
    
    # Create dummy image
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    image = Image.fromarray(img_array)
    
    note = "Patient presents with chest pain and shortness of breath."
    
    result = pipeline.analyze_with_image(
        clinical_note=note,
        image=image,
        image_type="Chest X-Ray",
        patient_age=60
    )
    
    assert result.clinical_summary is not None
    assert result.text_analysis is not None
    assert result.image_analysis is not None
    assert result.overall_confidence >= 0.0
    
    pipeline.cleanup()
```

### Performance Benchmarks

```python
# benchmark.py
import time
from pipelines import MultimodalPipeline
from examples.example_data import EXAMPLE_CASES

def benchmark_text_analysis():
    """Benchmark clinical text analysis"""
    pipeline = MultimodalPipeline()
    
    case = EXAMPLE_CASES["case_1_respiratory"]
    
    start = time.time()
    result = pipeline.analyze_clinical_text(
        clinical_note=case["clinical_note"],
        patient_age=case["patient_age"]
    )
    elapsed = time.time() - start
    
    print(f"Text Analysis: {elapsed:.2f}s")
    pipeline.cleanup()
    
    return elapsed

if __name__ == "__main__":
    text_time = benchmark_text_analysis()
    
    print("\nBenchmark Results:")
    print(f"  Text Pipeline: {text_time:.2f}s")
    print(f"  Target: <10s on RTX 3080")
```

## 🔍 Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
```python
# Use 8-bit quantization
pipeline = MultimodalPipeline(use_8bit_llm=True)

# Clear cache between runs
import torch
torch.cuda.empty_cache()

# Reduce max_new_tokens
result = pipeline.analyze_clinical_text(
    clinical_note=note,
    max_new_tokens=256  # Reduced from 512
)
```

#### 2. Model Download Fails

**Error**: `OSError: Can't load model`

**Solutions**:
```bash
# Set Hugging Face cache directory
export HF_HOME=/path/to/large/storage/.cache/huggingface

# Use offline mode (after initial download)
export TRANSFORMERS_OFFLINE=1

# Manual download
from huggingface_hub import snapshot_download
snapshot_download("google/medgemma-7b")
```

#### 3. Slow Performance on CPU

**Error**: Running on CPU instead of GPU

**Solutions**:
```python
# Verify CUDA
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.device_count())   # Should be >= 1

# Check device placement
from utils.memory import check_cuda_setup
info = check_cuda_setup()
print(info)
```

#### 4. Import Errors

**Error**: `ModuleNotFoundError: No module named 'X'`

**Solutions**:
```bash
# Reinstall requirements
pip install -r requirements.txt --force-reinstall

# Install specific package
pip install transformers --upgrade
```

### Debug Mode

Enable detailed logging:

```python
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Then run pipeline
from pipelines import MultimodalPipeline
pipeline = MultimodalPipeline()
```

## 📊 Resource Monitoring

### GPU Monitoring

```bash
# Real-time GPU monitoring
nvidia-smi -l 1

# Memory usage
nvidia-smi --query-gpu=memory.used,memory.free --format=csv

# Process-specific
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
```

### Python Memory Monitoring

```python
from utils.memory import get_memory_manager

memory_mgr = get_memory_manager()

# Before inference
memory_mgr.log_memory_usage("Before inference")

# Run inference
result = pipeline.analyze_clinical_text(note)

# After inference
memory_mgr.log_memory_usage("After inference")

# Get detailed stats
stats = memory_mgr.get_memory_stats()
print(stats)
```

## 🎯 Validation Checklist

Before deploying or submitting:

- [ ] CUDA available and functioning
- [ ] All models successfully load
- [ ] Demo script runs without errors
- [ ] Example cases produce valid outputs
- [ ] Safety disclaimers present in all outputs
- [ ] GPU memory usage < 10GB on RTX 3080
- [ ] Inference time < 15s for multimodal
- [ ] All outputs are valid JSON
- [ ] No diagnostic language in outputs
- [ ] Documentation is complete

## 📝 Example Test Run

```bash
# Full test sequence
python -c "from utils.memory import check_cuda_setup; print(check_cuda_setup())"
python -c "from examples.example_data import list_example_cases; list_example_cases()"
python main.py
```

Expected runtime: 5-10 minutes first time (model download), 2-3 minutes subsequent runs.

## 🎓 Learning Path

For developers new to medical AI:

1. **Start with**: Run `main.py` and review outputs
2. **Next**: Examine `clinical_text_pipeline.py` to understand flow
3. **Then**: Read safety mechanisms in `utils/safety.py`
4. **Finally**: Customize prompts in `models/medgemma.py`

## 📞 Support

If issues persist:

1. Check CUDA/GPU setup
2. Verify Python version (3.10 or 3.11)
3. Review error logs
4. Check disk space (need 30GB+)
5. Try fallback models (Gemma-2 instead of MedGemma)

---

**Ready to get started?** Run: `python main.py`
