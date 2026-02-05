# MedGemma Edge: Offline Clinical AI for Low-Resource Settings

> **Kaggle MedGemma Impact Challenge** — Edge AI Submission

[![Edge AI](https://img.shields.io/badge/Edge%20AI-Optimized-green)]()
[![MedGemma](https://img.shields.io/badge/HAI--DEF-MedGemma%204B-blue)]()
[![License](https://img.shields.io/badge/License-MIT-yellow)]()

## 🎯 Project Overview

**MedGemma Edge** brings Google's healthcare AI models to mobile devices, enabling offline clinical decision support in environments without reliable internet access.

| Model | Original | Quantized | Reduction |
|-------|----------|-----------|-----------|
| **MedGemma 4B-IT** | 8.6 GB | 2.2 GB | 74% |
| **BiomedCLIP** | 329 MB | 84 MB | 74% |

**Target**: Android devices

## 🚀 Quick Start

```bash
# Clone and setup
git clone https://github.com/Umarii101/The-MedGemma-Impact-Challenge.git
cd The-MedGemma-Impact-Challenge

# Install dependencies
pip install -r requirements.txt

#Download The quantized Models First, I have provided the link to the models below

# Run validation tests
python tests/run_all_tests.py
```

## 📥 Quantized Model Downloads

The Models that I have quantized are too large for Git. Download from Google Drive:

**[📁 Download Models (Google Drive)](https://drive.google.com/file/d/1JZmLMVmimPnL3tiSe0GkNxYENRP49qAP/view?usp=sharing)**


## 📁 Repository Structure

```
├── README.md                 # You are here
├── EDGE_DEPLOYMENT.md        # ⭐ Full edge deployment story
├── requirements.txt
│
├── edge_deployment/          # Quantized models for mobile
│   ├── models/
│   │   ├── biomedclip/       # ONNX INT8 (84 MB)
│   │   └── medgemma/         # GGUF Q4_K_S (2.2 GB)
│   └── README.md
│
├── quantization/             # Quantization scripts & methodology
│   ├── scripts/
│   ├── results/
│   └── README.md
│
├── benchmarks/               # Performance measurements
│   └── README.md
│
├── android_app/              # Android demo application
│   └── README.md
│
├── tests/                    # Validation test suite
│   ├── test_biomedclip.py
│   ├── test_medgemma.py
│   └── run_all_tests.py
│
├── models/                   # Desktop model loaders
├── pipelines/                # Analysis pipelines
├── schemas/                  # Data models
├── utils/                    # Utilities
│
└── docs/                     # Additional documentation
    ├── SETUP_GUIDE.md
    ├── DOCUMENTATION.md
    └── PROJECT_SUMMARY.md
```

## 📊 Key Results

### Quantization Performance

| Model | Size | Accuracy | Speed |
|-------|------|----------|-------|
| BiomedCLIP INT8 | 84 MB | 99.95% vs FP32 | ~100ms CPU |
| MedGemma Q4_K_S | 2.2 GB | High quality | 9+ tok/s CPU |

### Validated Tests

```
[PASS] BiomedCLIP INT8 - Cosine similarity: 0.9995
[PASS] MedGemma Q4_K_S - Speed: 9.0 tok/s
ALL TESTS PASSED ✅
```

## 🌍 Impact

**Use Case**: Rural health clinics without internet access

1. Health worker captures patient symptoms + X-ray image
2. BiomedCLIP extracts visual features (100ms)
3. MedGemma provides clinical assessment (10-15s)
4. All processing happens **on-device** — no cloud required

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [EDGE_DEPLOYMENT.md](EDGE_DEPLOYMENT.md) | Full edge deployment story |
| [quantization/README.md](quantization/README.md) | Quantization methodology |
| [benchmarks/README.md](benchmarks/README.md) | Performance measurements |
| [docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md) | Development setup |

## 🔗 Links

- **Competition**: [Kaggle MedGemma Impact Challenge](https://kaggle.com/competitions/med-gemma-impact-challenge)
- **HAI-DEF Models**: [Google Health AI Developer Foundations](https://huggingface.co/google/medgemma-4b-it)
- **Video Demo**: [Coming Soon]


## ⚠️ Medical Disclaimer

This system is for **assistive purposes only**. Not FDA approved. All outputs require validation by licensed healthcare providers.

*Built for the Kaggle MedGemma Impact Challenge — Edge AI*
