# CaliDrift: Measuring Confidence Drift Between Verbalized and Internal Probability in Aligned Small Language Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains the code, prompts, and evaluation data for the paper:

> **CaliDrift: Measuring Confidence Drift Between Verbalized and Internal Probability in Aligned Small Language Models**  
> Samuel Stephen, R. Vignesh  
> Karunya Institute of Technology and Sciences, Coimbatore, India

CaliDrift introduces the **Confidence Drift Index (CDI)** — a bounded scalar metric for measuring the mismatch between a language model's verbalized confidence and its internal token-level confidence, approximated via entropy.

---

## Key Findings

- **Architecture-dependent calibration correction:** Gemma-2B shows large correction (ΔCDI = −0.360, d = 3.60), Llama-3.2-1B moderate (ΔCDI = −0.202, d = 1.36), Qwen2.5-1.5B no significant change (p = 0.137)
- **OED dominates at 98.0%** of 3,417 responses — both base and instruct models verbally overstate confidence
- **CDI is a population-level diagnostic**, not a per-response hallucination classifier (AUROC = 0.446)

---

## Repository Structure

```
CaliDrift/
├── CaliDrift.py                    # Main experiment pipeline (all 3 model pairs)
├── CaliDrift_SemanticEntropy.py    # Semantic entropy validation (Gemma-2B-IT, TruthfulQA)
├── calidrift.ipynb                 # Kaggle notebook with full run outputs
├── requirements.txt                # Python dependencies
├── prompts/
│   ├── instruct_prompt.txt         # Direct instruction prompt for instruct models
│   └── base_prompt.txt             # Few-shot completion prompt for base models
├── data/
│   └── checkpoint_merged.json      # Merged experiment results (3,417 responses)
├── results/
│   └── calidrift_results_final.xlsx # All tables from paper
└── figures/
    ├── Figure2_Reliability_Diagrams.png
    ├── Figure3_VC_vs_IC_Scatter.png
    └── Figure4_CDI_Histograms.png
```

---

## Models Evaluated

| Pair | Base Model | Instruct Model |
|------|-----------|----------------|
| Pair 1 | Qwen/Qwen2.5-1.5B | Qwen/Qwen2.5-1.5B-Instruct |
| Pair 2 | google/gemma-2b | google/gemma-2b-it |
| Pair 3 | meta-llama/Llama-3.2-1B | meta-llama/Llama-3.2-1B-Instruct |

---

## Datasets

| Dataset | Source |
|---------|--------|
| TruthfulQA | https://github.com/sylinrl/TruthfulQA |
| SimpleQA | https://openai.com/research/simpleqa |
| FaithDial | https://github.com/McGill-NLP/FaithDial |

---

## Setup

```bash
git clone https://github.com/77samuel/CaliDrift.git
cd CaliDrift
pip install -r requirements.txt
```

For Gemma models, a HuggingFace token with gated model access is required:

```python
from huggingface_hub import login
login(token="your_hf_token")
```

---

## Running the Experiment

### Main Experiment (all 3 pairs)

```bash
python CaliDrift.py
```

Configure at the top of `CaliDrift.py`:

```python
SAMPLE_SIZE = 100       # samples per dataset
SEEDS       = [1, 2, 3] # random seeds
DATA_PATH   = "/path/to/datasets/"
OUTDIR      = "./calidrift_results"
```

### Semantic Entropy Validation

```bash
python CaliDrift_SemanticEntropy.py
```

Runs on Gemma-2B-IT, TruthfulQA only, 100 questions × 5 draws.

---

## Compute Requirements

All experiments run on Kaggle T4 × 2 GPU (16 GB VRAM). All models loaded at FP16 precision without quantization.

| Model | VRAM |
|-------|------|
| Qwen2.5-1.5B / Instruct | ~3 GB |
| Gemma-2B / IT | ~5 GB |
| Llama-3.2-1B / Instruct | ~2.5 GB |

---

## Results Summary

| Model Pair | CDI (Base) | CDI (Inst.) | ΔCDI | Cohen's d | p-value |
|------------|-----------|-------------|------|-----------|---------|
| Qwen2.5-1.5B | 0.520 | 0.539 | +0.019 | 0.13 | 0.137 (ns) |
| Gemma-2B | 0.721 | 0.362 | −0.360 | 3.60 | 0.0005 |
| Llama-3.2-1B | 0.747 | 0.546 | −0.202 | 1.36 | 0.0081 |

---

## Citation

```bibtex
@article{stephen2025calidrift,
  title     = {CaliDrift: Measuring Confidence Drift Between Verbalized and Internal Probability in Aligned Small Language Models},
  author    = {Stephen, Samuel and Vignesh, R.},
  journal   = {Under Review},
  year      = {2025},
  note      = {Karunya Institute of Technology and Sciences}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

Samuel Stephen — samuels24@karunya.edu.in  
R. Vignesh — vignesh@karunya.edu
