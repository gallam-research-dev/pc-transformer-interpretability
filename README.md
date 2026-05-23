# Predictive Coding Tendencies in Transformers

> **Do Transformers Implicitly Implement Predictive Coding?**  
> Multi-Method Empirical Evidence Across Six Models

📄 **Paper**: [arXiv link] *(to be updated after submission)*  
👤 **Author**: Jong-O Yun — Independent Researcher, Korea  
📧 gallam.research@gmail.com

---

## Overview

This repository contains code and data for the paper:

> *"Predictive Coding Tendencies in Transformers: Evidence from Residual Stream Convergence, Activation Patching, MLP Transform Analysis, Zero-Ablation, and Logit Lens"*

We investigate whether transformer-based language models implicitly exhibit computational tendencies consistent with **Predictive Coding (PC)** — without explicit design — across **6 models** using **5 complementary experiments**.

### Models
| Model | Params | Layers | Training Data |
|-------|--------|--------|--------------|
| GPT-2 Small | 117M | 12 | WebText |
| GPT-2 Medium | 345M | 24 | WebText |
| GPT-2 Large | 774M | 36 | WebText |
| GPT-Neo 125M | 125M | 12 | The Pile |
| Pythia 160M | 160M | 12 | The Pile |
| Pythia 410M | 410M | 24 | The Pile |

### Key Findings

1. **Training-configuration-dependent convergence**: English (primary training language) shows the strongest mid-layer cosine similarity in GPT-2 models; this pattern weakens in GPT-Neo and Pythia.
2. **Mid-layer semantic specialization**: Activation patching identifies normalized layer position 0.25–0.33 as the peak causal locus for semantic processing, consistent across all 6 models.
3. **W-shaped MLP magnitude**: A spike–convergence–spike structure in GPT-2 is scale-invariant; a dual-spike structure appears in GPT-Neo and Pythia.
4. **Convergence zone functional primacy**: Zero-ablation shows the *low-magnitude* convergence zone is *most critical* — dissociating magnitude from functional importance.
5. **Monotonic prediction refinement**: Logit lens confirms hierarchical convergence of target token rank (initial rank 375–5592 → final rank <1) across all GPT-2 variants.

---

## Setup

```bash
git clone https://github.com/gallam-research-dev/pc-transformer-interpretability
cd pc-transformer-interpretability
pip install -r requirements.txt
```

**requirements.txt:**
```
transformer_lens>=1.14.0
torch>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
matplotlib>=3.7.0
pandas>=2.0.0
```

All experiments were run on **Google Colab T4 GPU**.

---

## Experiments

### Experiment 1: Residual Stream Convergence
Measures layer-wise cosine similarity and delta magnitude across 7 languages (140 prompts total).

```bash
python experiments/exp1_residual_stream.py
```

### Experiment 2: Activation Patching
Identifies causally important layers for semantic processing across 23 semantic pairs.

```bash
python experiments/exp2_activation_patching.py
```

### Experiment 3: MLP Transform Magnitude
Profiles MLP transformation magnitude across normalized layer positions.

```bash
python experiments/exp3_mlp_magnitude.py
```

### Experiment 4: Zero-Ablation
Tests functional importance of MLP zones by zeroing out each zone's outputs.

```bash
python experiments/exp4_zero_ablation.py
```

### Experiment 5: Logit Lens
Tracks target token rank across layers to measure prediction refinement.

```bash
python experiments/exp5_logit_lens.py
```

### All-in-one (recommended)
Runs all experiments sequentially for a single model.  
**Edit `RUN_MODEL` at the top of the script before running.**

```bash
python experiments/unified_experiment.py
```

---

## Results

Pre-computed results are available in `results/`:

- `multi_model_results_v2.json` — Experiments 1–3 (all 6 models)
- `exp4_improved_results.json` — Experiment 4 (GPT-2 family)
- `exp4_exp5_results.json` — Experiment 5 (GPT-2 family)

---

## Figures

All figures used in the paper are in `figures/`.

| File | Description |
|------|-------------|
| `figure1.png` | Mid-layer cosine similarity by language (6 models) |
| `figure2.png` | MLP transform magnitude by model family |
| `figure3.png` | Activation patching heatmap (6 models) |
| `figure4_delta_p.png` | Delta p-values heatmap (Appendix) |
| `figure5_ablation_improved.png` | Zero-ablation results |
| `figure6_logit_lens.png` | Logit lens — progressive prediction refinement |

---

## Reproducing the Paper

> ⚠️ **Run one model at a time.** T4 GPU RAM (15 GB) is insufficient to run
> all 6 models in a single session. Results accumulate across sessions via JSON.

### Step-by-step

**1. Open Google Colab with T4 GPU**
`Runtime → Change runtime type → T4 GPU`

**2. Upload to Colab**
```
run_experiments.py
multi_model_results_v2.json   ← only needed when continuing a previous session
```

**3. Set the target model** (line 37 of `run_experiments.py`)
```python
RUN_MODEL = "gpt2"   # edit this before each run
```

| Model | Approx. RAM | Approx. time |
|-------|------------|-------------|
| `gpt2` | ~6 GB | ~15 min |
| `EleutherAI/gpt-neo-125M` | ~6 GB | ~15 min |
| `EleutherAI/pythia-160m` | ~6 GB | ~15 min |
| `gpt2-medium` | ~9 GB | ~25 min |
| `EleutherAI/pythia-410m` | ~9 GB | ~25 min |
| `gpt2-large` | ~13 GB | ~40 min |

**4. Run all cells**

Results save to `multi_model_results_v2.json`; all figures are generated and downloaded automatically.

**5. Reset runtime between models**
`Runtime → Factory reset runtime`

This fully clears GPU RAM before loading the next model.

**6. Repeat for each model**

Re-upload `run_experiments.py` + the saved JSON, change `RUN_MODEL`, and run again.
Results accumulate in the JSON file across sessions.

### Tip: Use Google Drive to avoid re-uploading JSON

```python
from google.colab import drive
drive.mount('/content/drive')
# Then set in run_experiments.py:
OUTPUT_JSON = "/content/drive/MyDrive/pc_results/multi_model_results_v2.json"
```

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{yun2026predictive,
  title={Predictive Coding Tendencies in Transformers: 
         Evidence from Residual Stream Convergence, Activation Patching,
         MLP Transform Analysis, Zero-Ablation, and Logit Lens},
  author={Yun, Jong-O},
  journal={arXiv preprint},
  year={2026}
}
```
*(arXiv ID to be updated after submission)*

---

## Acknowledgements

This work uses [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) by Neel Nanda and Joseph Bloom.
Experiments were conducted on Google Colab.

---

## License

MIT License
