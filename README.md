# Predictive Coding Tendencies in the GPT-2 Family

> **Multi-Method Empirical Evidence from Residual Stream Convergence,
> Activation Patching, MLP Transform Analysis, Zero-Ablation, and Logit Lens**

📄 **Paper**: [arXiv link] *(to be updated after submission)*
👤 **Author**: Jong-O Yun — Independent Researcher, South Korea
📧 [gallam.research@gmail.com](mailto:gallam.research@gmail.com)
🔗 <https://github.com/gallam-research-dev/pc-transformer-interpretability>

-----

## Overview

This repository contains code and data for the paper:

> *“Predictive Coding Tendencies in the GPT-2 Family: Multi-Method Empirical
> Evidence from Residual Stream Convergence, Activation Patching, MLP Transform
> Analysis, Zero-Ablation, and Logit Lens”*

I investigate whether GPT-2 family language models implicitly exhibit
computational tendencies consistent with **Predictive Coding (PC)** without
explicit design, using GPT-Neo 125M and Pythia 160M/410M as architectural
controls.

**Direct answer**: I find no mechanistic proof that transformers *implement* PC,
but consistent multi-method evidence that the GPT-2 family exhibits
PC-*like tendencies*.

-----

## Models

|Model       |Params|Layers|Training Data|Tokenizer|
|------------|------|------|-------------|---------|
|GPT-2 Small |117M  |12    |WebText      |GPT-2 BPE|
|GPT-2 Medium|345M  |24    |WebText      |GPT-2 BPE|
|GPT-2 Large |774M  |36    |WebText      |GPT-2 BPE|
|GPT-Neo 125M|125M  |12    |The Pile     |GPT-2 BPE|
|Pythia 160M |160M  |12    |The Pile     |NeoX     |
|Pythia 410M |410M  |24    |The Pile     |NeoX     |

-----

## Key Findings

1. **Training-configuration-dependent convergence**: English ranks first in
   mid-layer cosine similarity in all GPT-2 models; this pattern weakens in
   GPT-Neo and Pythia, implicating training configuration over architecture.
1. **Mid-layer semantic specialization**: Activation patching identifies
   normalized position 0.25–0.33 as the peak causal locus across all 6 models
   (range: 0.11–0.21).
1. **W-shaped MLP magnitude**: A spike–convergence–spike structure in GPT-2
   is scale-invariant; GPT-Neo and Pythia show a dual-spike variant.
1. **Convergence zone functional primacy**: The *lowest*-magnitude zone causes
   the *largest* performance drop when removed. All GPT-2 95% CIs exclude zero:
   [0.25, 0.44], [0.36, 0.75], [0.53, 1.15].
1. **Monotonic prediction refinement**: Logit lens confirms Early Spike rank
   reduction rates 17–110× faster than the Convergence zone across GPT-2 models.

-----

## Setup

```bash
git clone https://github.com/gallam-research-dev/pc-transformer-interpretability
cd pc-transformer-interpretability
pip install -r requirements.txt
```

All experiments were run on **Google Colab T4 GPU**.

-----

## Reproducing the Paper

> ⚠️ **Run one model at a time.** T4 GPU RAM (15 GB) cannot hold all 6 models
> in a single session. Results accumulate across sessions via JSON.

### Step-by-step

**1. Open Google Colab with T4 GPU**
`Runtime → Change runtime type → T4 GPU`

**2. Upload files**

```
run_experiments.py
multi_model_results_v2.json   ← only needed when continuing a previous session
```

**3. Set the target model** — edit line 37 of `run_experiments.py`:

```python
RUN_MODEL = "gpt2"   # change this each run
```

|Model                    |Approx. RAM|Approx. time|
|-------------------------|-----------|------------|
|`gpt2`                   |~6 GB      |~15 min     |
|`EleutherAI/gpt-neo-125M`|~6 GB      |~15 min     |
|`EleutherAI/pythia-160m` |~6 GB      |~15 min     |
|`gpt2-medium`            |~9 GB      |~25 min     |
|`EleutherAI/pythia-410m` |~9 GB      |~25 min     |
|`gpt2-large`             |~13 GB     |~40 min     |

**4. Run all cells**

Results save to `multi_model_results_v2.json`.
All 6 figures are generated and downloaded automatically.

**5. Reset runtime between models**
`Runtime → Factory reset runtime`

**6. Repeat for each model**

Re-upload `run_experiments.py` + saved JSON, change `RUN_MODEL`, run again.

### Tip: Google Drive (recommended)

```python
from google.colab import drive
drive.mount('/content/drive')
# Set in run_experiments.py:
OUTPUT_JSON = "/content/drive/MyDrive/pc_results/multi_model_results_v2.json"
```

-----

## Repository Structure

```
pc-transformer-interpretability/
├── run_experiments.py        # All 5 experiments + all 6 figures (single script)
├── requirements.txt
├── README.md
├── LICENSE
├── figures/                  # All figures used in the paper
│   ├── figure1.png           # Mid-layer cosine similarity (6 models)
│   ├── figure2.png           # MLP transform magnitude by family
│   ├── figure3.png           # Activation patching heatmap
│   ├── figure4_delta_p.png   # delta_p heatmap (Appendix)
│   ├── figure5_ablation_improved.png  # Zero-ablation (clipped y-axis)
│   └── figure6_logit_lens.png         # Logit lens (split panels)
└── result/
    └── multi_model_results_v2.json    # All experiment results (Exp 1–5)
```

-----

## Citation

```bibtex
@article{yun2026predictive,
  title={Predictive Coding Tendencies in the GPT-2 Family:
         Multi-Method Empirical Evidence from Residual Stream Convergence,
         Activation Patching, MLP Transform Analysis, Zero-Ablation,
         and Logit Lens},
  author={Yun, Jong-O},
  journal={arXiv preprint},
  year={2026}
}
```

*(arXiv ID to be updated after submission)*

-----

## Acknowledgements

This work uses [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)
by Neel Nanda and Joseph Bloom. Experiments were conducted on Google Colab.

-----

## License

MIT License