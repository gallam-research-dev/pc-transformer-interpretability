# ================================================================
# Figure 5 & 6 수정
# - Figure 6: GPT-2 계열 / 기타 계열 2패널로 분리
# - Figure 5: y축 고정 + Pythia 160M 이상값 주석
# JSON만 있으면 실험 재실행 불필요
# ================================================================

import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

JSON_FILE = "multi_model_results_v2.json"
with open(JSON_FILE) as f:
    data = json.load(f)

ML = {
    "gpt2":         "GPT-2 (117M)",
    "gpt2-medium":  "GPT-2 Med (345M)",
    "gpt2-large":   "GPT-2 Large (774M)",
    "gpt-neo-125M": "GPT-Neo (125M)",
    "pythia-160m":  "Pythia (160M)",
    "pythia-410m":  "Pythia (410M)",
}
MC = {
    "gpt2":         "#1f77b4",
    "gpt2-medium":  "#aec7e8",
    "gpt2-large":   "#0a4a7a",
    "gpt-neo-125M": "#d62728",
    "pythia-160m":  "#9467bd",
    "pythia-410m":  "#8c564b",
}
ZC = {
    "early_spike": "#e74c3c",
    "convergence": "#3498db",
    "late_spike":  "#e67e22",
}

# ── Figure 5 수정 ────────────────────────────────────────────────
print("[Figure 5] 수정 중...")

zones  = ["early_spike", "convergence", "late_spike"]
ZL     = {
    "early_spike": "Early\nSpike\n(0.10–0.20)",
    "convergence": "Convergence\n(0.25–0.75)",
    "late_spike":  "Late\nSpike\n(0.80–0.95)",
}

# 모델 순서 고정
MODEL_ORDER = ["gpt2", "gpt2-medium", "gpt2-large",
               "gpt-neo-125M", "pythia-160m", "pythia-410m"]
md = [m for m in MODEL_ORDER
      if m in data and data[m].get("D_zero_ablation")]

fig5, axes5 = plt.subplots(1, len(md), figsize=(4*len(md), 5.5), sharey=True)
if len(md) == 1: axes5 = [axes5]

for ax, m in zip(axes5, md):
    res   = data[m]["D_zero_ablation"]
    means = [res.get(z, {}).get("mean_drop", 0) for z in zones]
    stds  = [res.get(z, {}).get("std_drop",  0) for z in zones]
    ns    = [res.get(z, {}).get("n_pairs",   0) for z in zones]

    # y축 범위: -0.5 ~ 1.5 고정 (이상값 클리핑)
    clipped_means = [max(-0.5, min(1.5, v)) for v in means]
    clipped_stds  = [min(s, 0.5) for s in stds]

    bars = ax.bar(range(3), clipped_means,
                  color=[ZC[z] for z in zones],
                  alpha=0.85, width=0.6,
                  yerr=clipped_stds, capsize=5,
                  edgecolor="white")

    ax.axhline(0, color="black", lw=1, ls="--", alpha=0.6)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels([ZL[z] for z in zones], fontsize=9)
    ax.set_title(ML.get(m, m), fontsize=11, fontweight="bold")
    if ax == axes5[0]:
        ax.set_ylabel("Performance Drop\n(+= harmful, -= helpful)", fontsize=10)
    ax.grid(axis="y", alpha=0.3, ls="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, mean, orig_mean, nv in zip(bars, clipped_means, means, ns):
        # 클리핑된 경우 실제 값 표시
        display = f"{orig_mean:+.2f}\n(n={nv})"
        if abs(orig_mean) > 1.5:
            display = f"{orig_mean:+.2f}*\n(n={nv})"
        ax.text(bar.get_x() + bar.get_width()/2,
                max(-0.45, min(1.45, mean)) + 0.03,
                display,
                ha="center", fontsize=8, fontweight="bold")

fig5.suptitle(
    "Figure 5: Zero-Ablation of MLP Zones\n"
    "Convergence zone shows largest drop despite lowest MLP magnitude  "
    "(*value clipped for display)",
    fontsize=12, fontweight="bold"
)
plt.tight_layout()
plt.savefig("figure5_ablation_improved.png", dpi=150,
            bbox_inches="tight", facecolor="white")
print("  figure5_ablation_improved.png 저장 완료")

# ── Figure 6 수정 ────────────────────────────────────────────────
print("[Figure 6] 수정 중...")

# GPT-2 계열 / 나머지 계열로 분리
gpt2_models  = [m for m in MODEL_ORDER
                if "gpt2" in m and m in data
                and data[m].get("E_logit_lens")]
other_models = [m for m in MODEL_ORDER
                if "gpt2" not in m and m in data
                and data[m].get("E_logit_lens")]

fig6, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(16, 6))

# 왼쪽: GPT-2 계열
for m in gpt2_models:
    ranks = data[m]["E_logit_lens"]
    nl    = len(ranks)
    x     = [i/(nl-1) for i in range(nl) if ranks[i] is not None]
    y     = [r for r in ranks if r is not None]
    ax_l.plot(x, y, marker="o", markersize=4, linewidth=2.2,
              color=MC[m], label=ML[m], alpha=0.9)

ax_l.axvspan(0.10, 0.20, alpha=0.08, color="red")
ax_l.axvspan(0.25, 0.75, alpha=0.08, color="blue")
ax_l.axvspan(0.80, 0.95, alpha=0.08, color="orange")
ax_l.set_xlabel("Normalized Layer Position", fontsize=12)
ax_l.set_ylabel("Avg Rank of Target Token\n(lower = better)", fontsize=11)
ax_l.set_title("GPT-2 Family", fontsize=13, fontweight="bold")
ax_l.legend(fontsize=10)
ax_l.grid(alpha=0.3, ls="--")
ax_l.invert_yaxis()
ax_l.spines["top"].set_visible(False)
ax_l.spines["right"].set_visible(False)

# 오른쪽: GPT-Neo + Pythia
for m in other_models:
    ranks = data[m]["E_logit_lens"]
    nl    = len(ranks)
    x     = [i/(nl-1) for i in range(nl) if ranks[i] is not None]
    y     = [r for r in ranks if r is not None]
    ax_r.plot(x, y, marker="o", markersize=4, linewidth=2.2,
              color=MC[m], label=ML[m], alpha=0.9)

ax_r.axvspan(0.10, 0.20, alpha=0.08, color="red")
ax_r.axvspan(0.25, 0.75, alpha=0.08, color="blue")
ax_r.axvspan(0.80, 0.95, alpha=0.08, color="orange")
ax_r.set_xlabel("Normalized Layer Position", fontsize=12)
ax_r.set_ylabel("Avg Rank of Target Token\n(lower = better)", fontsize=11)
ax_r.set_title("GPT-Neo & Pythia Family", fontsize=13, fontweight="bold")
ax_r.legend(fontsize=10)
ax_r.grid(alpha=0.3, ls="--")
ax_r.invert_yaxis()
ax_r.spines["top"].set_visible(False)
ax_r.spines["right"].set_visible(False)

fig6.suptitle(
    "Figure 6: Logit Lens — Progressive Prediction Refinement\n"
    "Red=Early spike | Blue=Convergence | Orange=Late spike  "
    "(Left: GPT-2 family shows clear monotonic refinement)",
    fontsize=13, fontweight="bold"
)
plt.tight_layout()
plt.savefig("figure6_logit_lens.png", dpi=150,
            bbox_inches="tight", facecolor="white")
print("  figure6_logit_lens.png 저장 완료")

# ── 다운로드 ─────────────────────────────────────────────────────
try:
    from google.colab import files
    files.download("figure5_ablation_improved.png")
    files.download("figure6_logit_lens.png")
    print("다운로드 완료")
except ImportError:
    print("(Colab 외 환경)")

print("\n✅ 완료!")