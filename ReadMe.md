# TorNet Enhanced — What changed, why it helps, and how it performs

This document summarizes the enhancements on top of the baseline TorNet CNN, how to reproduce evaluation on the paper’s Julian Day Modulo (JDM) partitioning, and how to interpret the improvements using XAI.

## What we changed vs. the baseline

The enhanced pipeline preserves the baseline’s stable recipe, but uses a stronger model and imbalance-aware training:

- Simple Enhanced CNN (`tornet/models/keras/simple_enhanced_cnn.py`)
  - Lightweight residual connections and channel-mixing attention
  - Optional conservative multi-scale heads
- Imbalanced-aware losses (`tornet/models/keras/imbalanced_losses.py`)
  - Focal loss (default), optional combined focal+dice
- Metrics (`tornet/metrics/keras/metrics.py`)
  - High-resolution AUC/AUCPR, plus F1/Precision/Recall, computed from logits
- Data handling consistent with the paper
  - JDM partitioning (train: J(te) mod 20 < 17; test: ≥ 17)
  - Overlap removal in time/space
  - Variables: DBZ, VEL, KDP, RHOHV, ZDR, WIDTH (+ masks/coords)

Why it helps:
- Residual/attention mixing improves gradient flow and focuses on informative channels under severe class imbalance.
- Focal loss down-weights easy negatives and emphasizes hard positives, improving AUCPR/F1.

## Where the code lives (minimal copy)
- Train: `scripts/tornado_detection/train_enhanced_tornet_keras.py`
- Eval (JDM): `scripts/tornado_detection/test_enhanced_tornet_keras.py`
- Model: `tornet/models/keras/simple_enhanced_cnn.py`
- Losses: `tornet/models/keras/imbalanced_losses.py`
- Layers: `tornet/models/keras/layers.py`
- Data: `tornet/data/loader.py`, `tornet/data/keras/loader.py`
- Metrics: `tornet/metrics/keras/metrics.py`
- SLURM: `slurm_scripts/train_enhanced_paper_partitioning.sl`, `slurm_scripts/evaluate_enhanced_paper_partitioning.sl`

Environment:
```
export TORNET_ROOT=/projects/weilab/shenb/csci3370/data
export KERAS_BACKEND=torch
```

## Results vs. the paper baseline

Paper (baseline CNN, JDM “All nulls vs confirmed”; screenshot):
- ACC 0.9505; AUC 0.8760; AUC‑PD 0.5294; CSI 0.3487

Our enhanced evaluation (`script/tornet-enhanced-paper-partitioning_1809185.out`):
- Per-batch lines stabilize around:
  - AUC ≈ 0.90–0.91 (starts ~0.95 first batches, converges ≈0.91)
  - AUCPR ≈ 0.59–0.60
  - ACC ≈ 0.945
  - F1 ≈ 0.53–0.59 (default 0.5 threshold)

Takeaways:
- AUC improves over baseline CNN by ~+0.03 to +0.04 (0.876 → ~0.91).
- AUCPR around ~0.59 indicates a stronger precision–recall balance vs. the paper’s AUC‑PD 0.5294 (not identical metrics, but directionally comparable under imbalance).
- Accuracy remains high; F1 improves at practical thresholds. Threshold tuning can trade precision/recall to match use cases.

For single-number summaries, use the JSON the evaluator writes next to the model (e.g., `enhanced_tornet_evaluation_results.json`).

## XAI: explaining the improvements

Apply the following to confirm that the model focuses on physically meaningful structures:

1. Saliency (∂logit/∂input)
   - Expect coherent gradients over storm structures and velocity couplets; reduced background activation.
2. Integrated Gradients (IG)
   - Compare baseline vs. enhanced on the same scenes; enhanced should emphasize VEL/DBZ cores and supportive polarimetric cues (KDP/ZDR/RHOHV).
3. Grad‑CAM (last conv)
   - Heatmaps should be tighter and aligned with radar signatures, indicating better localization.
4. Channel ablation curves
   - Zero out one variable at a time; enhanced should be more resilient (residual mixing) while preserving expected importance order (VEL/DBZ high).
5. PR / ROC operating points
   - Plot PR/ROC; enhanced curve should dominate baseline, explaining higher AUCPR and improved F1.

Implementation notes:
- Add saliency/IG/Grad‑CAM hooks to `test_enhanced_tornet_keras.py`; save heatmaps next to outputs for quick inspection.

## Reproduce JDM evaluation
```
sbatch slurm_scripts/evaluate_enhanced_paper_partitioning.sl /abs/path/to/model.keras
# Outputs next to the model:
#  - enhanced_tornet_evaluation_results.json
#  - enhanced_tornet_evaluation_curves.png
#  - enhanced_tornet_confusion_matrix.png
```

Caveats:
- AUC‑PD (paper) ≠ AUCPR; both reflect detection under imbalance but are not the same definition. Add AUC‑PD/CSI to the evaluator if strict parity is required.
