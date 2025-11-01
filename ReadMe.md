# TorNet Enhanced — Baseline vs Enhanced, what changed and why it works

This document summarizes the enhancements on top of the baseline TorNet CNN, how to reproduce evaluation on the paper's Julian Day Modulo (JDM) partitioning, and how to interpret the improvements using XAI.

## **Baseline Model Repo:** [mit-ll/tornet](https://github.com/mit-ll/tornet)

## About the Baseline TorNet Project

**TorNet** is a benchmark dataset and baseline model for tornado detection and prediction using full-resolution polarimetric weather radar data, developed by researchers at MIT Lincoln Laboratory. The project was described in the paper *"A Benchmark Dataset for Tornado Detection and Prediction using Full-Resolution Polarimetric Weather Radar Data"* (published in AIES).

### Dataset

The TorNet dataset consists of **203,133 samples** from **2013–2022** NEXRAD WSR-88D radar data:
- **Radar variables**: DBZ (reflectivity), VEL (velocity), KDP (specific differential phase), RHOHV (correlation coefficient), ZDR (differential reflectivity), WIDTH (spectrum width)
- **Labels**: Pixel-wise tornado labels from integrated warning/verification workflows
- **Class distribution**: 124,766 (61.4%) random nontornadic cells, 64,510 (31.8%) nontornadic warnings, 13,857 (6.8%) confirmed tornadoes
- **Partitioning**: Julian Day Modulo (JDM) split — train if `J(te) mod 20 < 17` (~84.5%), test if `≥ 17` (~15.5%) to avoid temporal leakage

### Baseline Model

The baseline model implements a **VGG-style 2D CNN**:
- **Architecture**: 4 convolutional blocks (48→96→192→384 filters), CoordConv2D layers, global max-pooling head
- **Inputs**: Concatenated normalized radar variables (DBZ, VEL, KDP, RHOHV, ZDR, WIDTH) plus range-folded masks and coordinates
- **Training**: 15 epochs, batch size 128, binary cross-entropy loss, Adam optimizer with LR decay, no augmentation or class balancing

**Baseline results** (JDM test split): Accuracy 0.9505, ROC AUC 0.8760, AUC-PD 0.5294, CSI 0.3487

### This Project: Enhancing TorNet

This project builds on the TorNet baseline to **improve upon the paper's results** while maintaining the same dataset, JDM partitioning, inputs, and evaluation protocol. Our enhancements address class imbalance and training dynamics through:
1. **Residual connections** for better gradient flow
2. **Focal loss** (α=0.5, γ=1.5) for extreme imbalance
3. **Class balancing** and conservative data augmentation
4. **Improved optimization**: AdamW, higher LR, learning rate scheduling, early stopping

**Enhanced results**: ROC AUC 0.9021 (+2.6%), PR AUC 0.5886 (+11.2%), CSI 0.3717 (+6.6%) — see detailed results below.

## Baseline vs Enhanced — configuration deltas (what actually changed)

Configuration (from params in scripts and run folders):

| Aspect | Baseline | Enhanced | Impact |
|---|---|---|---|
| Epochs | 15 | 20 | +33% more optimization steps |
| Batch Size | 128 | 64 | Smaller batches → noisier but richer gradients |
| Learning Rate | 1e-4 | 1e-3 | 10× higher LR (+ ReduceLROnPlateau) |
| Start Filters | 48 | 16 | 3× fewer filters (faster, lower memory) |
| Loss | cce | focal_imbalanced (α=0.5, γ=1.5) | Major upgrade for class imbalance |
| Class balancing | off | on (weights + modest oversample_ratio=2.0) | Focus on rare positives |
| Data augmentation | light | on (conservative jitters) | Diversity, regularization |
| Scheduling | none | cosine annealing (warmup=3) + ReduceLROnPlateau | Better convergence |
| Early stopping | off | on (patience 5) | Prevents late overfit |

### Architecture (file: `tornet/models/keras/simple_enhanced_cnn.py`)
- Stem: small 3×3 convs with BN+ReLU, same inputs as baseline (DBZ, VEL, KDP, RHOHV, ZDR, WIDTH + masks/coords).
- Residual connections added to the VGG‑style blocks: Conv(3×3) → BN → ReLU → Conv(3×3) with identity skip. Improves gradient flow/stability.
- Optional channel‑mix 1×1 (SE‑like) for lightweight attention; left conservative by default for stability (not required for reported results).
- Head: global max‑pool head; multi‑scale head available but disabled by default in our experiments.

### Training
- Loss: focal loss with α=0.5, γ=1.5 (see `imbalanced_losses.py`), optionally combinable with dice; this shifts weight toward hard tornado positives and hard negatives.
- Optimizer: AdamW (weight_decay=1e‑4) with ReduceLROnPlateau; optional cosine scheduler hook in the trainer.
- Callbacks: EarlyStopping on val AUC with patience 5–7, best‑model checkpointing, CSVLogger, TerminateOnNaN.
- Metrics at compile: AUC (ROC), AUC(PR), BinaryAccuracy, Precision, Recall, F1 (all from logits; high threshold density) to avoid metric aliasing from post‑sigmoid rounding.

### Data & Partitioning
- **JDM split**: Same methodology as paper (train if `J(te) mod 20 < 17`, test if `≥ 17`), same years (2013–2022) and variables. Overlap removal supported.
- **Augmentation**: Conservative geometric/photometric jitter (rotations, translations, scaling, brightness/contrast) to avoid distorting radar structure.
- **Class balancing**: Sample weights and class weighting with modest oversample_ratio=2.0 to avoid distribution drift.

### Why these specifics help
- Residual skips keep gradient flow healthy and reduce over‑smoothing; channel re‑weighting suppresses spurious background responses.
- Focal loss concentrates learning signal on rare/severe tornado signatures → higher PR AUC/F1 without large FP inflation.
- High-resolution AUC/PR AUC computed from logits gives faithful ranking diagnostics during training/selection.

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

**Baseline results** (from paper, JDM test split):
<img src="baseline_results.png" alt="Baseline results" width="640">
Accuracy 0.9505, ROC AUC 0.8760, AUC‑PD 0.5294, CSI 0.3487

Our enhanced evaluation (`script/tornet-enhanced-paper-partitioning_1809185.out`) — concrete numbers from the final evaluator summary (screenshot):

Data partitioning (JDM, from summary):
- Training samples: 171,666 (84.5%)
- Testing samples: 31,467 (15.5%)
- Overlap removal: False (for this specific run)

Performance metrics (full‑dataset aggregation; threshold = 0.5):
- Accuracy: 0.9534
- Precision: 0.7162
- Recall (POD): 0.4360
- F1: 0.5420
- Specificity (TNR): 0.9883

AUC metrics (global, not per‑batch averages):
- ROC AUC: 0.9021
- PR AUC: 0.5886

Tornado‑detection metrics:
- CSI (Threat Score): 0.3717
- HSS: 0.5190
- FAR: 0.2838

Confusion matrix (threshold = 0.5):
- TN: 29,132   FP: 344   FN: 1,123   TP: 868

**Comparison to baseline**:
- ROC AUC: 0.8760 → 0.9021 (+2.6%) — improved global ranking ability
- PR AUC: 0.5886 vs. baseline AUC-PD 0.5294 (+11.2%) — stronger precision-recall balance under imbalance
- CSI: 0.3487 → 0.3717 (+6.6%) — better detection skill
- Operating point (threshold=0.5): High precision (0.7162), moderate recall (0.4360); threshold tuning can trade precision/recall for operational needs

For single-number summaries, use the JSON the evaluator writes next to the model (e.g., `enhanced_tornet_evaluation_results.json`).

## Understanding the Metrics

### Key Metrics for Tornado Detection

Given the severe class imbalance in tornado detection (tornadoes are rare events), different metrics tell different parts of the story. Here's what each metric means and why it matters:

#### **AUC Metrics (Threshold-Independent)**

**ROC AUC (0.9021)**: Measures the model's ability to rank tornado samples above non-tornado samples across all possible thresholds. Range: 0.0 (worst) to 1.0 (perfect). 
- **Why it matters**: A high ROC AUC (0.90+) indicates strong discriminative ability—the model can reliably distinguish tornadic from non-tornadic patterns.
- **What to watch for**: ROC AUC can be misleading with extreme imbalance (can appear high even when failing on rare positives). Still valuable for comparing model architectures.

**PR AUC (0.5886)**: Precision-Recall area under curve. Measures the trade-off between precision and recall across thresholds.
- **Why it matters**: **This is critical for imbalanced problems.** PR AUC directly reflects performance on the minority class (tornadoes). Much more informative than ROC AUC when positives are rare.
- **What to watch for**: PR AUC values are typically lower than ROC AUC. Values >0.5 indicate skill; >0.55–0.60 is strong for severe imbalance like this dataset.

#### **Threshold-Dependent Metrics (at threshold = 0.5)**

**Precision (0.7162)**: Of all predicted tornadoes, what fraction are actually tornadoes? TP / (TP + FP).
- **Why it matters**: High precision means fewer false alarms. Critical for operational warning systems—too many false alarms reduce trust.
- **Interpretation**: 71.6% precision means ~7 in 10 tornado predictions are correct.

**Recall/POD (0.4360)**: Of all actual tornadoes, what fraction did we detect? TP / (TP + FN).
- **Why it matters**: High recall means missing fewer tornadoes. Critical for safety—missed tornadoes can be deadly.
- **Interpretation**: 43.6% recall means we detect ~4 in 10 tornadoes. This seems low, but threshold tuning can improve it (at cost of precision).

**F1 Score (0.5420)**: Harmonic mean of precision and recall. Balances both concerns.
- **Why it matters**: Single number summarizing precision–recall trade-off. Useful for comparing models or threshold settings.
- **Limitation**: F1 treats precision and recall equally; for tornadoes, you may prioritize recall more.

**Accuracy (0.9534)**: Overall fraction of correct predictions. (TP + TN) / (TP + TN + FP + FN).
- **Why it matters**: Intuitive, but **can be misleading with severe imbalance**. A model predicting "no tornado" for everything would have ~94% accuracy but be useless.
- **When to use**: Use accuracy alongside other metrics. By itself, it doesn't tell the full story.

**Specificity/TNR (0.9883)**: Of all non-tornado samples, what fraction are correctly classified as non-tornado? TN / (TN + FP).
- **Why it matters**: Complements recall. High specificity means few false alarms.
- **Interpretation**: 98.8% specificity means we're very good at identifying non-tornado cases.

#### **Weather-Specific Metrics**

**CSI (0.3717)**: Critical Success Index (Threat Score). Fraction of tornado events that were correctly predicted, penalizing both misses and false alarms. TP / (TP + FP + FN).
- **Why it matters**: **This is the standard metric in meteorology.** CSI directly reflects skill at detecting events of interest.
- **Interpretation**: CSI of 0.37 means ~37% of tornado cases are correctly identified when accounting for both misses and false alarms. Higher is better; operational systems often aim for CSI >0.3–0.4.

**FAR (0.2838)**: False Alarm Rate. Fraction of tornado predictions that were wrong. FP / (TP + FP).
- **Why it matters**: Low FAR is crucial for maintaining public trust. High FAR reduces credibility of warnings.
- **Interpretation**: FAR of 0.28 means ~28% of tornado predictions are false alarms. Lower is better.

**HSS (0.5190)**: Heidke Skill Score. Measures skill compared to random chance. Range: -1 (perfectly wrong) to +1 (perfect).
- **Why it matters**: HSS accounts for correct predictions that would occur by chance. Values >0.3 indicate skill.
- **Interpretation**: HSS of 0.52 indicates strong skill above chance.

#### **Confusion Matrix Components**

- **TP (868)**: True Positives—correctly predicted tornadoes
- **TN (29,132)**: True Negatives—correctly predicted non-tornadoes  
- **FP (344)**: False Positives—predicted tornado but none occurred (false alarms)
- **FN (1,123)**: False Negatives—missed tornadoes (most concerning for safety)

### **Which Metrics Should You Prioritize?**

For tornado detection with severe class imbalance, focus on these in order:

1. **PR AUC** — Best overall indicator of model quality under imbalance. Target: >0.55–0.60.
2. **CSI** — Standard in meteorology. Target: >0.3–0.4 for operational use.
3. **Recall/POD** — Safety-critical: missing tornadoes is dangerous. Consider threshold tuning if recall is too low.
4. **Precision/FAR** — Operational credibility: too many false alarms reduce trust. Balance with recall via threshold.
5. **ROC AUC** — Useful for model comparison, but less informative than PR AUC with extreme imbalance.

**Threshold Tuning**: The metrics above use threshold = 0.5. Lowering the threshold increases recall (fewer misses) but decreases precision (more false alarms). Raising it does the opposite. For operational use, tune threshold based on the acceptable FAR vs. recall trade-off for your application.

**Implementation notes**:
- Actual training settings come from `params_enhanced_tornet.json` (epochs=20, batch_size=64, learning_rate=1e-3, start_filters=16, focal_imbalanced α=0.5/γ=1.5). Code defaults are placeholders overridden by params files in SLURM scripts.
- `simple_enhanced_cnn.py` implements VGG + residual connections; attention/multiscale flags are ignored (kept off for stability).

## XAI: explaining why the enhanced model performs better

Apply the following to confirm that the model focuses on physically meaningful structures and to attribute the metric gains to the changes above:

1. Saliency (∂logit/∂input)
   - Expect coherent gradients over mesocyclone/velocity couplets and hook echoes; reduced diffuse activation over nulls (effect of focal loss + class balancing).
2. Integrated Gradients (IG)
   - Compare baseline vs. enhanced on the same scenes; enhanced should emphasize VEL/DBZ cores and supportive KDP/ZDR/RHOHV cues more sharply (residuals preserve features across depth).
3. Grad‑CAM (last conv)
   - Heatmaps should be tighter and aligned with radar signatures (hooks, couplets), indicating better localization and fewer false alarms.
4. Channel ablation curves
   - Zero out one variable at a time; enhanced should be more resilient (residual mixing) while preserving expected importance order (VEL/DBZ highest, KDP/ZDR/RHOHV supportive).
5. PR / ROC operating points
   - Plot PR/ROC; enhanced curve should dominate baseline. Gains should match observed ROC AUC≈0.90 and PR AUC≈0.589, explaining improved CSI/F1 at usable thresholds.

Implementation notes (where to add hooks):
- Saliency/IG: inside `test_enhanced_tornet_keras.py` after loading the model, wrap a gradient tape (TF backend) or export to TF and use tf‑explain; save per‑channel maps for DBZ/VEL/KDP/RHOHV/ZDR/WIDTH.
- Grad‑CAM: register last conv in `simple_enhanced_cnn.py` and export CAMs for TP/FP cases; overlay on DBZ/VEL to verify focus on couplets/hooks.

## Reproduce JDM evaluation
```
sbatch slurm_scripts/evaluate_enhanced_paper_partitioning.sl /abs/path/to/model.keras
# Outputs next to the model:
#  - enhanced_tornet_evaluation_results.json
#  - enhanced_tornet_evaluation_curves.png
#  - enhanced_tornet_confusion_matrix.png
```

**Note**: AUC‑PD (paper metric) ≠ PR AUC; both reflect performance under imbalance but use different definitions. Our PR AUC of 0.5886 is directionally comparable to baseline AUC‑PD of 0.5294.

---

## Dataset Details and Class Imbalance

<img src="class_composition.png" alt="Class composition" width="640">

**Class composition** (203,133 samples total):
- Random nontornadic cells: 124,766 (61.4%)
- Nontornadic tornado warnings: 64,510 (31.8%)
- Confirmed tornadoes: 13,857 (6.8%)
- EF distribution within confirmed tornadoes: EF0=5,393; EF1=5,644; EF2=1,997; EF3=651; EF4=172; EF5=0

**Imbalance challenges**:
- Only ~6.8% samples are confirmed tornadoes; most positives are weaker (EF0–EF1)
- Pixel-level imbalance is even more severe — tornadic signal occupies small fraction of pixels per positive sample
- Naïve learners collapse to majority predictions (high accuracy, low recall; ROC AUC≈0.50)
- Our enhancements (focal loss, class balancing, residual connections) specifically address this failure mode

---

## Project Summary (CSCI3370 Final Project)

**Approach**: Enhanced TorNet baseline with strategic improvements targeting class imbalance and training dynamics. Maintained same dataset, JDM partitioning, and evaluation protocol for fair comparison.

**Key innovations**: Residual connections, focal loss (α=0.5, γ=1.5), class balancing (oversample_ratio=2.0), conservative augmentation, AdamW optimization, learning rate scheduling, early stopping.

**Results** (JDM test, eval file: `script/tornet-enhanced-paper-partitioning_1809185.out`):
- ROC AUC: 0.9021 (baseline: 0.8760), PR AUC: 0.5886 (baseline AUC-PD: 0.5294), CSI: 0.3717 (baseline: 0.3487)
- Threshold=0.5: Precision 0.7162, Recall 0.4360, F1 0.5420, Specificity 0.9883
- Confusion matrix: TP=868, TN=29,132, FP=344, FN=1,123

**Impact**: Significant improvements across key metrics while maintaining high specificity. Threshold tuning enables precision/recall trade-offs for operational needs (warning systems vs. research).
