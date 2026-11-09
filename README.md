# Semi-Supervised MNIST: Reproducing and Extending Pseudo-Label (Lee, 2013)

PyTorch reproduction of [Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks](pseudo_label_final.pdf) (Dong-Hyun Lee, ICML 2013 Workshop on Challenges in Representation Learning), followed by an ablation study on extreme label scarcity ($n = 30$, 3 labels per class).

## What is in this repository

The project has two parts:

1. **Paper reproduction (`01` and `02`)**: Reproduces the original `DropNN`, `+PL`, and `+PL+DAE` (Denoising Autoencoder pre-training) benchmarks across $n \in \{100, 600, 1000, 3000\}$ labeled examples, along with the conditional entropy analysis from Table 1 of the paper.
2. **Low-label regime & confidence ablations (`03` to `06`)**: Pushes the setup down to $n = 30$ labeled samples (3 per class), where standard pseudo-labeling breaks down due to confirmation bias, and tests three modifications: dynamic confidence thresholding (`thr`), sticky pseudo-label locking (`lock`), and sample retirement (`retire`).

```text
semi-supervised-mnist/
├── 01_baseline/             # DropNN vs +PL reproduction (n = 100, 600, 1000, 3000)
├── 02_DAE_pretraining/      # Unsupervised DAE pre-training + PL
├── 03_low_label_baseline/   # Extreme scarcity benchmark (n = 30): DropNN vs +PL
├── 04_threshold_only/       # Control ablation: dynamic confidence masking (thr)
├── 05_sticky_lock/          # Permanent locking after consecutive confident checks (lock)
├── 06_lock_retire/          # Retiring locked samples from active pool (retire)
├── common/                  # Shared model, compiled training loop (pl.py), and lock state (lock.py)
└── pseudo_label_final.pdf   # Original paper (Lee, 2013)
```

## Architecture and training details

All experiments match the paper's architecture and optimization schedule:

- **Network (`DropNN`)**: $784 \to 500 \to 10$ single hidden layer MLP with ReLU activations, 50% dropout on the hidden layer (with $\times 0.5$ weight scaling), and independent sigmoid outputs trained with summed binary cross-entropy (BCE).
- **Optimizer**: Manual SGD with exponentially decaying learning rate ($\epsilon_0 = 1.5$, decay $0.998$ per epoch) and linearly annealed momentum ($p_0 = 0.5 \to p_f = 0.99$ over 500 epochs).
- **Pseudo-label schedule**: Unlabeled loss weight $\alpha(t) = 0$ for $t < 100$, linear ramp to $\alpha_f = 3.0$ between $t = 100$ and $t = 600$, and $\alpha_f = 3.0$ until epoch 1000.
- **Batching and speed**: Each step pairs 32 labeled samples (sampled with replacement) with 256 unlabeled samples. The inner training steps in `common/pl.py` are compiled with `torch.compile` and explicit `torch.autograd.grad` updates to avoid Python overhead over 1000 epochs (~105s per 1000-epoch run on an A100).

## Part 1: Paper reproduction ($n \in \{100, 600, 1000, 3000\}$)

Test error rates (%) averaged across 3 random seeds, compared against the figures reported in Lee (2013):

| $n_{\text{labeled}}$ | `DropNN` (Repro) | `DropNN` (Paper) | `+PL` (Repro) | `+PL` (Paper) | `+PL+DAE` (Repro) | `+PL+DAE` (Paper) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **100** | 22.16 ± 0.86% | 21.89% | 14.47 ± 2.08% | 16.15% | **7.56 ± 0.38%** | 10.49% |
| **600** | 10.02 ± 0.61% | 8.57% | 7.09 ± 0.44% | 5.03% | **4.39 ± 0.38%** | 4.01% |
| **1000** | 7.65 ± 0.22% | 6.59% | 6.01 ± 0.19% | 4.30% | **3.81 ± 0.17%** | 3.46% |
| **3000** | 4.64 ± 0.07% | 3.72% | 4.05 ± 0.06% | 2.80% | **3.14 ± 0.28%** | 2.69% |

### Conditional entropy ($n = 600$, Table 1 reproduction)

Lee (2013) shows that pseudo-labeling acts as an approximate entropy regularizer, pulling predictions toward low-entropy class clusters on the unlabeled and test splits:

| Split | `DropNN` (Repro) | `DropNN` (Paper) | `+PL` (Repro) | `+PL` (Paper) | `+PL+DAE` (Repro) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | $1.42 \times 10^{-4}$ | $2.63 \times 10^{-9}$ | $2.73 \times 10^{-6}$ | $6.10 \times 10^{-9}$ | $1.72 \times 10^{-4}$ |
| **Unlabeled** | 0.0619 | 0.0349 | 0.0005 | 0.0067 | 0.0147 |
| **Test** | 0.0578 | 0.0317 | 0.0164 | 0.0114 | 0.0274 |

## Part 2: Extreme label scarcity ($n = 30$, 3 labels/class)

At $n = 30$, the supervised baseline has ~35% test error when $\alpha(t)$ turns on at epoch 100. Assigning hard targets to every unlabeled sample injects heavy label noise. To isolate what helps in this regime, experiments `03` through `06` evaluate four modes across 3 paired seeds (seeds 42, 43, 44, sharing exact data splits and initial weights):

- **`dropNN`**: Supervised only ($\alpha = 0$).
- **`pl`**: Standard Pseudo-Label (every unlabeled sample gets $\arg\max$ target).
- **`thr`**: Dynamic confidence mask. Only unlabeled samples with $p_{\max} \ge 0.9$ and all other classes $< 0.3$ contribute to the unlabeled loss (re-evaluated every 5 epochs, never locked).
- **`lock`**: Sticky lock. Samples meeting the confidence criterion for `patience = 2` consecutive checks have their pseudo-label permanently frozen for the rest of training.
- **`retire`**: Same as `lock`, except locked samples are removed from the active unlabeled batch pool to reduce steps per epoch.

### Results at $n = 30$

| Mode | Val-Selected Test Err | Last-Epoch Test Err | Paired Diff vs `pl` (Val-sel) | Coverage | PL Precision | Time (s) | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `dropNN` | 35.36 ± 2.65% | 35.56 ± 2.65% | +0.77 ± 3.70% | N/A | N/A | 88.2s | 845 MB |
| `pl` | 34.59 ± 5.03% | 35.73 ± 5.84% | *Baseline* | 100% | N/A | 105.2s | 1067 MB |
| **`thr`** | **26.33 ± 5.03%** | **26.37 ± 4.99%** | **-8.03 ± 4.13%** | **97.42%** | **73.43%** | 110.2s | 1072 MB |
| `lock` | 33.39 ± 5.56% | 33.29 ± 5.40% | -1.20 ± 0.85% | 100.00% | 65.54% | 110.7s | 1072 MB |
| `retire` | 35.37 ± 2.64% | 51.67 ± 6.84% | +0.57 ± 2.99% | 99.68% | 55.46% | 91.2s | 1072 MB |

### Key findings from the ablations

1. **Vanilla Pseudo-Label stalls at $n = 30$**: `pl` barely beats `dropNN` on validation-selected test error (34.59% vs 35.36%) while doubling seed variance (5.03% vs 2.65%). On unfavorable splits (seed 43), forcing low-confidence pseudo-labels pushes last-epoch test error up to 41.89%.
2. **Dynamic thresholding (`thr`) drives the gain**: Masking unconfident samples cuts test error by **8.03 percentage points** on average (and up to 13.84 points on seed 44) without any late-epoch degradation (26.33% val-selected vs 26.37% last-epoch). Because targets are not frozen, the network can revise early mistakes as decision boundaries sharpen, reaching 73.43% pseudo-label precision at 97.42% coverage.
3. **Permanent locking (`lock`) freezes early mistakes**: While `lock` consistently edges out `pl` (-1.20 ± 0.85% paired difference), it trails `thr` by 7 points. With independent sigmoid outputs and BCE loss, confident-looking predictions ($p_{\max} \ge 0.9$) appear early even on misclassified digits. `lock` reaches 100% coverage and permanently freezes a 34.46% error rate into the unlabeled targets (65.54% precision vs 73.43% for `thr`).
4. **Retiring locked samples (`retire`) starves the unlabeled regularizer**: Removing locked samples cuts wall-clock time by ~17% (91.2s vs 109.5s), but last-epoch test error collapses to 51.67%. As easy digits leave the pool, batches concentrate on ambiguous samples; once 99.68% of the pool is locked, fewer than `BATCH_U = 256` samples remain (`steps = 0`) and training falls back to overfitting the 30 labeled examples. Peak VRAM remains unchanged (~1072 MB) because memory peaks at epoch 100 before any samples are retired.

## Running the experiments

Requirements: Python 3.10+, PyTorch 2.0+, `torchvision`, `numpy`.

Run the unit checks for the confidence mask and lock state machine:

```bash
python common/lock.py
```

Each experiment folder (`01_baseline` through `06_lock_retire`) contains a self-contained `train_eval.ipynb` notebook that can be executed locally or opened directly in Google Colab.

## Next steps and future work

- **Combine DAE pre-training with dynamic thresholding (`+DAE+thr`) at $n = 30$**: In Part 1, unsupervised DAE pre-training cut $n = 100$ error in half (from 14.47% to 7.56%). Starting `thr` from DAE features at $n = 30$ should substantially raise initial pseudo-label precision above 73%.
- **Delayed or annealed locking (`T_lock >= T2`)**: Prevent `lock` from freezing early errors by only allowing permanent locks after the $\alpha(t)$ ramp finishes at epoch 600, or by allowing a locked label to unlock if model confidence on that class drops below a threshold.
- **Class-balanced pseudo-label selection**: At $n = 30$, early confidence is uneven across classes, causing majority-class drift. Replacing the global threshold $p_{\max} \ge 0.9$ with per-class top-$k$ or quantile thresholding (similar to FlexMatch / curriculum pseudo-labeling) would keep class priors balanced during the ramp-up phase.
- **Soft retirement via sampling weights**: Instead of dropping locked samples completely in `retire` (which collapses the batch pool and ruins regularization), keep them in the pool with a reduced sampling probability (e.g., $0.1\times$) so easy anchors remain present without dominating compute.
- **Tighter confidence intervals**: Expand the $n = 30$ paired evaluation from 3 seeds to 10 seeds and log per-epoch coverage/precision curves to visualize exactly when `lock` diverges from `thr`.
