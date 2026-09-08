# LFW: Learnable Feature Weighting under a Fully Frozen Backbone

Code and full experimental results for the manuscript
*"Learnable Feature Weighting for Lightweight Transfer Learning under a Fully Frozen Backbone"*.

## Overview

LFW adapts a **fully frozen** backbone to a downstream task by learning only one
non-negative multiplicative weight per input feature. No backbone parameter is ever
inserted or modified, which makes the method applicable when the backbone is a
black box, served behind an API, contractually frozen, or stored on a device that
must remain bit-identical.

Key properties (all numbers reproducible from `results/`):

- **1:842** trainable-parameter ratio vs full fine-tuning (tabular setting)
- **+1.3 pp** mean macro-F1 over zero-shot transfer across 9 UCI datasets
- Wilcoxon-significant improvements over **5 of 12** baselines (alpha = 0.05, n = 45)
- Graceful degradation under label noise (beats full fine-tuning on 7/9 datasets at
  30% noise) and input noise; a built-in safeguard falls back to zero-shot when
  the learned weights do not improve validation macro-F1
- Instance-level interpretability directly from the learned weights

A supplementary study on a compact vision Transformer (MNIST→USPS,
CIFAR-10→CIFAR-10-C) validates the same formulation on the architecture family
for which BitFit/SSF/Adapter/LoRA were designed, and delineates the applicability
boundary of input-only adaptation (Section 4.8 of the manuscript).

## Repository structure

| File | Purpose |
|---|---|
| `code/data_sci.py` | UCI dataset loading and preprocessing |
| `code/data_utils.py` | Shared data utilities |
| `code/mlp.py` | MLP backbone, LFW, and NumPy baselines |
| `code/peft.py` | BitFit / SSF / Adapter / LoRA baselines |
| `code/run_sci.py` | Main protocol: 9 datasets × 13 methods × 5 seeds |
| `code/run_sci_vision.py` | Vision-Transformer validation (resumable) |
| `code/make_figures_sci.py` | All figures (300 DPI PNG) |
| `code/build_paper_sci.py` | Rebuild the manuscript (part 1) from results |
| `code/build_paper_sci_part2.py` | Rebuild the manuscript (part 2) |
| `code/verify_paper_sci.py` | Cell-level manuscript-vs-CSV consistency checks |
| `code/refs_*.json` | Bibliography databases |
| `results/*.csv` | Raw per-seed experimental outputs backing every table |
| `results/*.json` | Analysis summaries (statistics, hyper-parameters, provenance) |

## Requirements

Python 3.12 with:

```
numpy==1.26.4
scipy==1.13.1
scikit-learn==1.4.2
matplotlib==3.8.4
python-docx==1.2.0
torch==2.5.1
imagecorruptions==1.1.2
```

Install with `pip install -r requirements.txt`.

## Reproducing

```bash
python code/run_sci.py          # tabular experiments (CPU, hours)
python code/run_sci_vision.py   # vision experiments (CPU, ~24 h, resumable)
python code/make_figures_sci.py # regenerate all figures
python code/build_paper_sci.py  # rebuild the .docx manuscript
python code/verify_paper_sci.py # verify every number against the CSVs
```

Pretrained backbones are regenerated deterministically (seeded) by the run
scripts; `results/` contains the exact per-seed outputs used in the manuscript,
so tables and figures can be rebuilt without re-running the experiments.

## Data

All datasets are public: UCI/OpenML datasets (HAR, Bank, Adult, Satimage,
Digits, Spambase, Credit-g, WDBC, Heart) for the tabular study; MNIST, USPS,
CIFAR-10, and CIFAR-10-C (severity 3, generated with `imagecorruptions`
following the official corruption recipes) for the vision study.

## License

MIT — see [LICENSE](LICENSE).
