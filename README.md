# Anonymous Code Repository

Code for the paper:
**A Probabilistic Source-Free Domain Adaptation Method  for Concept Bottleneck Models**.

Not included:
- Raw medical images (PBC, RaabinWBC, Fitzpatrick17k, DDI)

## Setup

```bash
pip install -r requirements.txt
```

## Dataset Path Configuration (No Code Edits Required)

`data_utils.py` reads dataset roots from:
1. `config/paths.yaml` (if present), then
2. environment overrides `LF_CBM_PATH_<KEY>`.
You can point to a different file with `LF_CBM_PATHS_FILE=/abs/path/to/paths.yaml`.

To configure quickly:

```bash
cp config/paths.example.yaml config/paths.yaml
# then edit config/paths.yaml for your machine
```

Example env override:

```bash
export LF_CBM_PATH_PBC_ROOT=/path/to/PBC_dataset_normal_DIB
```

## 1) Adaptation Runs

### PBC -> RaabinWBC TestA (cosine PL)

```bash
python adapt_cbm_conda.py --config config/adapt/pbc_to_raabin_cosine.yaml
```

### PBC -> RaabinWBC TestA (KL / PL-CBM)

```bash
python adapt_cbm_conda.py --config config/adapt/pbc_to_raabin_kl.yaml
```

### FitzSkin -> DDI (cosine PL)

```bash
python adapt_cbm_conda.py --config config/adapt/fitzskin_to_ddi_cosine.yaml
```

### FitzSkin -> DDI (KL / PL-CBM)

```bash
python adapt_cbm_conda.py --config config/adapt/fitzskin_to_ddi_kl.yaml
```

## 2) Source CBM Training

### FitzSkin source CBM

```bash
python train_cbm_supervised.py --config config/train/fitzskin_s999.yaml
```

### PBC source CBM

```bash
python train_cbm_supervised.py --config config/train/pbc_s999.yaml
```
