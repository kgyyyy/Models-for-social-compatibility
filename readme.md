# Models for Social Compatibility

**Code & Data for Training / Evaluating Social-Aware and Baseline Trajectory Prediction Models**

> This repository provides a *social-aware* model and a *baseline* model implemented on top of the **MTR (Motion TRansformer, https://github.com/sshaoshuai/MTR)** framework. Both inherit the original MTR architecture and training utilities but are trained on *different curated datasets* emphasizing safety-critical, interaction-rich scenarios (e.g., braking and crash-risk situations). Installation, configuration, and launch procedures remain consistent with upstream MTR.

---

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Repository Structure](#repository-structure)
3. [Installation](#installation)  
4. [Dataset Preparation](#dataset-preparation)  
5. [Configuration](#configuration)  
6. [Training](#training)  
7. [Testing / Evaluation](#testing--evaluation)  
8. [Result Organization & Post-Processing](#result-organization--post-processing)   
9. [Contact](#contact)
---

## Project Overview
Accurate multi-agent motion forecasting under *socially- and safety-critical* scenarios (e.g., abrupt braking, potential collision chains) is crucial for downstream socially compatible autonomous driving policy learning.  
This repository extends **MTR** with:
- Curated scenario subsets (baseline vs. social-aware)
- Scripts for synthetic *brake test* scenario generation (540 cases)
- Unified preprocessing for raw → model-ready pickle datasets
- Consistent training / evaluation harness (mirroring upstream MTR) for comparability

---

## Repository Structure

```
~/ 
├── data/
│   ├── baseline_raw_train.pkl
│   ├── baseline_train.pkl
│   ├── brake_test_data_raw.pkl
│   ├── brake_test_data.pkl
│   ├── crash_data_raw_test.pkl
│   ├── crash_data_test.pkl
│   ├── data_raw_train.pkl
│   ├── data_raw_val.pkl
│   ├── data_train.pkl
│   ├── data_val.pkl
│   ├── map.pkl
│   ├── test_scenario_generation.py
├── mtr/
│   ├── datasets/
│   │   ├── social_aware/
│   │   │   ├── map/
│   │   │   ├── data_preprocess.py
│   │   │   ├── eval_forecasting.py
│   │   │   ├── social_dataset.py
│   │   │   ├── utils.py
│   │   ├── __init__.py
│   │   ├── dataset.py
├── output/
├── tools/
├── readme.md
└── setup.py
```

Use `data_preprocess.py` to transform each `*_raw_*.pkl` file into model-compatible inputs. The `data/` folder already contains preprocessed data for immediate use.

---


## Installation


```bash
pip install -r requirements.txt
python setup.py develop
```

---

## Dataset Preparation
All raw and processed datasets reside in `data/`. If you need to reprocess:
```bash
python mtr/datasets/social_aware/data_preprocess.py   --input data/baseline_raw_train.pkl   --output data/baseline_train.pkl
```

Brake test scenarios (540) are generated via:
```bash
python data/test_scenario_generation.py
```

---

## Configuration
YAML configuration files under `tools/cfgs/` control:
- Dataset paths
- Model hyperparameters
- Training schedules
- Logging & checkpoint frequency

Edit (e.g.) `baseline_train.yaml` or `social_aware_train.yaml` prior to launching.

---

## Training
All training scripts live in `tools/`.

Single-GPU example (baseline):
```bash
cd tools
python train.py   --launcher none   --cfg_file cfgs/social_aware/baseline_train.yaml   --batch_size 32   --epochs 100   --extra_tag my_baseline_train
```

Social-aware model:
```bash
python train.py   --launcher none   --cfg_file cfgs/social_aware/social_aware_train.yaml   --batch_size 32   --epochs 100   --extra_tag my_social_train
```

---

## Testing / Evaluation
Use `test.py` with an appropriate config and checkpoint.

Example:
```bash
cd tools
python test.py   --launcher none   --cfg_file cfgs/social_aware/test.yaml   --ckpt ../output/social_aware_train/result_social_aware/ckpt/best_model.pth   --batch_size 32   --extra_tag my_social_test
```

---

## Result Organization & Post-Processing
Results are written to `output/`:

```
output/
├── baseline_train/
│   ├── result_baseline/
├── social_aware_train/
│   ├── result_social_aware/
├── test/
│   ├── baseline_test_brake/
│   ├── baseline_test_crash_scenario/
│   ├── social_aware_test_brake/
│   ├── social_aware_test_crash_scenario/
├── brake_test_results.pkl
├── crash_scenario_results.pkl
├── eval_results.pkl
├── results_process.py
```

`results_process.py` consolidates scenario-level predictions into unified dictionaries (model outputs + metadata).

---


*Update with volume / pages / DOI when published.*


---

## Contact

Primary contact: **Bingbing Nie** — `nbb@tsinghua.edu.cn`  
Project Maintainers: Jinghe Lin, Gaoyuan Kuang


---

*Last updated: 2025-07-27*

