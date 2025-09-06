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
│   ├── ...
│   ├── map.pkl
│   ├── test_scenario_generation.py
├── mtr/
│   ├── datasets/
│   │   ├── social_aware/
├── output/  # Content of the output folder is shown and introduced below
├── tools/
│   ├── cfgs/
│   ├── eval_utils/
│   ├── train_utils/
│   ├── test.py
│   ├── train.py
├── readme.md
└── requirments.txt
└── setup.py
```
---


## Installation

Create a conda environment suitable for your hardware and install PyTorch. Then install the dependencies as follows:
```bash
pip install -r requirements.txt
python setup.py develop
```

---

## Dataset Preparation
After feeding the `*_raw_*.pkl` files into the model, the model will preprocess and organize them, generating corresponding cache files.
You can set the `reuse_temp_file` parameter in the configuration file to allow the model to directly read the cache files. All data files and their corresponding cache files are already provided in the `data/` folder.

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
All training scripts live in `tools/`. Specify the configuration file and other parameters to launch training via the `train.py` script.

Single-GPU example (baseline):
```bash
cd tools
python train.py   --launcher none   --cfg_file cfgs/baseline_train.yaml   --batch_size 32   --epochs 100   --extra_tag my_baseline_train
```

Evaluation results, checkpoints, and logs will be saved under `output/cofing_file_name/extra_tag`. For example, the results from the command above will be saved in `output/baseline_train/my_baseline_train`.

---

## Testing / Evaluation
Use `test.py` with an appropriate config and checkpoint.

Example:
```bash
cd tools
python test.py   --launcher none   --cfg_file cfgs/test.yaml   --ckpt ../output/social_aware_train/result_social_aware/ckpt/best_model.pth   --batch_size 32   --extra_tag my_social_test
```

---

## Result Organization & Post-Processing
Results from training or testing are written to `output/`:

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

Three folders, named the same as the configuration files in `tools/cfgs/`, store the result files and log files from the training or testing launched with the corresponding configuration file.
The path for evaluation result files is `output/cofing_file_name/extra_tag/eval/eval_with_train/epoch_x/result.pkl`, for example: `output/baseline_train/result_baseline/eval/eval_with_train/epoch_200/result.pkl`
The path for checkpoint files is `output/cofing_file_name/extra_tag/ckpt/best_model.pth`, for example: `output/baseline_train/result_baseline/ckpt/best_model.pth`

The `results_process.py` script can be used to combine the prediction results of the baseline and social-aware models on the same test set into a single file. Before using results_process.py, please modify the paths of the prediction result files that need to be combined in the code.
The three `.pkl` files in the `output/` folder are the prediction results of the two models on the brake test set, the collision test set, and the non-collision validation set, respectively.
---


*Update with volume / pages / DOI when published.*


---

## Contact

Primary contact: **Bingbing Nie** — `nbb@tsinghua.edu.cn`  
Project Maintainers: Jinghe Lin, Gaoyuan Kuang


---

*Last updated: 2025-09-06*

