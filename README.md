# ICL

Enhancing Mixture of Experts with Independent and Collaborative Learning for Long‑Tail Visual Recognition (IJCAI 2025)

---

## 1. Introduction
This project implements a Mixture of Experts (MoE) framework that integrates **independent** and **collaborative** learning to address long‑tail visual recognition. The approach achieves robust classification performance on benchmarks such as CIFAR10/100‑LT while emphasizing complementary and diverse experts. This repository accompanies our IJCAI 2025 submission.

---

## 2. Project Structure
```
ICL/
├─ Trainer/                  # Trainers and extensions
├─ config/                   # Configuration files
├─ datasets/                 # Dataset wrappers and long‑tail sampling
├─ loss/                     # Custom loss functions (MoE, HNM, etc.)
├─ metrics/                  # Evaluation metrics
├─ models/                   # ResNet‑MoE model definitions
├─ utils/                    # Utility functions, schedulers, etc.
└─ train_cifar.py            # Main training entry
```

---

## 3. Environment & Dependencies
- Python ≥ 3.8  
- PyTorch ≥ 1.13, Torchvision  
- NumPy, SciPy, scikit‑learn  
- Loguru, tqdm, wandb (optional for logging/visualization)  
- nni (optional for hyper‑parameter search)

Install dependencies:
```bash
pip install torch torchvision numpy scipy scikit-learn loguru tqdm wandb nni
```

---

## 4. Data Preparation
We use CIFAR10/100 and their long‑tail variants (`IMBALANCECIFAR10`, `IMBALANCECIFAR100`). On the first run, `torchvision` automatically downloads data to the paths specified in `config/config_cifar_base.py` and `config/config_cifar_moe.py`, e.g.:

```python
label_dir = 'Datasets/ltvr/cifar100'
data_dir  = 'Datasets/ltvr/cifar100'
```

Modify these fields to customize data locations.

---

## 5. Quick Start

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd ICL
   ```

2. **Optional: set random seed**
   Use the `--seed` flag in the training script or call `set_seed` in `utils/utils.py`.

3. **Example run**
   ```bash
   python train_cifar.py \
       --task ICL \
       --model ResNet_MoE \
       --dataset IMBALANCECIFAR100 \
       --seed 123 \
       --save_log True
   ```
   - `--task`: task name for logging and directory separation  
   - `--model`: model architecture (`ResNet_MoE`)  
   - `--dataset`: dataset (`IMBALANCECIFAR10` or `IMBALANCECIFAR100`)  
   - `--save_log`: whether to save training logs and model weights


---

For questions or suggestions, feel free to open an issue or pull request. Happy researching!
