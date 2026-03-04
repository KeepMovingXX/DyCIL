# Towards OOD Generalization in Dynamic Graphs via Causal Invariant Learning


This repository provides the official PyTorch implementation of:

> **Towards OOD Generalization in Dynamic Graphs via Causal Invariant Learning**  
> Xinxun Zhang, Pengfei Jiao, Mengzhou Gao, Tianpeng Li, Xuan Guo  
> AAAI Conference on Artificial Intelligence (AAAI), 2026

---

## Introduction

Out-of-distribution (OOD) generalization in dynamic graphs is a fundamental yet challenging problem due to evolving environments and shifting data distributions.  

In this work, we propose a novel framework that explicitly exploits invariant spatio-temporal patterns from a causal perspective.

Our method:

- a dynamic causal subgraph generator to extract causal structural information
- a causal-aware spatio-temporal attention module to capture intrinsic evolution rationale
- an adaptive environment generator to model underlying distribution shifts.


## ⚙️ Dependencies

- CUDA = 11.3  
- Python ≥ 3.9  
- PyTorch ≥ 1.12.0  
- PyTorch Geometric ≥ 2.3.0  
- NumPy ≥ 1.24.3  
- SciPy ≥ 1.16.0  
- scikit-learn ≥ 1.2.2  
- tqdm ≥ 4.65.0  

---

## 🚀 Quick Start

Run the following command:

```bash
python main.py --dataset dataset_name
```

### Supported Datasets

**Link Prediction**
- `collab`
- `act`
- `synthetic` (0.4, 0.6, 0.8)

**Node Classification**
- `Aminer`
- `dymotif_data`

---


## 📖 Citation

If you find this work useful in your research, please cite:

### AAAI Version

```bibtex
@inproceedings{zhang2026ood,
  title={Towards OOD Generalization in Dynamic Graphs via Causal Invariant Learning},
  author={Zhang, Xinxun and Jiao, Pengfei and Gao, Mengzhou and Li, Tianpeng and Guo, Xuan},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```

### arXiv Version

```bibtex
@misc{zhang2026oodgeneralizationdynamicgraphs,
  title={Towards OOD Generalization in Dynamic Graphs via Causal Invariant Learning},
  author={Xinxun Zhang and Pengfei Jiao and Mengzhou Gao and Tianpeng Li and Xuan Guo},
  year={2026},
  eprint={2603.01626},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2603.01626}
}
```
