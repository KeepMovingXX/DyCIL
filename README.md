### This repository is a PyTorch implementation of "Causal Invariant Learning for Out-of-Distribution Generalization in Dynamic Graphs"
### Dependencies
* cuda = 11.3
* python >= 3.9
* pytorch >= 1.12.0
* numpy >= 1.24.3
* pyg >= 2.3.0
* scikit-learn >= 1.2.2
* scipy >= 1.16.0
* tqdm >= 4.65.0
### Quick Start
* python main.py --dataset 'dataset'
* dataset for link prediction: collab, act, synthetic (0.4, 0.6, 0.8)
* dataset for node classification: Aminer, dymotif_data
