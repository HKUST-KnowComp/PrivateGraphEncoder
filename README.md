# PrivateGraphEncoder
Source Code for CIKM2023 paper "Independent Distribution Regularization for Private Graph Embedding"

## Introduction
We propose an independent regularization for private graph representation learning to protect graph embeddings against attribute inference attacks. The framework learns two graph autoencoders and forces the independence constraint.

![PVGAE](figure/framework.png)

## Reproduction

### Package Dependencies

* numpy
* pandas
* scipy
* scikit-learn == 1.1.2
* torch >= 1.12
* dgl == 0.9.0

### Train PVGAE

PVGAE sample usage at Credit defaulter dataset:

```bash
python main.py --model VGAEPrivacy --dataset credit --epoch 500 --local_epoch 1 --lr 0.005 --hidden1 64 --hidden2 32 --beta 1 --seed 1234 --use_pretrain False
```

APGE sample usage at Credit defaulter dataset:

```bash
python main_APGE.py --model APGE --dataset credit --lr 0.001 --epoch 500 --hidden1 64 --hidden2 32 --sead 1234 --use_pretrain False
``` 

### Citations
The details of this pipeline are described in the following paper. If you use this code in your work, please kindly cite it.

```bibtex

```

### Miscellaneous

Please send any questions about the code and/or the algorithm to <qhuaf@connect.ust.hk>.
