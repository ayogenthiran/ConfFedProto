# ConFedProto: Adaptive Confidence Based Prototype Aggregation for Federated Prototype Learning on Non-IID Data

This repository implements FedProto, a novel federated learning approach that uses prototype learning to handle heterogeneous clients. The implementation supports both MNIST and CIFAR10 datasets and compares FedProto with standard Federated Learning.

## Overview

FedProto addresses heterogeneity in federated learning by exchanging prototypes instead of model parameters. This approach helps in:
- Handling non-IID data distributions
- Reducing communication costs
- Improving model performance

## Citation

If you use this code, please cite the original FedProto paper:
```bibtex
@article{tan2022fedproto,
  title={FedProto: Federated prototype learning across heterogeneous clients},
  author={Tan, Yue and Long, Guodong and Liu, Lu and Zhou, Tianyi and Lu, Qinghua and Jiang, Jing and Zhang, Chengqi},
  journal={AAAI Conference on Artificial Intelligence},
  year={2022}
}
