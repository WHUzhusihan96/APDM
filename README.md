## APDM

Source code of [paper](https://www.sciencedirect.com/science/article/abs/pii/S0893608024001655) `Adversarial pair-wise distribution matching for remote sensing image cross-scene classification`.

## Requirement

* python 3
* pytorch 1.10 or above

## Datasets

Four scene classification datasets are used in our experiments: `AID, CLRS, MLRSN, and OPTIMAL-31`, which can be found in `data_dir`. 

For more information on the scene classification (or more tasks) dataset, check out this [paper](https://ieeexplore.ieee.org/document/9393553).

If you want to use your own dataset, please organize your data in the following structure and add related info to `./apdm/utils/config`.

```
RootDir
└───Domain1Name
│   └───Class1Name
│       │   file1.jpg
│       │   file2.jpg
│       │   ...
│   ...
└───Domain2Name
|   ...    
```

## Usage

1. Modify the train or test file in the `./apdm/train_test_file`, such as data and training-related parameters

2. run `python base_main.py --config "./apdm/train_test_file/base-train-config.yaml"`

## Reference

```
@article{ZHU2024106241,
title = {Adversarial pair-wise distribution matching for remote sensing image cross-scene classification},
journal = {Neural Networks},
volume = {174},
pages = {106241},
year = {2024},
issn = {0893-6080},
doi = {https://doi.org/10.1016/j.neunet.2024.106241},
url = {https://www.sciencedirect.com/science/article/pii/S0893608024001655},
author = {Sihan Zhu and Chen Wu and Bo Du and Liangpei Zhang}
}
```
