Attention-based Deep Multiple Instance Learning with Temporal Ensembling
================================================

by Luciano de la Iglesia and Wilson Tang

Overview
--------

This repository applies the temporal ensembling approach from Laine and Aila (2017) to the MIL models and datasets from Ilse et al. (2018). This work was a final project for CSE 599C "Data Management for ML" in 2021. Read `report.pdf` for a description of the methods and results. 

Installation
------------

Run `pip install -r requirements.txt`. Works on Python 3 with or without GPU.

How to Use
----------
`dataloader.py`: Generates training and test set by combining multiple MNIST images to bags. A bag is given a positive label if it contains one or more images with the label specified by the variable target_number.

`mnist_bags_loader.py`: The original data loader used in Ilse et al. (2018). It can handle any bag length without the dataset becoming unbalanced. It is probably not the most efficient way to create the bags. Furthermore it is only test for the case that the target number is ‘9’.

`breast_cancer_dataloader.py`: The data loader used for the breast cancer dataset.

`main.py`: Trains the model on the MNIST-bags dataset. In order to run experiments on the breast cancer dataset, [download it](http://bioimage.ucsb.edu/research/bio-segmentation) and use the `--cancer` flag when training.

`model.py`: Defines the convolutional neural networks.

References
--------------------

```
@article{laine2017temporal,
      title={Temporal Ensembling for Semi-Supervised Learning}, 
      author={Samuli Laine and Timo Aila},
      year={2017},
      eprint={1610.02242},
      archivePrefix={arXiv},
      primaryClass={cs.NE}
}
@article{ITW:2018,
  title={Attention-based Deep Multiple Instance Learning},
  author={Ilse, Maximilian and Tomczak, Jakub M and Welling, Max},
  journal={arXiv preprint arXiv:1802.04712},
  year={2018}
}
```
