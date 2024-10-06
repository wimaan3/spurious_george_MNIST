# Spurious Correlation in MNIST with George Pipeline

This project implements the [George](https://arxiv.org/abs/2011.12945) pipeline for addressing the **spurious correlations problem** using the [SpuCo](https://spuco.readthedocs.io/en/latest/) package. We train a deep learning model on the [SpuCoMNIST](https://spuco.readthedocs.io/en/latest/reference/datasets.html#module-spuco.datasets.spuco_mnist) dataset to improve the model's robustness against spurious correlations.

### Workflow
1. We train a Convolutional Neural Network (CNN) model (LeNet) using Empirical Risk Minimization ([ERM](https://spuco.readthedocs.io/en/latest/reference/robust_train.html#module-spuco.robust_train.erm)) on the **SpuCoMNIST** dataset.
2. After ERM training, we cluster the model’s outputs using the [Cluster](https://spuco.readthedocs.io/en/latest/reference/group_inference.html#module-spuco.group_inference.cluster) class from SpuCo's **group_inference** module.
3. We then perform **group-balanced training** using the [GroupBalanceBatchERM](https://spuco.readthedocs.io/en/latest/reference/robust_train.html#module-spuco.robust_train.group_balance_batch_erm) method to ensure equal representation of each group during training.
4. Finally, we evaluate the model's predictions on the MNIST digits and output the accuracy.

# Installation Instructions

Before running the project, ensure you have the required dependencies. Follow the steps below to set up your environment:

### 1. Install Python 3.x
Ensure that you have Python 3.x installed. You can download it from the official Python website: [Download Python](https://www.python.org/downloads/).

### 2. Install Required Libraries

Run the following command to install the necessary libraries:

```bash
pip install torch spuco tqdm pandas
