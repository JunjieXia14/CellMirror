![image](https://github.com/JunjieXia14/CellMirror/blob/main/CellMirror_banner.jpg)

# CellMirror: Deciphering Cell Populations from Spatial Transcriptomics Data by Interpretable Contrastive Learning.

## Overview

![image](https://github.com/JunjieXia14/CellMirror/blob/main/CellMirror_utils/Main_figure_CellMirror.png)

**Overview of CellMirror.** **a** Given the reference and target datasets as input, CellMirror adopts interpretable contrastive variational autoencoder (i.e., cLDVAE) with nonlinear encoder and linear decoder to learn salient features that are unique to the target data and shared features representing biological variations in both datas. **b** CellMirror leverages MNN to learn common features by removing the batch effects between two datasets in the shared feature space. **c** The common features can be used for label transfer and visualization, and the weights of genes for each feature can be used for biological interpretations.

## Getting started

See documentation and tutorials https://cellmirror.readthedocs.io/.

## Installation

Installation was tested on Red Hat 7.6 with Python 3.8.5 and torch 1.4.0 on a machine with one 40-core Intel(R) Xeon(R) Gold 5115 CPU addressing with 132GB RAM, and two NVIDIA TITAN V GPU addressing 24GB. CellMirror is implemented in the Pytorch framework. Please run CellMirror on CUDA if possible.

### 1. Grab source code of CellMirror

```bash
git clone https://github.com/JunjieXia14/CellMirror.git
cd CellMirror
```

### 2. Install CellMirror in the virtual environment by conda

* Firstly, install conda: https://docs.anaconda.com/anaconda/install/index.html
* Then, automatically install all used packages (described by "requirements.yml") for CellMirror in a few mins.

```bash
conda config --set channel_priority strict
conda env create -f requirements.yml
conda activate CellMirror
```

Other software dependencies are listed in "used_package.txt".

### 3. Install R packages

Our MNN program is deployed on R software and rpy2 library, please install r-base and related package dependecies via conda.

Run the following commands in Linux Bash Shell:

```bash
conda install r-base
conda install r-dplyr (here, magrittr, tidyverse, batchelor, BiocParallel, FNN)
```

Or you can install these package dependencies by install.packages() and BiocManager::install() commands in R script.

Note: To reduce your waiting time, we recommend using the rpy2 library to call the path of R software installed in your existing virtual environment.

## Quick start

### Alignment of TCGA and CCLE datasets

#### Input

We take the global alignment of tumors and cell lines as an example input, which includes two types of files: (1) gene expression data and (2) annotation file. These input files are available at `Datasets/README/md` .

#### Run

Run the following commands in Linux Bash Shell:

```bash
cd Tutorials/code
python TCGA_CCLE_CellMirror.py
```

This script automatically (1) loads the input data as `AnnData` object, (2) learns shared features of target and reference datasets by contrastive learning, (3) extracts target-specific representations, (4) eliminates the batch effects between two datasets in the shared feature space for common features, and (5) saves linear decoder weight for processing latent (shared and salient) features. It takes ~5 mins for loading tumor and cell line data, ~8 mins for contrastive learning, and ~30 seconds for batch-correcting.

**Hyperparameters**

* max_epoch: defines the max iteration for training cLDVAE model. The default value is 1000. You can modify it. The smaller the parameter, the less time.
* lr_cLDVAE: defines learning rate parameter for learning shared features of target and reference sets by cLDVAE. The default value of the parameters is 3e-6.
* beta: defines the penalty for the KL divergence. The default value is 1. You can adjust it from 0 to 1 by 0.1;
* gamma: defines the penalty for the Total Correlation loss. The default value is 0. You can further improve the results of shared features by adjusting it from -100 to 100 by 10.
* batch_size: defines the batch size for training cLDVAE model. The default value is 128. You can modify it based on your memory size. The larger the parameter, the less time.
* n_latent_s: defines the dimension of salient features of cLDVAE model. The default value of the tumor and cell line dataset is 2.
* n_latent_z: defines the dimension of shared features of cLDVAE model. The default value of the tumor and cell line dataset is 100. Given a specific dimension of salient features, a higher dimension of shared features is recommended.
* k1: defines the number of nearest neighbors of target data in the reference data. The default value is 5.
* k2: defines the number of nearest neighbors of reference data in the target data. The default value is 50.
* ndist: defines the ndist parameter used for MNN. The default value is 3.

#### Output

#### Run Downstream analysis on the output file

* TCGA_CCLE_data_tumor_X_cLDVAE_only.csv
* TCGA_CCLE_data_CL_X_cLDVAE_only.csv
* TCGA_CCLE_data_tumor_salient_features.csv
* TCGA_CCLE_data_salient_loadings_matrix.csv
* TCGA_CCLE_data_tumor_X_CellMirror.csv
* TCGA_CCLE_data_CL_X_CellMirror.csv

```
python Downstream_analysis.py
```

This function provides 3 downstream analyses as follows:

1. Label transfer: uses the aligned results (i.e., TCGA_CCLE_data_tumor_X_cLDVAE_only.csv & TCGA_CCLE_data_CL_X_cLDVAE_only.csv or TCGA_CCLE_data_tumor_X_CellMirror.csv & TCGA_CCLE_data_CL_X_CellMirror.csv) as input to predict the cell type of each cell in the target set or reference set.
2. Interpretability: uses the target-specific representations and linear decoder weight for processing salient features (i.e., CGA_CCLE_data_tumor_salient_features.csv & TCGA_CCLE_data_salient_loadings_matrix.csv) as input to find the highly expressed genes in salient features.
3. Visualization: maps the aligned results (i.e., TCGA_CCLE_data_tumor_X_cLDVAE_only.csv & TCGA_CCLE_data_CL_X_cLDVAE_only.csv or TCGA_CCLE_data_tumor_X_CellMirror.csv & TCGA_CCLE_data_CL_X_CellMirror.csv) into 2D-UMAP spaces for visualization.

## More tutorials

More detailed tutorials and further visualization are introduced in the `Tutorials/notebook` folder to demonstrate how to implement CellMirror: (To be updated)

## References

* Celligner: https://github.com/broadinstitute/Celligner_ms
* contrastive VAE: https://github.com/abidlabs/contrastive_vae
* LDVAE: https://github.com/YosefLab/scVI

## Citation

* Xia J, Cui J, Huang Z, Zhang S, Yao F, Zhang Y, Zuo C. CellMirror: Deciphering Cell Populations from Spatial Transcriptomics Data by Interpretable Contrastive Learning. *2023* *IEEE International Conference on Medical Artificial Intelligence (MedAI)*, Beijing, China, 2023, pp. 165-176, doi: 10.1109/MedAI59581.2023.00029.
