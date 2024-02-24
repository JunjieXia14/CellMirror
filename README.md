![image](https://github.com/JunjieXia14/CellMirror/blob/main/CellMirror_banner.jpg)

# CellMirror: Deciphering Cell Populations from Spatial Transcriptomics Data by Interpretable Contrastive Learning.

![image](https://github.com/JunjieXia14/CellMirror/blob/main/CellMirror_utils/Main_figure_CellMirror.png)

**Overview of CellMirror.** **a** Given the reference and target datasets as input, CellMirror adopts interpretable contrastive variational autoencoder (i.e., cLDVAE) with nonlinear encoder and linear decoder to learn salient features that are unique to the target data and shared features representing biological variations in both datas. **b** CellMirror leverages MNN to learn common features by removing the batch effects between two datasets in the shared feature space. **c** The common features can be used for label transfer and visualization, and the weights of genes for each feature can be used for biological interpretations.

# Getting started

See documentation and tutorials https://cellmirror.readthedocs.io/.

# Installation

## Install CellMirror

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

## Install R packages

Our MNN program is deployed on R software and rpy2 library, please install r-base and related package dependecies via conda.

Run the following commands in Linux Bash Shell:

```bash
conda install r-base
conda install r-dplyr (here, magrittr, tidyverse, batchelor, BiocParallel, FNN)
```

Or you can install these package dependencies by install.packages() and BiocManager::install() commands in R script.

Note: To reduce your waiting time, we recommend using the rpy2 library to call the path of R software installed in your existing virtual environment.

# Quick start

## Input

Take the global alignment of tumors and cell lines as an example, the expression data and annotation file are available at https://github.com/JunjieXia14/CellMirror/tree/main/Data

## Run

### Step 1. Run cLDVAE model

This function automatically (1) learns common features of target and reference sets by contrastive learning, (2) extracts target-specific representations, and (3) linear decoder weight for processing salient features. It takes ~5 mins for loading tumor and cell line data, and ~8 mins for contrastive learning.

```
python cLDVAE_model.py
```

In running, the useful parameters:

* max_epoch: defines the max iteration for training cLDVAE model. The default value is 1000. You can modify it. The smaller the parameter, the less time.
* lr_cLDVAE: defines learning rate parameter for learning common features of target and reference sets by cLDVAE. The default value of the parameters is 3e-6.
* beta: defines the penalty for the KL divergence. The default value is 1. You can adjust it from 0 to 1 by 0.1;
* gamma: defines the penalty for the Total Correlation loss. The default value is 0. You can further improve the results of common features by adjusting it from -100 to 100 by 10.
* batch_size: defines the batch size for training cLDVAE model. The default value is 128. You can modify it based on your memory size. The larger the parameter, the less time.
* s_latent_dim: defines the dimension of salient features of cLDVAE model. The default value of the tumor and cell line dataset is 2.
* z_latent_dim: defines the dimension of common features of cLDVAE model. The default value of the tumor and cell line dataset is 100. Given a specific dimension of salient features, a higher dimension of common features is recommended.

Note: To reduce your waiting time, we have uploaded the processed results into the folder ./Data/CellMirror_test_data/. You can directly perform step 2.

### Step 2. Run CellMirror model

This function from R file named CellMirror_model.R automatically learns batch corrected results of the common features of target and reference sets using MNN. It takes ~30 seconds.

```
Rscript CellMirror_model.R
```

In running, the useful parameters:

* k1: defines the number of nearest neighbors of target data in the reference data. The default value is 5.
* k2: defines the number of nearest neighbors of reference data in the target data. The default value is 50.
* ndist: defines the ndist parameter used for MNN. The default value is 3.
* subset_genes: defines a set of biologically relevant genes (e.g., highly variable genes) to facilitate identification of MNNs. The default subset_genes are the highly variable genes that are common in both expression data.

## Output

### Run Downstream analysis on the output file

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

# References

* Celligner: https://github.com/broadinstitute/Celligner_ms
* contrastive VAE: https://github.com/abidlabs/contrastive_vae
* LDVAE: https://github.com/YosefLab/scVI

# Citation

* Xia J, Cui J, Huang Z, Zhang S, Yao F, Zhang Y, Zuo C. CellMirror: Deciphering Cell Populations from Spatial Transcriptomics Data by Interpretable Contrastive Learning. *IEEE International Conference on Medical Artificial Intelligence (MedAI)*, 2023.
