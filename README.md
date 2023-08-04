![image](https://github.com/JunjieXia14/CellMirror/blob/main/CellMirror_banner.jpg)

# CellMirror: Dissecting cell populations across heterogeneous transcriptome data using interpretable contrastive learning.

![image](https://github.com/JunjieXia14/CellMirror/blob/main/CellMirror_utils/Main_figure_CellMirror.jpg)

**Overview of CellMirror.** **a** Given two different transcriptome data as target and reference data, CellMirror adopts contrastive variational autoencoder (with non-linear encoder and linear decoder) to learn salient features that are unique to the target dataset and common features that are shared by both datasets (Methods). **b** CellMirror uses the common features disentangled from cLDVAE as input for MNN to eliminate batch effects (Methods), and ultimately aligns target and reference datasets. **c** The learned weights from the linear decoder in cLDVAE can be used to identify the genes or gene programs related to features (Methods), with a higher value indicating enrichment of the corresponding gene in a latent feature. The dataset aligned by CellMirror can be used for visualization and label transfer: (i) annotating each cell line or tumor sample with the most frequent cancer type among its nearest neighbors, when applied to integrate cancer samples and cell lines; (ii) predicting cell type probabilities for each spot based on the cell type proportions of its nearest neighbors, when integrating scRNA-seq and ST data.

# Installation

## Install CellMirror

Installation was tested on Red Hat 7.6 with Python 3.8.5 and torch 1.4.0 on a machine with one 40-core Intel(R) Xeon(R) Gold 5115 CPU addressing with 132GB RAM, and two NVIDIA TITAN V GPU addressing 24GB. CellMirror is implemented in the Pytorch framework. Please run CellMirro on CUDA if possible.

### 1. Grab source code of CellMirror

```
git clone https://github.com/JunjieXia14/CellMirror.git

cd CellMirror
```

### 2. Install CellMirror in the virtual environment by conda

* Firstly, install conda: https://docs.anaconda.com/anaconda/install/index.html
* Then, automatically install all used packages (described by "used_package.txt") for CellMirror in a few mins.

```
conda create -n CellMirror python=3.8.5 pip

source activate

conda activate CellMirror

pip install -r used_package.txt
```

## Install R packages

* Install tested on R = 4.0.0
* install.packages(c("Seurat", "ggplot2", "patchwork", "stringr", "magrittr", "here", "tidyverse"))

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
