# CellMirror: Dissecting cell populations across heterogeneous transcriptome data using interpretable contrastive learning.

![image](https://github.com/JunjieXia14/CellMirror/blob/main/CellMirror_utils/Main_figure_CellMirror.jpg)

**Overview of CellMirror. a** Given two different transcriptome data as target and reference sets, CellMirror adopts cLDVAE to learn salient features that are unique to the target set and common features that are shared by both sets using contrastive learning. **b** CellMirror uses the common components disentangled from cLDVAE as the input of MNN for eliminating batch effects and ultimately aligns target and reference sets. **c** The learned weights from linear decoder in cLDVAE are used to establish relations between cell (or spot) representation and gene expression, with a higher value indicating enrichment of corresponding gene in the latent feature. The aligned set by CellMirror can be used for visualization and label transfer: (i) annotating each cell line or bulk tumor sample with the most frequent cancer type by its nearest neighbors when applied to combine tumors and cell lines; (ii) predicting the cell type probabilities for each spot with the cell type proportions of its nearest neighbors when applied to integrate scRNA-seq and ST data. **d** Running time comparison of Celligner and CellMirror as we increased the number of cells.

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

* max_epoch:
* lr_cLDVAE:
* beta:
* gamma:
* batch_size:
* s_latent_dim:
* z_latent_dim:

### Step 2. Run CellMirror model

This function from R file named CellMirror_model.R automatically learns batch corrected results of the common features of target and reference sets using MNN. It takes ~30 seconds.

In running, the useful parameters:

* k1:
* k2:
* ndist:
* subset_genes:

## Output

### Run Downstream analysis on the output file

```
python Downstream_analysis.py
```

This function provides 3 downstream analyses as follows:

* Label transfer:
* Interpretability:
* Visualization:

# References

* Celligner: https://github.com/broadinstitute/Celligner_ms
* contrastive VAE: https://github.com/abidlabs/contrastive_vae
* LDVAE: https://github.com/YosefLab/scVI

# Citation
