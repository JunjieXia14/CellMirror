
Introduction
============

.. image:: ../CellMirror_utils/Main_figure_CellMirror.png

We propose CellMirror, an explainable contrastive learning model to dissect heterogeneous cell populations in ST data using scRNA-seq data. Specifically, CellMirror (i) employs a contrastive variational autoencoder (with nonlinear 
encoder and linear decoder) to learn salient and shared features from reference (e.g., scRNA-seq) and target (e.g., ST) data, indicating target-specific and shared variations; and (ii) leverages mutual nearest neighbor (MNN) to learn 
common features by eliminating sequencing bias on the shared features. Once converged, we (1) transferred labels from reference to target samples based on 𝐾 nearest neighbors (KNN) in the common feature space; (2) identified the genes 
related to target-specific and shared features; and (3) visualized alignment between samples using uniform manifold approximation and projection (UMAP). 

CellMirror outperforms other tools in estimating cell populations in ST data and learning interpretable factors for 
biological understanding. Particularly, in breast cancer studies, CellMirror detects finer domains in ST data missed by other competing methods, and is also robust to deciphering cell populations in ST data by independent scRNA-seq data. 
Importantly, such contrastive learning model employed by CellMirror offers a flexible framework to decipher tumor structure, by integrating spatial and single-cell epigenomics or proteomics data.