library(here)
library(magrittr)
library(tidyverse)
source(here::here('CellMirror_utils','CellMirror_methods.R'))

TCGA_cor <- read.csv('TCGA_CCLE_data_tumor_X_cLDVAE_only.csv')
rownames(TCGA_cor) <- TCGA_cor$X
TCGA_cor <- as.matrix(TCGA_cor[,-1])

CCLE_cor <- read.csv('TCGA_CCLE_data_CL_X_cLDVAE_only.csv')
rownames(CCLE_cor) <- CCLE_cor$X
CCLE_cor<-as.matrix(CCLE_cor[,-1])

mnn_res <- run_MNN(CCLE_cor, TCGA_cor, k1 = 80, k2 = 100, ndist = global$mnn_ndist,subset_genes = colnames(TCGA_cor))

write.csv(mnn_res$corrected,'TCGA_CCLE_data_tumor_X_CellMirror.csv')