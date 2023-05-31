TCGA_ann_init <- TCGA_ann
CCLE_ann_init <- CCLE_ann
TCGA_mat_init <- read.csv('TCGA_CCLE_data_tumor_X.csv')
rownames(TCGA_mat_init)<-TCGA_mat_init$sampleID
TCGA_mat_init<-as.matrix(TCGA_mat_init[,-1])
CCLE_mat_init <- read.csv('TCGA_CCLE_data_CL_X.csv')
rownames(CCLE_mat_init)<-CCLE_mat_init$sampleID
CCLE_mat_init<-as.matrix(CCLE_mat_init[,-1])

cell_num_list <- seq(1000,13000,1000)
running_time_list <- c()
for (i in seq(1, length(cell_num_list))){
  start <- as.numeric(Sys.time())
  
  cell_num <- cell_num_list[i]
  if (cell_num==1000){
    TCGA_mat <- TCGA_mat_init[1:cell_num,]
    CCLE_mat <- CCLE_mat_init[1:cell_num,]
    
    TCGA_ann <- TCGA_ann_init[1:cell_num,]
    CCLE_ann <- CCLE_ann_init[1:cell_num,]
  }
  else if (cell_num==13000){
    TCGA_mat <- TCGA_mat_init
    CCLE_mat <- CCLE_mat_init
    
    TCGA_ann <- TCGA_ann_init
    CCLE_ann <- CCLE_ann_init
  }
  else{
    TCGA_mat <- TCGA_mat_init[1:cell_num,]
    CCLE_mat <- CCLE_mat_init
    
    TCGA_ann <- TCGA_ann_init[1:cell_num,]
    CCLE_ann <- CCLE_ann_init
  }
  
  TCGA_obj <- create_Seurat_object(TCGA_mat, TCGA_ann, type='tumor')
  CCLE_obj <- create_Seurat_object(CCLE_mat, CCLE_ann, type='CL')
  
  TCGA_obj <- cluster_data(TCGA_obj)
  CCLE_obj <- cluster_data(CCLE_obj)
  cov_diff_eig <- run_cPCA(TCGA_obj, CCLE_obj)
  cur_vecs <- cov_diff_eig$vectors[, seq(1,2), drop = FALSE]
  #*#
  TCGA_cor <- resid(lm(t(TCGA_mat) ~ 0 + cur_vecs)) %>% t()
  CCLE_cor <- resid(lm(t(CCLE_mat) ~ 0 + cur_vecs)) %>% t()
  #*#
  mnn_res <- run_MNN(CCLE_cor, TCGA_cor, k1 = 80, k2 = 100, ndist = global$mnn_ndist,subset_genes = colnames(TCGA_cor))
  
  duration <- as.numeric(Sys.time())-start
  print(duration)
  running_time_list[i]<-duration
}
print(running_time_list)