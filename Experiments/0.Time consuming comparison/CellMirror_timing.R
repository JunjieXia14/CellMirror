cLDVAE_timing_list <- read.csv('cLDVAE_timing.csv')

cell_num_list <- seq(1000,13000,1000)

running_time_list <- c()

for (i in seq(1, length(cell_num_list))){
  
  cell_num <- cell_num_list[i]
  
  TCGA_cor<-read.csv(paste('cell_num_',cell_num,'_TCGA_CCLE_data_tumor_X_cLDVAE_only.csv',sep = ''))
  rownames(TCGA_cor)<-TCGA_cor$sampleID
  TCGA_cor<-as.matrix(TCGA_cor[,-1])
  
  CCLE_cor<-read.csv(paste('cell_num_',cell_num,'_TCGA_CCLE_data_CL_X_cLDVAE_only.csv',sep = ''))
  rownames(CCLE_cor)<-CCLE_cor$sampleID
  CCLE_cor<-as.matrix(CCLE_cor[,-1])
  
  start<-Sys.time()
  
  mnn_res <- run_MNN(CCLE_cor, TCGA_cor, k1 = 80, k2 = 100, ndist = global$mnn_ndist,subset_genes = colnames(TCGA_cor))
  
  duration<-Sys.time()-start
  
  running_time_list[i]<-duration + as.numeric(cLDVAE_timing_list[i])

}

print(running_time_list)