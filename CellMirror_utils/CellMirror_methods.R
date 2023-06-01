# Parameters
global <- list(
  n_genes = 'all', # set to 'all' to use all protein coding genes found in both datasets 
  umap_n_neighbors = 10, # num nearest neighbors used to create UMAP plot
  umap_min_dist = 0.5, # min distance used to create UMAP plot
  mnn_k_CL = 5, # number of nearest neighbors of tumors in the cell line data
  mnn_k_tumor = 50, # number of nearest neighbors of cell lines in the tumor data
  top_DE_genes_per = 1000, # differentially expressed genes with a rank better than this is in the cell line or tumor data
  # are used to identify mutual nearest neighbors in the MNN alignment step
  remove_cPCA_dims = c(1,2,3,4), # which cPCA dimensions to regress out of the data 
  distance_metric = 'euclidean', # distance metric used for the UMAP projection
  mod_clust_res = 5, # resolution parameter used for clustering the data
  mnn_ndist = 3, # ndist parameter used for MNN
  n_PC_dims = 70, # number of PCs to use for dimensionality reduction
  reduction.use = 'umap', # 2D projection used for plotting
  fast_cPCA = 10 # to run fast cPCA (approximate the cPCA eigenvectors instead of calculating all) set this to a value >= 4
)



# run mutual nearest neighbors batch correction
run_MNN <- function(CCLE_cor, TCGA_cor,  k1 = global$mnn_k_tumor, k2 = global$mnn_k_CL, ndist = global$mnn_ndist, 
                    subset_genes) {
  mnn_res <- modified_mnnCorrect(CCLE_cor, TCGA_cor, k1 = k1, k2 = k2, ndist = ndist, 
                                 subset_genes = subset_genes)
  
  return(mnn_res)
}



# MNN --------------------------------------------------------------------

# Modification of the scran::fastMNN (https://github.com/MarioniLab/scran)
# Allows for separate k values per dataset, and simplifies some of the IO and doesn't use PCA reduction
modified_mnnCorrect <- function(ref_mat, targ_mat, k1 = 20, k2 = 20, 
                                ndist = 3, subset_genes = NULL) {
  if (is.null(subset_genes)) {
    subset_genes <- colnames(ref_mat)
  }
  
  sets <- batchelor::findMutualNN(ref_mat[, subset_genes], 
                                  targ_mat[, subset_genes], 
                                  k1 = k2, k2 = k1, 
                                  BPPARAM = BiocParallel::SerialParam())
  

  mnn_pairs <- as.data.frame(sets) %>% 
    dplyr::mutate(ref_ID = rownames(ref_mat)[first],
                  targ_ID = rownames(targ_mat)[second],
                  pair = seq(nrow(.))) %>% 
    dplyr::select(-first, -second)
  
  # Estimate the overall batch vector.
  ave.out <- .average_correction(ref_mat, sets$first, targ_mat, sets$second)
  overall.batch <- colMeans(ave.out$averaged)
  
  #remove variation along the overall batch vector
  ref_mat <- .center_along_batch_vector(ref_mat, overall.batch)
  targ_mat <- .center_along_batch_vector(targ_mat, overall.batch)
  
  # Recompute correction vectors and apply them.
  re.ave.out <- .average_correction(ref_mat, sets$first, targ_mat, sets$second)

  targ_mat <- .tricube_weighted_correction(targ_mat, re.ave.out$averaged, re.ave.out$second, 
                                           k=k2, ndist=ndist, subset_genes, BPPARAM=BiocParallel::SerialParam())
  
  final <- list(corrected = targ_mat, 
                pairs = mnn_pairs)
  return(final)
}

# Copied from dev version of scran (2018-10-28) with slight modifications as noted
#https://github.com/MarioniLab/scran
.average_correction <- function(refdata, mnn1, curdata, mnn2)
  # Computes correction vectors for each MNN pair, and then
  # averages them for each MNN-involved cell in the second batch.
{
  corvec <- refdata[mnn1,,drop=FALSE] - curdata[mnn2,,drop=FALSE]
  
  corvec <- rowsum(corvec, mnn2)
  
  npairs <- table(mnn2)
  
  stopifnot(identical(names(npairs), rownames(corvec)))
  
  corvec <- unname(corvec)/as.vector(npairs)
  
  list(averaged=corvec, second=as.integer(names(npairs)))
}


.center_along_batch_vector <- function(mat, batch.vec) 
  # Projecting along the batch vector, and shifting all cells to the center _within_ each batch.
  # This removes any variation along the overall batch vector within each matrix.
{
  batch.vec <- batch.vec/sqrt(sum(batch.vec^2))
  
  batch.loc <- as.vector(mat %*% batch.vec)
  
  central.loc <- mean(batch.loc)
  
  mat <- mat + outer(central.loc - batch.loc, batch.vec, FUN="*")
  
  return(mat)
}

#' @importFrom BiocNeighbors queryKNN
#' @importFrom BiocParallel SerialParam
.tricube_weighted_correction <- function(curdata, correction, in.mnn, k=20, ndist=3, subset_genes, BNPARAM=NULL, BPPARAM=BiocParallel::SerialParam())
  # Computing tricube-weighted correction vectors for individual cells,
  # using the nearest neighbouring cells _involved in MNN pairs_.
  # Modified to use FNN rather than queryKNN for nearest neighbor finding
{
  cur.uniq <- curdata[in.mnn,,drop=FALSE]
  safe.k <- min(k, nrow(cur.uniq))
  # closest <- queryKNN(query=curdata, X=cur.uniq, k=safe.k, BNPARAM=BNPARAM, BPPARAM=BPPARAM)
  closest <- FNN::get.knnx(cur.uniq[, subset_genes], query=curdata[, subset_genes], k=safe.k)
  # weighted.correction <- .compute_tricube_average(correction, closest$index, closest$distance, ndist=ndist)
  weighted.correction <- .compute_tricube_average(correction, closest$nn.index, closest$nn.dist, ndist=ndist)
  curdata + weighted.correction
}

.compute_tricube_average <- function(vals, indices, distances, bandwidth=NULL, ndist=3) 
  # Centralized function to compute tricube averages.
  # Bandwidth is set at 'ndist' times the median distance, if not specified.
{
  if (is.null(bandwidth)) {
    middle <- ceiling(ncol(indices)/2L)
    mid.dist <- distances[,middle]
    bandwidth <- mid.dist * ndist
  }
  bandwidth <- pmax(1e-8, bandwidth)
  
  rel.dist <- distances/bandwidth
  rel.dist[rel.dist > 1] <- 1 # don't use pmin(), as this destroys dimensions.
  tricube <- (1 - rel.dist^3)^3
  weight <- tricube/rowSums(tricube)
  
  output <- 0
  for (kdx in seq_len(ncol(indices))) {
    output <- output + vals[indices[,kdx],,drop=FALSE] * weight[,kdx]
  }
  
  if (is.null(dim(output))) {
    matrix(0, nrow(vals), ncol(vals))
  } else {
    output
  }
}