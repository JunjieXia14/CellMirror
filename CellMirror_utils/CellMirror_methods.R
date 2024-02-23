# Parameters
global <- list(
  mnn_k1 = 5, # number of nearest neighbors of the first dataset (with more cells/spots) in the other dataset (with less cells/spots)
  mnn_k2 = 50, # vice versa
  mnn_ndist = 3, # ndist parameter used for MNN
)


tissue_colors <- c(`central_nervous_system`= "#f5899e",`engineered_central_nervous_system` = "#f5899e",
                   `teratoma` = "#f5899e",
                   `bone` = "#9f55bb",   
                   `pancreas` = "#b644dc", 
                   `soft_tissue` = "#5fdb69",
                   `skin` = "#6c55e2",    
                   `liver` = "#9c5e2b",
                   `blood` = "#da45bb",
                   `lymphocyte`=  "#abd23f",
                   `peripheral_nervous_system` = "#73e03d",
                   `ovary` = "#56e79d",`engineered_ovary` = "#56e79d",
                   `adrenal` = "#e13978",  `adrenal_cortex` = "#e13978",
                   `upper_aerodigestive` = "#5da134",
                   `kidney` = "#1f8fff",`engineered_kidney` = "#1f8fff",
                   `gastric` = "#dfbc3a",
                   `eye` = "#349077",
                   `nasopharynx` = "#a9e082",
                   `nerve` = "#c44c90",
                   `unknown` = "#999999",
                   `cervix` = "#5ab172",
                   `thyroid` = "#d74829",
                   `lung` = "#51d5e0",`engineered_lung` = "#51d5e0",
                   `rhabdoid` = "#d04850",
                   `germ_cell` = "#75dfbb",   `embryo` = "#75dfbb",
                   `colorectal` = "#96568e",
                   `endocrine` = "#d1d684",
                   `bile_duct` = "#c091e3",                        
                   `pineal` = "#949031",
                   `thymus` = "#659fd9",
                   `mesothelioma` = "#dc882d",
                   `prostate` = "#3870c9", `engineered_prostate` = "#3870c9",
                   `uterus` = "#e491c1",
                   `breast` = "#45a132",`engineered_breast` = "#45a132",
                   `urinary_tract` = "#e08571",
                   `esophagus` = "#6a6c2c",
                   `fibroblast` = "#d8ab6a",
                   `plasma_cell` = "#e6c241")



# run mutual nearest neighbors batch correction
run_MNN <- function(CCLE_cor, TCGA_cor,  k1 = global$mnn_k1, k2 = global$mnn_k2, ndist = global$mnn_ndist) {

  mnn_res <- modified_mnnCorrect(CCLE_cor, TCGA_cor, k1 = k1, k2 = k2, ndist = ndist)

  return(mnn_res)
}



# MNN --------------------------------------------------------------------

# Modification of the scran::fastMNN (https://github.com/MarioniLab/scran)
# Allows for separate k values per dataset, and simplifies some of the IO and doesn't use PCA reduction
modified_mnnCorrect <- function(ref_mat, targ_mat, k1 = 20, k2 = 20, ndist = 3) {

  sets <- batchelor::findMutualNN(ref_mat, 
                                  targ_mat, 
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
                                           k=k2, ndist=ndist, BPPARAM=BiocParallel::SerialParam())
  
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
.tricube_weighted_correction <- function(curdata, correction, in.mnn, k=20, ndist=3, BNPARAM=NULL, BPPARAM=BiocParallel::SerialParam())
  # Computing tricube-weighted correction vectors for individual cells,
  # using the nearest neighbouring cells _involved in MNN pairs_.
  # Modified to use FNN rather than queryKNN for nearest neighbor finding
{
  cur.uniq <- curdata[in.mnn,,drop=FALSE]
  safe.k <- min(k, nrow(cur.uniq))
  # closest <- queryKNN(query=curdata, X=cur.uniq, k=safe.k, BNPARAM=BNPARAM, BPPARAM=BPPARAM)
  closest <- FNN::get.knnx(cur.uniq, query=curdata, k=safe.k)
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

# run GSEA using the fgsea package, with either gene or label permutation
run_fGSEA <- function (gsc, X = NULL, y = NULL, perm_type = "label", nperm = 10000, 
          min_set_size = 3, max_set_size = 300, stat_type = "t_stat", 
          stat_trans = "none", gseaParam = 1, nproc = 0, gene_stat = NULL) 
{
  if (any(grepl("package:piano", search()))) {
    detach("package:piano", unload = TRUE)
  }
  library(fgsea)
  stopifnot(perm_type %in% c("label", "gene"))
  if (perm_type == "label") {
    stopifnot(is.matrix(X) & !is.null(y))
    print(sprintf("Running sample-permutation testing with %d perms", 
                  nperm))
    used_samples <- which(!is.na(y))
    used_genes <- which(apply(X[used_samples, ], 2, var, 
                              na.rm = T) > 0)
    fgseaRes <- fgsea::fgsea(pathways = GSEABase:::geneIds(gsc), mat = t(X[used_samples, 
                                                                 used_genes]), labels = y[used_samples], minSize = min_set_size, 
                              maxSize = max_set_size, nperm = nperm, gseaParam = gseaParam, 
                              nproc = nproc)
  }
  else if (perm_type == "gene") {
    print(sprintf("Running gene-permutation testing with %d perms", 
                  nperm))
    fgseaRes <- fgsea::fgsea(pathways = GSEABase:::geneIds(gsc), stats = gene_stat, 
                             minSize = min_set_size, maxSize = max_set_size, 
                             nperm = nperm, gseaParam = gseaParam, nproc = nproc)
  }
  return(fgseaRes)
}