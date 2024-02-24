import pandas as pd
import numpy as np
import scanpy as sc
import re
import argparse

def parameter_setting():

    parser      = argparse.ArgumentParser(description='contrastive Linear Decoded VAE')

    parser.add_argument('--weight_decay', type=float, default = 1e-6, help='weight decay')
    parser.add_argument('--eps', type=float, default = 1e-7, help='eps')

    parser.add_argument('--max_epoch', '-me', type=int, default = 1000, help='Max epoches for target and background data')
    parser.add_argument('--epoch_per_test', '-ept', type=int, default = 10, help='how many epoches to compute silhouette score')

    parser.add_argument('--lr_cLDVAE', type=float, default = 5e-5, help='Learning rate of cLDVAE model for target and background data')
    parser.add_argument('--beta', type=float, default = 0.1, help='coefficient of KL_loss')
    parser.add_argument('--gamma', type=float, default = 100, help='coefficient of TC_loss')
    parser.add_argument('--batch_size', '-bs', type=int, default = 128, help='batch size of target and background data for each epoch ')
    parser.add_argument('--last_batch_size', '-lbs', type=int, default = 0, help='batch size for the last batch')

    parser.add_argument('--n_hidden_en', type=list, default = [400], help='structure of contrasitve variational encoder')
    parser.add_argument('--n_hidden_de', type=list, default = [200], help='structure of linear decoder')

    parser.add_argument('--n_latent_s', type=int, default = 6, help='dimensionality of salient latent features')
    parser.add_argument('--n_latent_z', type=int, default = 50, help='dimensionality of shared latent features')

    parser.add_argument('--use_cuda', dest='use_cuda', default=True, action='store_true', help=" whether use cuda(default: True)")
    parser.add_argument('--bias', type=bool, default = False, help='Whether use bias in linear decoder')
    parser.add_argument('--seed', type=int, default = 991013, help='random seed for reproduction')

    return parser

def set_last_batchsize(args, tar_obj, bac_obj):

    n = bac_obj.n_obs if bac_obj.n_obs>tar_obj.n_obs else tar_obj.n_obs

    args.last_batch_size = n - int(n/args.batch_size)*args.batch_size

    return args

def pseudo_data_padding(tar_obj, bac_obj):

    if bac_obj.n_obs > tar_obj.n_obs:

        pseudo_bac = bac_obj.X.toarray()
        pseudo_bac = pseudo_bac.astype('float32')

        pseudo_tar = np.concatenate( (tar_obj.X.todense(), np.random.multivariate_normal(np.mean(pd.DataFrame(tar_obj.X.todense()), axis=0), np.cov(pd.DataFrame(tar_obj.X.todense()).T), bac_obj.n_obs-tar_obj.n_obs)), axis=0 )
        pseudo_tar = pseudo_tar.astype('float32')

    else:

        pseudo_tar = tar_obj.X.toarray()
        pseudo_tar = pseudo_tar.astype('float32')

        pseudo_bac = np.concatenate( (bac_obj.X.todense(), np.random.multivariate_normal(np.mean(pd.DataFrame(bac_obj.X.todense()), axis=0), np.cov(pd.DataFrame(bac_obj.X.todense()).T), tar_obj.n_obs-bac_obj.n_obs)), axis=0 )
        pseudo_bac = pseudo_bac.astype('float32')

    return pseudo_tar, pseudo_bac

def pseudo_data_deparser(tar_obj, pseudo_tar_z, bac_obj, pseudo_bac_z):

    if bac_obj.n_obs > tar_obj.n_obs:

        tar_z = pseudo_tar_z[:tar_obj.n_obs, :]

        bac_z = pseudo_bac_z
    
    else:

        bac_z = pseudo_bac_z[:bac_obj.n_obs, :]

        tar_z = pseudo_tar_z

    return tar_z, bac_z

def mnn_correct(tar_z, bac_z):

    import rpy2.robjects as robjects

    robjects.r.library("here")
    robjects.r.library("magrittr")
    robjects.r.library("tidyverse")

    rsource = robjects.r['source']
    rhere = robjects.r['here']
    rsource(rhere('CellMirror_utils', 'CellMirror_methods.R'))

    import rpy2.robjects.numpy2ri, rpy2.robjects.pandas2ri
    rpy2.robjects.numpy2ri.activate()
    rpy2.robjects.pandas2ri.activate()

    robjects.globalenv['tar_mat'] = pd.DataFrame(tar_z)
    robjects.globalenv['tar_mat'] = robjects.r['as.matrix'](robjects.globalenv['tar_mat'])
    robjects.globalenv['bac_mat'] = pd.DataFrame(bac_z)
    robjects.globalenv['bac_mat'] = robjects.r['as.matrix'](robjects.globalenv['bac_mat'])

    if bac_z.shape[0] > tar_z.shape[0]:

        k1 = 5

        k2 = (int(bac_z.shape[0] / tar_z.shape[0]) + 1) * k1

    else:

        k2 = 5

        k1 = (int(tar_z.shape[0] / bac_z.shape[0]) + 1) * k2

    rmnn = robjects.r['modified_mnnCorrect']
    robjects.globalenv['mnn_res'] = rmnn(robjects.globalenv['bac_mat'], robjects.globalenv['tar_mat'], 
                                         k1, k2
                                        )

    tar_corr = np.array( (robjects.globalenv['mnn_res']).rx2('corrected') )

    return tar_corr, bac_z

def estimate_cell_type(tar_obj, bac_obj, neighbors=20, used_obsm='CellMirror', used_label='Cell_subtype'):

    dist_mat = pd.DataFrame( np.corrcoef(bac_obj.obsm[used_obsm], tar_obj.obsm[used_obsm])[:bac_obj.n_obs, -tar_obj.n_obs:], index=bac_obj.obs.index, columns=tar_obj.obs.index )

    class_prop = [ 
                  ( bac_obj.obs.loc[ dist_mat[tar].sort_values(ascending=False).index[:neighbors] ][used_label].value_counts( ascending=False) / neighbors ).to_dict()
                  for tar in dist_mat.columns 
                 ]
    
    tar_obj.obs['class_prop'] = class_prop

    celltypes = np.unique(bac_obj.obs[used_label].values).tolist()
    tar_obj.obs[celltypes] = 0

    for type in celltypes:
        type_proportion = []
        for tar in tar_obj.obs.index:
            celltype_dict = tar_obj.obs.loc[tar]['class_prop']
            type_proportion.append( float(celltype_dict[type]) ) if (type in celltype_dict.keys()) else type_proportion.append( float(0) )
        tar_obj.obs[type] = type_proportion
    
    return tar_obj, bac_obj

def load_TCGA_CCLE_data(hgnc_file = 'hgnc_complete_set_7.24.2018.txt',
              tumor_file = 'TCGA_mat.tsv',
              cell_line_file = 'CCLE_mat.csv',
              annotation_file = 'Celligner_info.csv'):

    hgnc_complete_set = pd.read_table(hgnc_file,sep = '\t')

    TCGA_mat = pd.read_table(tumor_file,sep = '\t',index_col = 0)
    TCGA_mat = TCGA_mat.T

    common_genes = np.intersect1d(TCGA_mat.columns.values,np.array(hgnc_complete_set['symbol'])).tolist()

    TCGA_mat = TCGA_mat[common_genes]

    hgnc_complete_set = hgnc_complete_set[hgnc_complete_set['symbol'].isin(common_genes)]
    hgnc_complete_set = hgnc_complete_set.drop_duplicates(subset='symbol')

    hgnc_complete_set = hgnc_complete_set.set_index('symbol')
    hgnc_complete_set = hgnc_complete_set.loc[common_genes,:] 

    TCGA_mat.columns = hgnc_complete_set['ensembl_gene_id']

    CCLE_mat = pd.read_table(cell_line_file, sep=',',index_col = 0)

    col = CCLE_mat.columns.tolist()
    res = []
    for item in col:
        res.append(re.findall("[(](.*?)[)]", item)[0])
    CCLE_mat.columns = res

    exclude = ['non-coding RNA','pseudogene']
    hgnc_complete_set = hgnc_complete_set[~hgnc_complete_set['locus_group'].isin(exclude)]
    func_genes = hgnc_complete_set['ensembl_gene_id']

    genes_used = np.intersect1d(list(CCLE_mat.columns.values), list(TCGA_mat.columns.values)).tolist()

    genes_used = np.intersect1d(genes_used, list(func_genes))

    TCGA_mat = TCGA_mat[genes_used]

    CCLE_mat = CCLE_mat[genes_used]

    hgnc_complete_set=hgnc_complete_set.reset_index().set_index('ensembl_gene_id').loc[genes_used,:]

    ann = pd.read_table(annotation_file, sep=',', index_col=0)

    col_ann = ann.columns.tolist()
    if ('UMAP_1' in col_ann):
        col_ann.remove('UMAP_1')
        ann = ann[col_ann]
    if ('UMAP_2' in col_ann):
        col_ann.remove('UMAP_2')
        ann = ann[col_ann]
    if ('cluster' in col_ann):
        col_ann.remove('cluster')
        ann = ann[col_ann]

    TCGA_ann = ann[ann['type'] == 'tumor']
    CCLE_ann = ann[ann['type'] == 'CL']

    result = [hgnc_complete_set, TCGA_mat, CCLE_mat, TCGA_ann, CCLE_ann]

    return result

def load_BrCa_data( scRNA_path = '../NG_BRCA/scRNA/',
                                   CID4465_path = '../NG_BRCA/CID4465/filtered_count_matrix/',
                                   CID44971_path = '../NG_BRCA/CID44971/filtered_count_matrix/'):
    
    singleCell_obj = sc.read_mtx(f'{scRNA_path}count_matrix_sparse.mtx').T

    singleCell_ann = pd.read_csv(f'{scRNA_path}metadata.csv', index_col=0)
    
    singleCell_geneSymbol = pd.read_table(f'{scRNA_path}count_matrix_genes.tsv', sep='\t')
    singleCell_geneSymbol = np.concatenate((singleCell_geneSymbol.columns,np.concatenate(singleCell_geneSymbol.values)))
    singleCell_geneSymbol = pd.DataFrame({'symbol':singleCell_geneSymbol}, index=singleCell_geneSymbol)
    
    singleCell_obj.obs = singleCell_ann
    singleCell_obj.var = singleCell_geneSymbol

    spatial_obj_CID4465 = sc.read_mtx(f'{CID4465_path}matrix.mtx').T

    spatial_obj_CID4465_geneSymbol = pd.read_table(f'{CID4465_path}features.tsv', sep='\t')
    spatial_obj_CID4465_geneSymbol = np.concatenate((spatial_obj_CID4465_geneSymbol.columns,np.concatenate(spatial_obj_CID4465_geneSymbol.values)))
    spatial_obj_CID4465_geneSymbol = pd.DataFrame({'symbol':spatial_obj_CID4465_geneSymbol}, index=spatial_obj_CID4465_geneSymbol)

    spatial_obj_CID4465_spotID = pd.read_table(f'{CID4465_path}barcodes.tsv', sep='\t')
    spatial_obj_CID4465_spotID = np.concatenate((spatial_obj_CID4465_spotID.columns,np.concatenate(spatial_obj_CID4465_spotID.values)))
    spatial_obj_CID4465_spotID = pd.DataFrame({'spotID':spatial_obj_CID4465_spotID}, index=spatial_obj_CID4465_spotID)

    spatial_obj_CID4465.obs = spatial_obj_CID4465_spotID
    spatial_obj_CID4465.var = spatial_obj_CID4465_geneSymbol

    spatial_obj_CID44971 = sc.read_mtx(f'{CID44971_path}matrix.mtx').T

    spatial_obj_CID44971_geneSymbol = pd.read_table(f'{CID44971_path}features.tsv', sep='\t')
    spatial_obj_CID44971_geneSymbol = np.concatenate((spatial_obj_CID44971_geneSymbol.columns,np.concatenate(spatial_obj_CID44971_geneSymbol.values)))
    spatial_obj_CID44971_geneSymbol = pd.DataFrame({'symbol':spatial_obj_CID44971_geneSymbol}, index=spatial_obj_CID44971_geneSymbol)

    spatial_obj_CID44971_spotID = pd.read_table(f'{CID44971_path}barcodes.tsv', sep='\t')
    spatial_obj_CID44971_spotID = np.concatenate((spatial_obj_CID44971_spotID.columns,np.concatenate(spatial_obj_CID44971_spotID.values)))
    spatial_obj_CID44971_spotID = pd.DataFrame({'spotID':spatial_obj_CID44971_spotID}, index=spatial_obj_CID44971_spotID)

    spatial_obj_CID44971.obs = spatial_obj_CID44971_spotID
    spatial_obj_CID44971.var = spatial_obj_CID44971_geneSymbol

    common_genes = np.intersect1d( list( np.intersect1d(
                                        np.concatenate(spatial_obj_CID4465.var.values),
                                        np.concatenate(singleCell_obj.var.values)
                                                 ) ),
                                   list( np.concatenate(spatial_obj_CID44971.var.values) )
                                 ).tolist()
    
    singleCell_obj, spatial_obj_CID4465, spatial_obj_CID44971 = singleCell_obj[:, common_genes], spatial_obj_CID4465[:, common_genes], spatial_obj_CID44971[:, common_genes]

    singleCell_obj_CID4465 = singleCell_obj[singleCell_obj.obs['orig.ident']=='CID4465',:]
    singleCell_obj_CID44971 = singleCell_obj[singleCell_obj.obs['orig.ident']=='CID44971',:]

    return dict( singleCell_obj_CID4465 = singleCell_obj_CID4465,
                 singleCell_obj_CID44971 = singleCell_obj_CID44971,
                 spatial_obj_CID4465 = spatial_obj_CID4465,
                 spatial_obj_CID44971 = spatial_obj_CID44971 )