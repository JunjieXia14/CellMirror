import random
import pandas as pd
import numpy as np
import scanpy as sc
import datetime
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from CellMirror_utils.utilities import *

parser = parameter_setting()
args = parser.parse_known_args()[0]

np.random.seed(args.seed)
random.seed(args.seed)

def labelTransfer(target_set_path, reference_set_path):

    result = load_TCGA_CCLE_data()

    TCGA_obj = sc.AnnData(X=result[1], obs=result[3], var=result[0])
    CCLE_obj = sc.AnnData(X=result[2], obs=result[4], var=result[0])

    s_latent_dim = 2
    z_latent_dim = 100

    salient_colnames = list(range(1, s_latent_dim + 1))
    for sColumn in range(s_latent_dim):
        salient_colnames[sColumn] = "s" + str(salient_colnames[sColumn])
    irrelevant_colnames = list(range(1, z_latent_dim + 1))
    for iColumn in range(z_latent_dim):
        irrelevant_colnames[iColumn] = "z" + str(irrelevant_colnames[iColumn])

    tg_z_output = pd.read_csv(target_set_path,index_col=0)
    tg_z_output.index = TCGA_obj.obs.index
    tg_z_output.columns = irrelevant_colnames
    bg_z_output = pd.read_csv(reference_set_path,index_col=0)
    bg_z_output.index = CCLE_obj.obs.index
    bg_z_output.columns = irrelevant_colnames

    noContamination_output = tg_z_output
    bg_output = bg_z_output

    common_lineages = np.intersect1d(CCLE_obj.obs['lineage'].values,TCGA_obj.obs['lineage'].values)
    common_lineages = common_lineages.tolist()

    CCLE_i1 = CCLE_obj.obs.sort_values(by='lineage')[CCLE_obj.obs.sort_values(by='lineage')['lineage'].isin(common_lineages)]
    CCLE_common_lineages_sampleID = CCLE_i1.index

    TCGA_i1 = TCGA_obj.obs.sort_values(by='lineage')[TCGA_obj.obs.sort_values(by='lineage')['lineage'].isin(common_lineages)]
    TCGA_common_lineages_sampleID = TCGA_i1.index

    TCGA_common_x_cLDVAE = noContamination_output.loc[TCGA_common_lineages_sampleID,:]
    CCLE_common_x_cLDVAE = bg_output.loc[CCLE_common_lineages_sampleID,:]

    tumor_CL_dist_cLDVAE = pd.DataFrame( np.corrcoef(TCGA_common_x_cLDVAE,CCLE_common_x_cLDVAE)[:len(TCGA_common_lineages_sampleID),-len(CCLE_common_lineages_sampleID):] , index=TCGA_common_lineages_sampleID, columns=CCLE_common_lineages_sampleID)

    CL_tumor_class_cLDVAE = []
    for CL in tumor_CL_dist_cLDVAE.columns:
        CL_tumor_class_cLDVAE.append( TCGA_i1.loc[ tumor_CL_dist_cLDVAE[CL].sort_values(ascending=False).index[:10] ]['lineage'].value_counts( ascending=False ).index[0] )

    temp1= CCLE_i1
    temp1['new_class'] = CL_tumor_class_cLDVAE
    aggrement_accuracy_CL = round( np.sum(temp1['new_class']==temp1['lineage'])/len(temp1), 3 )

    print(f"Agreement accuracy of CL: {aggrement_accuracy_CL}")
    temp1.to_csv(f'CL_tumor_classes_cLDVAE_only_lr{args.lr_cLDVAE}_beta{args.beta}_gamma{args.gamma}_bs{args.batch_size}_s_dim{s_latent_dim}_z_dim{z_latent_dim}_time{datetime.datetime.now()}.csv')

    tumor_CL_class_cLDVAE = []
    for tumor in tumor_CL_dist_cLDVAE.index:
        tumor_CL_class_cLDVAE.append(CCLE_i1.loc[tumor_CL_dist_cLDVAE.loc[tumor].sort_values(ascending=False).index[:10]]['lineage'].value_counts(ascending=False).index[0])

    temp2 = TCGA_i1
    temp2['new_class'] = tumor_CL_class_cLDVAE
    aggrement_accuracy_tumor = round( np.sum(temp2['new_class']==temp2['lineage'])/len(temp2), 3 )

    print(f"Agreement accuracy of tumor: {aggrement_accuracy_tumor}")
    temp2.to_csv(f'tumor_CL_classes_cLDVAE_only_lr{args.lr_cLDVAE}_beta{args.beta}_gamma{args.gamma}_bs{args.batch_size}_s_dim{s_latent_dim}_z_dim{z_latent_dim}_time{datetime.datetime.now()}.csv')

def Visualization(target_set_path, reference_set_path):

    result = load_TCGA_CCLE_data()

    TCGA_obj = sc.AnnData(X=result[1], obs=result[3], var=result[0])
    CCLE_obj = sc.AnnData(X=result[2], obs=result[4], var=result[0])

    s_latent_dim = 2
    z_latent_dim = 100

    salient_colnames = list(range(1, s_latent_dim + 1))
    for sColumn in range(s_latent_dim):
        salient_colnames[sColumn] = "s" + str(salient_colnames[sColumn])
    irrelevant_colnames = list(range(1, z_latent_dim + 1))
    for iColumn in range(z_latent_dim):
        irrelevant_colnames[iColumn] = "z" + str(irrelevant_colnames[iColumn])

    tg_z_output = pd.read_csv(target_set_path,index_col=0)
    tg_z_output.index = TCGA_obj.obs.index
    tg_z_output.columns = irrelevant_colnames
    bg_z_output = pd.read_csv(reference_set_path,index_col=0)
    bg_z_output.index = CCLE_obj.obs.index
    bg_z_output.columns = irrelevant_colnames

    noContamination_output = tg_z_output
    bg_output = bg_z_output

    cLDVAE_mnn_obj = sc.AnnData( pd.concat([noContamination_output, bg_output], axis = 0), pd.concat([TCGA_obj.obs, CCLE_obj.obs], axis=0), pd.DataFrame(irrelevant_colnames,index=irrelevant_colnames) )
    sc.pp.neighbors(cLDVAE_mnn_obj, n_neighbors=10, metric='correlation',use_rep='X')
    sc.tl.umap(cLDVAE_mnn_obj,min_dist=0.5)
    cLDVAE_mnn_obj.obs = cLDVAE_mnn_obj.obs.merge(cLDVAE_mnn_obj.obsm.to_df()[['X_umap1','X_umap2']], how='inner', left_index=True, right_index=True)
    cLDVAE_mnn_obj.obs.to_csv(f"en[1000]_de[1000]_s2_z100_beta1_gamma-100_lr3e-6_4th_cLDVAE_mnn_k1_80_k2_100_comb_Ann_time{datetime.datetime.now()}.csv")

def Interpretability(salient_features_path, salient_loadings_path, gene_stats_path):

    Z_df = pd.read_csv(salient_features_path,index_col=0)
    W_df = pd.read_csv(salient_loadings_path,index_col=0)

    def Z_covariance(Z):
        Zcentered = Z - Z.mean(0)
        Zscaled = Zcentered / Z.std(0)
        ZTZ = np.cov(Zscaled.T)
        
        eigen_values, _ = np.linalg.eig(ZTZ)
        singular_values = np.sqrt(eigen_values)
        variance_explained = singular_values / singular_values.sum()

        return ZTZ, variance_explained

    _, variance_explained = Z_covariance(Z_df)
    idx = np.argsort(variance_explained)[::-1]

    variance_explained_df = pd.DataFrame({'variance_explained':variance_explained[idx]},index=idx+1)

    Z_df_ordered = Z_df.iloc[:,idx]
    W_df_ordered = W_df.iloc[:,idx]

    gene_stats=pd.read_csv(gene_stats,index_col=0)

    W_df_ordered = W_df_ordered.merge(gene_stats.loc[W_df_ordered.index,['symbol']], how='inner', left_index=True, right_index=True)
    
    text_shift = { (0, 'IGHG1'): (0, 0),
                (0, 'IGHG4'): (0, 0.1),
                (0, 'ALDOB'): (0, 0),
                (0, 'IGHG2'): (0, 0.1),
                (0, 'IGKC'): (0, 0),
                (0, 'IGHGP'): (0.1, -0.05),
                (0, 'IGKV2-28'): (0.1, 0.01),
                (0, 'IGKV1D-39'): (0, 0.1) }

    plt.figure(figsize=(18,8))

    for i in range(1):

        # -- tg_s plot -- 

        plt.subplot(1, 2, 2 * i + 1)

        plt.hist2d(
            Z_df_ordered[f's{2 * i + 1}'], Z_df_ordered[f's{2 * (i + 1)}'],
            bins=256,
            norm=mcolors.PowerNorm(0.25),
            cmap=plt.cm.gray_r,
            rasterized=True
        )

        plt.axis('equal');
        plt.xlabel(f'cLDVAE salient feature {2 * i + 1} ({round(variance_explained_df.loc[2 * i +1].values[0] * 100, 2)}% variance)')
        plt.ylabel(f'cLDVAE salient feature {2 * (i + 1)} ({round(variance_explained_df.loc[2 * (i +1)].values[0] * 100, 2)}% variance)')

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # -- W plot -- 

        plt.subplot(1, 2, 2 * (i + 1))

        w_columns = [f's{2 * i + 1}', f's{2 * (i + 1)}']

        sns.kdeplot(
            W_df_ordered[w_columns[0]], W_df_ordered[w_columns[1]],
            cmap='Blues_r',
            rasterized=True
        )

        plt.axis('equal');
        plt.xlabel(f'cLDVAE $W_{2 * i + 1}$',fontdict={'size':21})
        plt.ylabel(f'cLDVAE $W_{2 * (i + 1)}$',fontdict={'size':21})
        plt.xticks(size=18)
        plt.yticks(size=18)

        tmp_ = W_df_ordered.copy()
        tmp_['lnth'] = np.linalg.norm(tmp_[w_columns], axis=1)
        
        ggg=tmp_.sort_values('lnth', ascending=False).head(8)[['symbol', 'lnth', *w_columns]]
        print(ggg[['symbol', *w_columns]].values)

        texts = []
        arrows = []
        for g, r in ggg.iterrows():
            x_, y_ = r[w_columns[0]], r[w_columns[1]]
            
            ha = 'right'
            if x_ > 0:
                ha = 'left'
                
            va = 'top'
            if y_ > 0:
                va = 'bottom'
                
            arrows.append(plt.arrow(0, 0, x_, y_, length_includes_head=True, color='k'))

            xs, ys = 0, 0
            if (i, r.symbol) in text_shift:
                xs, ys = text_shift[(i, r.symbol)]
                texts.append(plt.text(x_ + xs, y_ + ys, r.symbol, ha=ha, va=va,fontdict={'fontsize':18}))

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_linewidth(3)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(3)

    plt.tight_layout()
    plt.savefig('elevated genes.png')

while True:
    print("*"*20)
    print('''1. Label transfer
2. Interpretability
3. Visualization
4. Quit''')
    print("*"*20)

    prompt = input("Select a downstream analysis:")

    if prompt==1:
        target_set_path = input("Select your target set processed by cLDVAE/CellMirror:")
        reference_set_path = input("Select your reference set processed by cLDVAE/CellMirror:")
        labelTransfer(target_set_path, reference_set_path)

    elif prompt==2:
        salient_features_path = input("Select your salient features learned by cLDVAE:")
        salient_loadings_path = input("Select your salient loadings learned by cLDVAE:")
        gene_stats_path = input("Select your gene stats file:")
        Interpretability(salient_features_path, salient_loadings_path, gene_stats_path)

    elif prompt==3:
        target_set_path = input("Select your target set processed by cLDVAE/CellMirror:")
        reference_set_path = input("Select your reference set processed by cLDVAE/CellMirror:")
        Visualization(target_set_path, reference_set_path)

    else:
        break