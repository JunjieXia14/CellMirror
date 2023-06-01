import math
import random
import pandas as pd
import numpy as np
import scanpy as sc
import datetime

from CellMirror_utils.utilities import *
from CellMirror_utils.layers import *
from CellMirror_utils.cLDVAE_torch import *
import torch.utils.data as data_utils

parser = parameter_setting()
args = parser.parse_known_args()[0]

args.batch_size = 128
args.lr_cLDVAE = 3e-6
args.beta = 1
args.gamma = -100

np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# load data & preprocessing

result = load_data()

TCGA_obj = sc.AnnData(X=result[1], obs=result[3], var=result[0])
CCLE_obj = sc.AnnData(X=result[2], obs=result[4], var=result[0])

sc.pp.highly_variable_genes(TCGA_obj, n_top_genes=5000)
sc.pp.highly_variable_genes(CCLE_obj, n_top_genes=5000)
common_HVGs=np.intersect1d(list(TCGA_obj.var.index[TCGA_obj.var['highly_variable']]),list(CCLE_obj.var.index[CCLE_obj.var['highly_variable']])).tolist()

TCGA_obj, CCLE_obj = TCGA_obj[:,common_HVGs], CCLE_obj[:,common_HVGs]
genes_info = TCGA_obj.var
TCGA_obj_X_df = pd.DataFrame(TCGA_obj.X, index=TCGA_obj.obs.index, columns=genes_info.index)
TCGA_obj_X_df = TCGA_obj_X_df - np.mean(TCGA_obj_X_df, axis=0)
CCLE_obj_X_df = pd.DataFrame(CCLE_obj.X, index=CCLE_obj.obs.index, columns=genes_info.index)
CCLE_obj_X_df = CCLE_obj_X_df - np.mean(CCLE_obj_X_df, axis=0)

# cLDVAE config

batch_size=args.batch_size
input_dim=len(genes_info)
intermediate_dim_en=[1000]
intermediate_dim_de=[1000]

s_latent_dim = 2
z_latent_dim = 100

salient_colnames = list(range(1, s_latent_dim + 1))
for sColumn in range(s_latent_dim):
    salient_colnames[sColumn] = "s" + str(salient_colnames[sColumn])
irrelevant_colnames = list(range(1, z_latent_dim + 1))
for iColumn in range(z_latent_dim):
    irrelevant_colnames[iColumn] = "z" + str(irrelevant_colnames[iColumn])

n = TCGA_obj.X.shape[0]
args.last_batch_size = n - int(n/batch_size)*batch_size

tumor_scale = TCGA_obj_X_df.values

CL_scale = np.concatenate( (CCLE_obj_X_df.values, np.random.multivariate_normal(np.mean(CCLE_obj_X_df, axis=0), np.cov(CCLE_obj_X_df.T), len(TCGA_obj_X_df)-len(CCLE_obj_X_df))), axis=0 )

background = (CL_scale).astype('float32')

target = (tumor_scale).astype('float32')

train = data_utils.TensorDataset(torch.from_numpy(target),torch.from_numpy(background))
train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)

total = data_utils.TensorDataset(torch.from_numpy(target),torch.from_numpy(background))
total_loader = data_utils.DataLoader(total, batch_size=batch_size, shuffle=False)

model_cLDVAE = cLDVAE(args=args, 
                    n_input = input_dim, 
                    n_hidden_en = intermediate_dim_en, n_hidden_de = intermediate_dim_de, 
                    n_latent_s = s_latent_dim, n_latent_z = z_latent_dim)

if args.use_cuda:
    model_cLDVAE.cuda()

history = model_cLDVAE.fit()

outputs = model_cLDVAE.predict(total_loader)
tg_z_output = outputs["tg_z_outputs"]
bg_z_output = outputs["bg_z_outputs"]

noContamination_output = pd.DataFrame(tg_z_output, index=TCGA_obj.obs.index, columns=irrelevant_colnames)

bg_output = pd.DataFrame(bg_z_output[:len(CCLE_obj.obs),:], index=CCLE_obj.obs.index, columns=irrelevant_colnames)

# data for interpretability, MNN & visualization 

noContamination_output.to_csv('TCGA_CCLE_data_tumor_X_cLDVAE_only.csv')
bg_output.to_csv('TCGA_CCLE_data_CL_X_cLDVAE_only.csv')

cLDVAE_only_obj = sc.AnnData( pd.concat([noContamination_output, bg_output], axis = 0), pd.concat([TCGA_obj.obs, CCLE_obj.obs], axis=0), pd.DataFrame(irrelevant_colnames,index=irrelevant_colnames) )
sc.pp.neighbors(cLDVAE_only_obj, n_neighbors=10, metric='correlation',use_rep='X')
sc.tl.umap(cLDVAE_only_obj,min_dist=0.5)
cLDVAE_only_obj.obs = cLDVAE_only_obj.obs.merge(cLDVAE_only_obj.obsm.to_df()[['X_umap1','X_umap2']], how='inner', left_index=True, right_index=True)
cLDVAE_only_obj.obs.to_csv(f"en{intermediate_dim_en}_de{intermediate_dim_de}_cLDVAE_only_comb_Ann_lr{args.lr_cLDVAE}_beta{args.beta}_gamma{args.gamma}_bs{args.batch_size}_time{datetime.datetime.now()}.csv")

tg_s_output = outputs["tg_s_outputs"]
tg_s_output = pd.DataFrame(tg_s_output, index=TCGA_obj.obs.index, columns=salient_colnames)
tg_s_output.to_csv(f"en{intermediate_dim_en}_de{intermediate_dim_de}_cLDVAE_only_TCGA_salient_features_lr{args.lr_cLDVAE}_beta{args.beta}_gamma{args.gamma}_bs{args.batch_size}_dim{s_latent_dim}_time{datetime.datetime.now()}.csv")

s_loadings_output = model_cLDVAE.get_loadings()[:,-(s_latent_dim):]
s_loadings_output = pd.DataFrame(s_loadings_output, index=genes_info.index, columns=salient_colnames)
s_loadings_output.to_csv(f"en{intermediate_dim_en}_de{intermediate_dim_de}_cLDVAE_only_salient_loadings_matrix_lr{args.lr_cLDVAE}_beta{args.beta}_gamma{args.gamma}_bs{args.batch_size}_time{datetime.datetime.now()}.csv")