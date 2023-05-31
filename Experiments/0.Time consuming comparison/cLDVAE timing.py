import random
import pandas as pd
import numpy as np
import scanpy as sc
import datetime
import torch.utils.data as data_utils

from CellMirror_utils.utilities import *
from CellMirror_utils.layers import *
from CellMirror_utils.cLDVAE_torch import *

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

# load data
result = load_data()

# preprocessing
TCGA_obj = sc.AnnData(X=result[1], obs=result[3], var=result[0])
CCLE_obj = sc.AnnData(X=result[2], obs=result[4], var=result[0])
sc.pp.highly_variable_genes(TCGA_obj, n_top_genes=5000)
sc.pp.highly_variable_genes(CCLE_obj, n_top_genes=5000)
common_HVGs=np.intersect1d(list(TCGA_obj.var.index[TCGA_obj.var['highly_variable']]),list(CCLE_obj.var.index[CCLE_obj.var['highly_variable']])).tolist()
TCGA_obj_init, CCLE_obj_init = TCGA_obj[:,common_HVGs], CCLE_obj[:,common_HVGs]

# cLDVAE config
genes_info = TCGA_obj_init.var
batch_size=args.batch_size
input_dim=len(genes_info)
intermediate_dim_en=[1000]
intermediate_dim_de=[1000]
s_latent_dim = 2
z_latent_dim = 100
common_colnames = list(range(1, z_latent_dim + 1))
for zColumn in range(z_latent_dim):
    common_colnames[zColumn] = "z" + str(common_colnames[zColumn])

# data for Celligner
pd.DataFrame(TCGA_obj_init.X, index=TCGA_obj_init.obs.index, columns=genes_info.index).to_csv('TCGA_CCLE_data_tumor_X.csv')
pd.DataFrame(CCLE_obj_init.X, index=CCLE_obj_init.obs.index, columns=genes_info.index).to_csv('TCGA_CCLE_data_CL_X.csv')

# adding cell num
cell_num_list = np.arange(1000, 14000, 1000)
running_time_list = []

for cell_num in cell_num_list:

    if cell_num == 1000:
        TCGA_obj = TCGA_obj_init[:cell_num,:]
        CCLE_obj = CCLE_obj_init[:cell_num,:]
    elif cell_num == 13000:
        TCGA_obj, CCLE_obj = TCGA_obj_init, CCLE_obj_init
    else:
        TCGA_obj = TCGA_obj_init[:cell_num,:]
        CCLE_obj = CCLE_obj_init

    TCGA_obj_X_df = pd.DataFrame(TCGA_obj.X, index=TCGA_obj.obs.index, columns=genes_info.index)
    TCGA_obj_X_df = TCGA_obj_X_df - np.mean(TCGA_obj_X_df, axis=0)
    CCLE_obj_X_df = pd.DataFrame(CCLE_obj.X, index=CCLE_obj.obs.index, columns=genes_info.index)
    CCLE_obj_X_df = CCLE_obj_X_df - np.mean(CCLE_obj_X_df, axis=0)

    print(f"Tumor number: {len(TCGA_obj_X_df)}\nCL number: {len(CCLE_obj_X_df)}")

    # cLDVAE

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

    # training

    model_cLDVAE = cLDVAE(args=args, 
                        n_input = input_dim, 
                        n_hidden_en = intermediate_dim_en, n_hidden_de = intermediate_dim_de, 
                        n_latent_s = s_latent_dim, n_latent_z = z_latent_dim)

    if args.use_cuda:
        model_cLDVAE.cuda()

    history = model_cLDVAE.fit()

    duration = history['duration']

    running_time_list.append(duration)

    outputs = model_cLDVAE.predict(total_loader)
    tg_z_output = outputs["tg_z_outputs"]
    bg_z_output = outputs["bg_z_outputs"]

    noContamination_output = pd.DataFrame(tg_z_output, index=TCGA_obj.obs.index, columns=common_colnames)
    bg_output = pd.DataFrame(bg_z_output[:len(CCLE_obj.obs),:], index=CCLE_obj.obs.index, columns=common_colnames)

    print(f"Saved tumor number: {len(noContamination_output)}\nSaved CL number: {len(bg_output)}")

    # data for CellMirror
    noContamination_output.to_csv(f'cell_num_{cell_num}_TCGA_CCLE_data_tumor_X_cLDVAE_only.csv')
    bg_output.to_csv(f'cell_num_{cell_num}_TCGA_CCLE_data_CL_X_cLDVAE_only.csv')

pd.Series(running_time_list).to_csv('cLDVAE_timing.csv',index=False)