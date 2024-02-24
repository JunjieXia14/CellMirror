import warnings
warnings.filterwarnings("ignore")

import os
os.environ['R_HOME'] = '/sibcb2/chenluonanlab7/zuochunman/anaconda3/envs/r4.0/lib/R'
os.environ['R_USER'] = '/sibcb2/chenluonanlab7/zuochunman/anaconda3/envs/CellMirror/lib/python3.8/site-packages/rpy2'

import random
import numpy as np
import scanpy as sc

from CellMirror_utils.utilities import *
from CellMirror_utils.layers import *
from CellMirror_utils.cLDVAE_torch import *
import torch.utils.data as data_utils

parser = parameter_setting()
args = parser.parse_known_args()[0]

# Set seed
np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Changeable parameters
args.n_hidden_en = [1000]; args.n_hidden_de = [1000]
args.lr_cLDVAE = 3e-6; args.beta = 1; args.gamma = -100
args.n_latent_s = 2; args.n_latent_z = 100

# Set workpath
Save_path = '/sibcb2/chenluonanlab7/zuochunman/Share_data/xiajunjie/TCGA_CCLE/'

# load data & preprocessing
result = load_TCGA_CCLE_data()
TCGA_obj = sc.AnnData(X=result[1], obs=result[3], var=result[0])
CCLE_obj = sc.AnnData(X=result[2], obs=result[4], var=result[0])

sc.pp.highly_variable_genes(TCGA_obj, n_top_genes=5000)
sc.pp.highly_variable_genes(CCLE_obj, n_top_genes=5000)
common_HVGs=np.intersect1d(list(TCGA_obj.var.index[TCGA_obj.var['highly_variable']]),list(CCLE_obj.var.index[CCLE_obj.var['highly_variable']])).tolist()

TCGA_obj, CCLE_obj = TCGA_obj[:,common_HVGs], CCLE_obj[:,common_HVGs]
print(len(common_HVGs))

print('\nShape of target object: ', TCGA_obj.shape, '\tShape of background object: ', CCLE_obj.shape)

# Last batch setting
args = set_last_batchsize(args, TCGA_obj, CCLE_obj)

# Pseudo-data padding
TCGA_obj.X = TCGA_obj.X - np.mean(TCGA_obj.X, axis=0); CCLE_obj.X = CCLE_obj.X - np.mean(CCLE_obj.X, axis=0)
pseudo_TCGA, pseudo_CCLE = pseudo_data_padding(TCGA_obj, CCLE_obj)

# Dataloader preparation
train = data_utils.TensorDataset(torch.from_numpy(pseudo_TCGA),torch.from_numpy(pseudo_CCLE))
train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True)

total = data_utils.TensorDataset(torch.from_numpy(pseudo_TCGA),torch.from_numpy(pseudo_CCLE))
total_loader = data_utils.DataLoader(total, batch_size=args.batch_size, shuffle=False)

# Run cLDVAE
model_cLDVAE = cLDVAE(args=args, n_input=TCGA_obj.shape[1]).cuda()
history = model_cLDVAE.fit(train_loader, total_loader)

# Pseudo-data deparsing
outputs = model_cLDVAE.predict(total_loader)
TCGA_obj.obsm['cLDVAE'], CCLE_obj.obsm['cLDVAE'] = pseudo_data_deparser(TCGA_obj, outputs['tg_z_outputs'], CCLE_obj, outputs['bg_z_outputs'])

TCGA_obj.obsm['salient_features'], _ = pseudo_data_deparser(TCGA_obj, outputs['tg_s_outputs'], CCLE_obj, outputs['bg_s_outputs'])

TCGA_obj.uns['loadings'] = CCLE_obj.uns['loadings'] = model_cLDVAE.get_loadings()

# Run MNN
TCGA_obj.obsm['CellMirror'], CCLE_obj.obsm['CellMirror'] = mnn_correct(TCGA_obj.obsm['cLDVAE'], CCLE_obj.obsm['cLDVAE'])

# Save data
TCGA_obj.write(Save_path + 'TCGA_CellMirror.h5ad', compression='gzip')
CCLE_obj.write(Save_path + 'CCLE_CellMirror.h5ad', compression='gzip')