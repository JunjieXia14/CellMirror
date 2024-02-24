import warnings
warnings.filterwarnings("ignore")

import os
os.environ['R_HOME'] = '/sibcb2/chenluonanlab7/zuochunman/anaconda3/envs/r4.0/lib/R'
os.environ['R_USER'] = '/sibcb2/chenluonanlab7/zuochunman/anaconda3/envs/CellMirror/lib/python3.8/site-packages/rpy2'

import random
import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt

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

args.lr_cLDVAE = 1e-5; args.gamma = -100

# Set workpath
Save_path = '/sibcb2/chenluonanlab7/zuochunman/Share_data/xiajunjie/PDAC/'

# Normalization
PDAC_A_st = sc.read_h5ad(Save_path + 'PDAC_A_st.h5ad')
PDAC_A_st.obs['type'] = 'Spatial'
PDAC_A_st.obs['y'] = 100 - PDAC_A_st.obs['y'] 
PDAC_A_st.obsm['spatial'] = PDAC_A_st.obs[['x','y']].values
sc.pp.normalize_total(PDAC_A_st)
sc.pp.log1p(PDAC_A_st)
sc.pp.highly_variable_genes(PDAC_A_st, flavor='seurat', n_top_genes=3000)

PDAC_A_sc = sc.read_h5ad(Save_path + 'PDAC_A_sc.h5ad')
PDAC_A_sc.obs['type'] = 'SingleCell'
sc.pp.normalize_total(PDAC_A_sc)
sc.pp.log1p(PDAC_A_sc)
sc.pp.highly_variable_genes(PDAC_A_sc, flavor='seurat', n_top_genes=3000)

common_HVGs=np.intersect1d(list(PDAC_A_sc.var.index[PDAC_A_sc.var['highly_variable']]),list(PDAC_A_st.var.index[PDAC_A_st.var['highly_variable']])).tolist()
print(len(common_HVGs))

PDAC_A_st, PDAC_A_sc = PDAC_A_st[:,common_HVGs], PDAC_A_sc[:,common_HVGs]

print('\nShape of target object: ', PDAC_A_st.shape, '\tShape of background object: ', PDAC_A_sc.shape)

# Last batch setting
args = set_last_batchsize(args, PDAC_A_st, PDAC_A_sc)

# Pseudo-data padding
pseudo_st, pseudo_sc = pseudo_data_padding(PDAC_A_st, PDAC_A_sc)

# Dataloader preparation
train = data_utils.TensorDataset(torch.from_numpy(pseudo_st),torch.from_numpy(pseudo_sc))
train_loader = data_utils.DataLoader(train, batch_size=args.batch_size, shuffle=True)

total = data_utils.TensorDataset(torch.from_numpy(pseudo_st),torch.from_numpy(pseudo_sc))
total_loader = data_utils.DataLoader(total, batch_size=args.batch_size, shuffle=False)

# Run cLDVAE
model_cLDVAE = cLDVAE(args=args, n_input=PDAC_A_st.shape[1]).cuda()
history = model_cLDVAE.fit(train_loader, total_loader)

# Pseudo-data deparsing
outputs = model_cLDVAE.predict(total_loader)
PDAC_A_st.obsm['cLDVAE'], PDAC_A_sc.obsm['cLDVAE'] = pseudo_data_deparser(PDAC_A_st, outputs['tg_z_outputs'], PDAC_A_sc, outputs['bg_z_outputs'])

# Run MNN
PDAC_A_st.obsm['CellMirror'], PDAC_A_sc.obsm['CellMirror'] = mnn_correct(PDAC_A_st.obsm['cLDVAE'], PDAC_A_sc.obsm['cLDVAE'])

# Cell type estimation
PDAC_A_st, PDAC_A_sc = estimate_cell_type(PDAC_A_st, PDAC_A_sc, used_obsm='CellMirror', used_label='cell_type_ductal', neighbors=50)

# Save data
colors = np.unique(PDAC_A_sc.obs['cell_type_ductal'].values).tolist()
sc.pl.spatial(PDAC_A_st, img_key=None, color=colors, color_map=plt.cm.get_cmap('plasma'), ncols=5, spot_size=1, frameon=False, show=False)
plt.savefig(Save_path + 'PDAC_A_CellMirror_celltype_proportion.jpg', dpi=100)

PDAC_A_st.write(Save_path + 'PDAC_A_CellMirror.h5ad', compression='gzip')