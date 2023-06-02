# common_lineages = np.intersect1d(CCLE_obj.obs['lineage'].values,TCGA_obj.obs['lineage'].values)
# common_lineages = common_lineages.tolist()

# CCLE_i1 = CCLE_obj.obs.sort_values(by='lineage')[CCLE_obj.obs.sort_values(by='lineage')['lineage'].isin(common_lineages)]
# CCLE_common_lineages_sampleID = CCLE_i1.index

# TCGA_i1 = TCGA_obj.obs.sort_values(by='lineage')[TCGA_obj.obs.sort_values(by='lineage')['lineage'].isin(common_lineages)]
# TCGA_common_lineages_sampleID = TCGA_i1.index

# TCGA_common_x_cLDVAE = noContamination_output.loc[TCGA_common_lineages_sampleID,:]
# CCLE_common_x_cLDVAE = bg_output.loc[CCLE_common_lineages_sampleID,:]

# tumor_CL_dist_cLDVAE = pd.DataFrame( np.corrcoef(TCGA_common_x_cLDVAE,CCLE_common_x_cLDVAE)[:len(TCGA_common_lineages_sampleID),-len(CCLE_common_lineages_sampleID):] , index=TCGA_common_lineages_sampleID, columns=CCLE_common_lineages_sampleID)

# CL_tumor_class_cLDVAE = []
# for CL in tumor_CL_dist_cLDVAE.columns:
#     CL_tumor_class_cLDVAE.append( TCGA_i1.loc[ tumor_CL_dist_cLDVAE[CL].sort_values(ascending=False).index[:10] ]['lineage'].value_counts( ascending=False ).index[0] )

# temp1= CCLE_i1
# temp1['new_class'] = CL_tumor_class_cLDVAE
# aggrement_accuracy_CL = round( np.sum(temp1['new_class']==temp1['lineage'])/len(temp1), 3 )

# print(f"Agreement accuracy of CL: {aggrement_accuracy_CL}")
# temp1.to_csv(f'CL_tumor_classes_cLDVAE_only_lr{args.lr_cLDVAE}_beta{args.beta}_gamma{args.gamma}_bs{args.batch_size}_s_dim{s_latent_dim}_z_dim{z_latent_dim}_time{datetime.datetime.now()}.csv')

# tumor_CL_class_cLDVAE = []
# for tumor in tumor_CL_dist_cLDVAE.index:
#     tumor_CL_class_cLDVAE.append(CCLE_i1.loc[tumor_CL_dist_cLDVAE.loc[tumor].sort_values(ascending=False).index[:10]]['lineage'].value_counts(ascending=False).index[0])

# temp2 = TCGA_i1
# temp2['new_class'] = tumor_CL_class_cLDVAE
# aggrement_accuracy_tumor = round( np.sum(temp2['new_class']==temp2['lineage'])/len(temp2), 3 )

# print(f"Agreement accuracy of tumor: {aggrement_accuracy_tumor}")
# temp2.to_csv(f'tumor_CL_classes_cLDVAE_only_lr{args.lr_cLDVAE}_beta{args.beta}_gamma{args.gamma}_bs{args.batch_size}_s_dim{s_latent_dim}_z_dim{z_latent_dim}_time{datetime.datetime.now()}.csv')

while True:
    print("*"*20)
    print('''1. Label transfer
2. Interpretability
3. Visualization
4. Quit''')
    print("*"*20)

    prompt = input("Select a downstream analysis:")

    if prompt==1:
        break
    elif prompt==2:
        break
    elif prompt==3:
        break
    else:
        break