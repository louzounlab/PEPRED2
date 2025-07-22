import sys, os
from create_otu_and_mapping_files import CreateOtuAndMappingFiles
import  preprocess_grid
import pandas as pd
# direc = "General_files"
# dataset = "VS_OTU"
# tag = "VS_TAG"
#
# file_path = os.path.join(direc, f"{dataset}.csv")
# # Log normalization
# preprocess_prms = {'taxonomy_level': 7, 'taxnomy_group': 'sub PCA', 'epsilon': 0.1,
#                      'normalization': 'log', 'z_scoring': 'Row', 'norm_after_rel': 'No',
#                      'std_to_delete': 0, 'pca': (0, 'PCA'), "rare_bacteria_threshold":-1}
#
#
# otu_file = file_path
# tag_file = os.path.join(direc, f"{tag}.csv")
# task_name = dataset
#
# mapping_file1 = CreateOtuAndMappingFiles(otu_file, tags_file_path=tag_file)
# mapping_file1.preprocess(preprocess_params=preprocess_prms, visualize=False, ip='127.0.0.1')
# print(mapping_file1.otu_features_df_b_pca)
# mapping_file1.otu_features_df_b_pca.to_csv('General_files/after_pp.csv')
#

df = pd.read_csv('../../PycharmProjects/swaps/VS_otu_genus.csv')
df = preprocess_grid.fill_taxonomy(df, 'columns')
df.to_csv('../../PycharmProjects/swaps/VS_otu_genus.csv')


