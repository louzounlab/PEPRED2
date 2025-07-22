from CNN.preprocess_grid import preprocess_data
import pandas as pd

def preprocess_(data_path, tax=7):
    """
    git pre-process for donors- enables to determine number of rare bacterias
    :param donors_path: path of donors
    :return: preprocessed donors df
    """
    donors = pd.read_csv(data_path, index_col=0,)
    fff = {'taxonomy_level': tax, 'taxnomy_group':'sub PCA', 'epsilon': 0.1,
           'normalization': "log", 'z_scoring': 'Row', 'norm_after_rel': 'No',
           'correlation_threshold': None, 'rare_bacteria_threshold':-1,#5
           'pca': (0, 'PCA'), 'std_to_delete': -1} #'sub PCA','row'
    as_data_frame, as_data_frame_b_pca, pca_obj, bacteria, pca = (
        preprocess_data(donors, fff, None))
    return as_data_frame

if __name__ == '__main__':

    data_for_preprocess = pd.read_csv(
        "../VS_OTU.csv")
    preprocessed = preprocess_(
        "../VS_OTU.csv")
    preprocessed.to_csv(
        "../old_mipmlp.csv")
    exit(0)


    # data_for_preprocess = pd.read_csv(
    #     "Data_for_git_process/PNAS/otus_with_tax.csv")
    # preprocessed = preprocess_(
    #     "Data_for_git_process/PNAS/otus_with_tax.csv")
    #
    # preprocessed.to_csv(
    #     "Data_TS/PNAS/otu_relative_mean_rare_bact_5_tax_7.csv")#including_rare/
    # exit(0)

    data_for_preprocess = pd.read_csv("Data_for_git_process/CRC/relative_otus_tax_6.csv")
    preprocessed = preprocess_("Data_for_git_process/CRC/relative_otus_tax_6.csv")
    preprocessed.to_csv(
        "Data_after_process_jacard_index/CRC/log_sub_pca_tax_4.csv")
    exit(0)
    # # allergy_time_predictions new:
    data_for_preprocess = pd.read_csv("Data_for_git_process/Allergy/Allergy_otus_for_preprocess.csv")
    preprocessed = preprocess_("Data_for_git_process/Allergy/Allergy_otus_for_preprocess.csv")
    preprocessed.to_csv(
        "outside_projects_data_for_git_preprocess/Allergy/After_preprocess/tax_7_subpca_log_pca_2.csv")
    exit(0)
    # thesis
    data_for_preprocess = pd.read_csv("Data_for_git_process/Allergy/Allergy_otus_for_preprocess.csv")
    preprocessed = preprocess_("Data_for_git_process/Allergy/Allergy_otus_for_preprocess.csv")
    preprocessed.to_csv(
        "Data/New_Allergy/tax_7_log_sub_pca_pca_4.csv")
    exit(0)

    # bgu
    data_for_preprocess = pd.read_csv("Datasets_for_thesis_2_preprocess/Parkinson/for_git_preprocess.csv")
    preprocessed = preprocess_("Datasets_for_thesis_2_preprocess/Parkinson/for_git_preprocess.csv")
    preprocessed.to_csv(
        "After_process_Thesis_2/Parkinson/tax_5_relative_mean.csv")
    exit(0)
    # Mimenet:
    #data_for_preprocess = pd.read_csv("Data_for_git_process/Ben_Gurion/merged_otu_tax.csv")
    #preprocessed = preprocess_("Data_for_git_process/Ben_Gurion/merged_otu_tax.csv")
    #preprocessed.to_csv("Data/Mimenet_send_reut.csv")
    # popphy T2D
    data_for_preprocess = pd.read_csv("Data_for_git_process/PopPhy_data/T2D/for_preprocess.csv")
    preprocessed = preprocess_("Data_for_git_process/PopPhy_data/T2D/for_preprocess.csv")
    preprocessed.to_csv(
        "Data/PopPhy_data/T2D/T2D_otu_mean_relative_rare_bact_5_tax_7_only_after_git_preprocess.csv")
    exit(0)
    # White_vs_black_vagina
    data_for_preprocess = pd.read_csv("Data_for_git_process/White_vs_black_vagina/try.csv")
    preprocessed = preprocess_("Data_for_git_process/White_vs_black_vagina/try.csv")
    preprocessed.to_csv(
        "Data/White_vs_black_vagina/White_vs_black_vagina_otu_sum_relative_rare_bact_5_tax_7_only_after_git_preprocess.csv")
    exit(0)
    # Male_vs_female
    data_for_preprocess = pd.read_csv("Data_for_git_process/Male_vs_female/try.csv")
    preprocessed = preprocess_("Data_for_git_process/Male_vs_female/try.csv")
    preprocessed.to_csv(
        "Data/Male_vs_female/Elderly_vs_young_otu_sub_pca_log_rare_bact_5_tax_7_only_after_git_preprocess.csv")
    exit(0)
    # Elderly_vs_young
    data_for_preprocess = pd.read_csv("Data_for_git_process/Elderly_vs_young/try.csv")
    preprocessed = preprocess_("Data_for_git_process/Elderly_vs_young/try.csv")
    preprocessed.to_csv(
        "Data/Elderly_vs_young/Elderly_vs_young_otu_sum_relative_rare_bact_5_tax_7_only_after_git_preprocess.csv")
    exit(0)
    # Cirrhosis_Knight_Lab
    data_for_preprocess = pd.read_csv("Data_for_git_process/Cirrhosis_Knight_Lab/try.csv")
    preprocessed = preprocess_("Data_for_git_process/Cirrhosis_Knight_Lab/try.csv")
    preprocessed.to_csv(
        "Data/Cirrhosis_Knight_Lab/Cirrhosis_Knight_Lab_otu_sub_pca_log_rare_bact_5_tax_7_only_after_git_preprocess.csv")
    exit(0)
    # IBD_Knight_Lab
    data_for_preprocess = pd.read_csv("Data_for_git_process/IBD_Knight_Lab/for_preprocess.csv")
    preprocessed = preprocess_("Data_for_git_process/IBD_Knight_Lab/for_preprocess.csv")
    preprocessed.to_csv(
        "Data/IBD_Knight_Lab/IBD_Knight_Lab_otu_sub_pca_log_rare_bact_5_tax_7_only_after_git_preprocess.csv")
    exit(0)
    # Gastro_vs_oral
    data_for_preprocess = pd.read_csv("Data_for_git_process/Gastro_vs_oral/for_git_preprocess.csv")
    preprocessed = preprocess_("Data_for_git_process/Gastro_vs_oral/for_git_preprocess.csv")
    preprocessed.to_csv("Data/Gastro_vs_oral/Gastro_vs_oral_otu_sum_relative_rare_bact_5_tax_7_only_after_git_preprocess.csv")
    # GDM
    #data_for_preprocess = pd.read_csv("Raw_data/GDM_stool/table-with-taxonomy.csv")
    #preprocessed = preprocess_("Raw_data/GDM_stool/table-with-taxonomy.csv")
    #preprocessed.to_csv("Data/GDM/GDM_otu_sum_relative_rare_bact_5_tax_7_only_after_git_preprocess.csv")
    exit(0)
    # IBD
    #data_for_preprocess = pd.read_csv("Data_for_git_process/IBD/IBD_otus_for_preprocess.csv")
    #preprocessed = preprocess_("Data_for_git_process/IBD/IBD_otus_for_preprocess.csv")
    #preprocessed.to_csv("Data/IBD/IBD_otu_sum_relative_rare_bact_5_tax_7.csv")
    #exit(0)
    # Allergy
    data_for_preprocess = pd.read_csv("Data_for_git_process/Allergy/Allergy_otus_for_preprocess.csv")
    preprocessed = preprocess_("Data_for_git_process/Allergy/Allergy_otus_for_preprocess.csv")
    preprocessed.to_csv("Data/Allergy_otu_sub_pca_log_rare_bact_5_tax_7.csv")
    exit(0)
    # BGU
    data_for_preprocess = pd.read_csv("Data_for_git_process/Ben_Gurion/merged_otu_tax.csv")
    preprocessed = preprocess_("Data_for_git_process/Ben_Gurion/merged_otu_tax.csv")
    preprocessed.to_csv("Data/BGU_otu_sum_relative_rare_bact_5_tax_7.csv")
    ################################################################################
    X=5