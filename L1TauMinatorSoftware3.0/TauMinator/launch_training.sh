# sparsity_values="0.0 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.80 0.90"

# for spars in $sparsity_values
# do 
#     python3 TauMinator_CB_ident.py --date 2023_07_04 --v 21 --caloClNxM 5x9 --sparsity $spars #--train
#     python3 TauMinator_CB_ident.py --date 2023_07_04 --v 21 --caloClNxM 5x9 --sparsity $spars --pt_weighted #--train
    
#     python3 TauMinator_CE_ident.py --date 2023_07_04 --v 21 --caloClNxM 5x9 --sparsity $spars #--train
#     python3 TauMinator_CE_ident.py --date 2023_07_04 --v 21 --caloClNxM 5x9 --sparsity $spars --pt_weighted #--train

#     #------------------------------------------------------------------------------------------------------

#     python3 TauMinator_CB_ident.py --date 2023_07_04 --v 21p1 --caloClNxM 5x9 --sparsity $spars #--train
#     python3 TauMinator_CB_ident.py --date 2023_07_04 --v 21p1 --caloClNxM 5x9 --sparsity $spars --pt_weighted #--train

#     python3 TauMinator_CE_ident.py --date 2023_07_04 --v 21p1 --caloClNxM 5x9 --sparsity $spars #--train
#     python3 TauMinator_CE_ident.py --date 2023_07_04 --v 21p1 --caloClNxM 5x9 --sparsity $spars --pt_weighted #--train

#     #------------------------------------------------------------------------------------------------------

#     python3 TauMinator_CB_ident.py --date 2023_07_04 --v 21p2 --caloClNxM 5x9 --sparsity $spars #--train
#     python3 TauMinator_CB_ident.py --date 2023_07_04 --v 21p2 --caloClNxM 5x9 --sparsity $spars --pt_weighted #--train

#     python3 TauMinator_CE_ident.py --date 2023_07_04 --v 21p2 --caloClNxM 5x9 --sparsity $spars #--train
#     python3 TauMinator_CE_ident.py --date 2023_07_04 --v 21p2 --caloClNxM 5x9 --sparsity $spars --pt_weighted #--train

# done

python3 TauMinator_CE_calib.py --date 2023_07_04 --v 21p1 --caloClNxM 5x9 --sparsity 0.0


# python3 TauMinator_CB_ident.py --date 2023_07_06 --v 22 --caloClNxM 5x9 --sparsity 0.0 #--train
# python3 TauMinator_CB_calib.py --date 2023_07_06 --v 22 --caloClNxM 5x9 --sparsity 0.0 --pt_weighted --train

# python3 TauMinator_CE_ident.py --date 2023_07_06 --v 22 --caloClNxM 5x9 --sparsity 0.0 --train
# python3 TauMinator_CE_calib.py --date 2023_07_06 --v 22 --caloClNxM 5x9 --sparsity 0.0 --pt_weighted --train


# python3 TauMinator_CB_ident.py --date 2023_07_06 --v 23 --caloClNxM 5x9 --sparsity 0.0 --train
# python3 TauMinator_CB_calib.py --date 2023_07_06 --v 23 --caloClNxM 5x9 --sparsity 0.0 --pt_weighted --train

# python3 TauMinator_CE_ident.py --date 2023_07_06 --v 23 --caloClNxM 5x9 --sparsity 0.0 --train
# python3 TauMinator_CE_calib.py --date 2023_07_06 --v 23 --caloClNxM 5x9 --sparsity 0.0 --pt_weighted --train




python3 TauMinator_CE_ident.py --date 2023_07_13 --v 24 --caloClNxM 5x9 --sparsity 0.0 --train
python3 TauMinator_CE_calib.py --date 2023_07_13 --v 24 --caloClNxM 5x9 --sparsity 0.0 --pt_weighted --train






# python3 TauMinator_CB_ident_QNTZD.py --date 2023_07_06 --v 22QNTZD --caloClNxM 5x9 --sparsity 0.10 #--train
# python3 TauMinator_CE_ident_QNTZD.py --date 2023_07_06 --v 22QNTZD --caloClNxM 5x9 --sparsity 0.10 #--train

# python3 TauMinator_CB_ident_QNTZD.py --date 2023_07_06 --v 22QNTZD --caloClNxM 5x9 --sparsity 0.25 #--train
# python3 TauMinator_CE_ident_QNTZD.py --date 2023_07_06 --v 22QNTZD --caloClNxM 5x9 --sparsity 0.25 #--train

# python3 TauMinator_CB_ident_QNTZD.py --date 2023_07_06 --v 22QNTZD --caloClNxM 5x9 --sparsity 0.50 #--train
# python3 TauMinator_CE_ident_QNTZD.py --date 2023_07_06 --v 22QNTZD --caloClNxM 5x9 --sparsity 0.50 #--train



# python3 TauMinator_CB_calib_QNTZD.py --date 2023_07_06 --v 22QNTZD --caloClNxM 5x9 --sparsity 0.3 --sparsityCNN 0.25 --pt_weighted --train
# python3 TauMinator_CE_calib_QNTZD.py --date 2023_07_06 --v 22QNTZD --caloClNxM 5x9 --sparsity 0.3 --sparsityCNN 0.25 --pt_weighted --train

# python3 TauMinator_CB_calib_QNTZD.py --date 2023_07_06 --v 22QNTZD --caloClNxM 5x9 --sparsity 0.4 --sparsityCNN 0.25 --pt_weighted --train
# python3 TauMinator_CE_calib_QNTZD.py --date 2023_07_06 --v 22QNTZD --caloClNxM 5x9 --sparsity 0.4 --sparsityCNN 0.25 --pt_weighted --train

# python3 TauMinator_CB_calib_QNTZD.py --date 2023_07_06 --v 22QNTZD --caloClNxM 5x9 --sparsity 0.5 --sparsityCNN 0.25 --pt_weighted --train
# python3 TauMinator_CE_calib_QNTZD.py --date 2023_07_06 --v 22QNTZD --caloClNxM 5x9 --sparsity 0.5 --sparsityCNN 0.25 --pt_weighted --train

# python3 TauMinator_CB_calib_QNTZD.py --date 2023_07_06 --v 22QNTZD --caloClNxM 5x9 --sparsity 0.6 --sparsityCNN 0.25 --pt_weighted --train
# python3 TauMinator_CE_calib_QNTZD.py --date 2023_07_06 --v 22QNTZD --caloClNxM 5x9 --sparsity 0.6 --sparsityCNN 0.25 --pt_weighted --train

# python3 TauMinator_CB_calib_QNTZD.py --date 2023_07_06 --v 22QNTZD --caloClNxM 5x9 --sparsity 0.7 --sparsityCNN 0.25 --pt_weighted --train
# python3 TauMinator_CE_calib_QNTZD.py --date 2023_07_06 --v 22QNTZD --caloClNxM 5x9 --sparsity 0.7 --sparsityCNN 0.25 --pt_weighted --train











# python3 TauMinator_CB_ident.py --date 2023_07_04 --v 21p1 --caloClNxM 5x9 --sparsity 0.5 --pt_weighted --train
# python3 TauMinator_CE_ident.py --date 2023_07_04 --v 21p1 --caloClNxM 5x9 --sparsity 0.3 --train


# python3 TauMinator_CB_ident_Q.py --date 2023_07_04 --v 21p1 --caloClNxM 5x9 --sparsity 0.5 --pt_weighted --train

# sparsity_values="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9"

# for spars in $sparsity_values
# do 
#     python3 TauMinator_CB_calib.py --date 2023_07_04 --v 21p1 --caloClNxM 5x9 --sparsity $spars --pt_weighted --train
#     python3 TauMinator_CE_calib.py --date 2023_07_04 --v 21p1 --caloClNxM 5x9 --sparsity $spars --pt_weighted --train
# done


# for spars in $sparsity_values
# do 
#     cp /data_CMS/cms/motta/Phase2L1T/2023_07_04_v21p1/TauMinator_CB_cltw5x9_Training_ptWeighted/TauMinator_CB_calib_sparsity${spars}_plots/responses_comparison.pdf /data_CMS/cms/motta/Phase2L1T/2023_07_04_v21p1/TauMinator_CB_cltw5x9_Training_ptWeighted/compare_plots/responses_comparison_${spars}.pdf
#     cp /data_CMS/cms/motta/Phase2L1T/2023_07_04_v21p1/TauMinator_CB_cltw5x9_Training_ptWeighted/TauMinator_CB_calib_sparsity${spars}_plots/scale_vs_pt.pdf /data_CMS/cms/motta/Phase2L1T/2023_07_04_v21p1/TauMinator_CB_cltw5x9_Training_ptWeighted/compare_plots/scale_vs_pt_${spars}.pdf

#     cp /data_CMS/cms/motta/Phase2L1T/2023_07_04_v21p1/TauMinator_CE_cltw5x9_Training_ptWeighted/TauMinator_CE_calib_sparsity${spars}_plots/responses_comparison.pdf /data_CMS/cms/motta/Phase2L1T/2023_07_04_v21p1/TauMinator_CE_cltw5x9_Training_ptWeighted/compare_plots/responses_comparison_${spars}.pdf
#     cp /data_CMS/cms/motta/Phase2L1T/2023_07_04_v21p1/TauMinator_CE_cltw5x9_Training_ptWeighted/TauMinator_CE_calib_sparsity${spars}_plots/scale_vs_pt.pdf /data_CMS/cms/motta/Phase2L1T/2023_07_04_v21p1/TauMinator_CE_cltw5x9_Training_ptWeighted/compare_plots/scale_vs_pt_${spars}.pdf
# done

# python3 TauMinator_CB_calib.py --date 2023_07_04 --v 21 --caloClNxM 5x9 --train
# python3 TauMinator_CB_calib.py --date 2023_07_04 --v 21 --caloClNxM 5x9 --train --pt_weighted

# python3 TauMinator_CE_calib.py --date 2023_07_04 --v 21 --caloClNxM 5x9 --train
# python3 TauMinator_CE_calib.py --date 2023_07_04 --v 21 --caloClNxM 5x9 --train --pt_weighted

# #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# python3 TauMinator_CB_calib.py --date 2023_07_04 --v 21p1 --caloClNxM 5x9 --train
# python3 TauMinator_CB_calib.py --date 2023_07_04 --v 21p1 --caloClNxM 5x9 --train --pt_weighted

# python3 TauMinator_CE_calib.py --date 2023_07_04 --v 21p1 --caloClNxM 5x9 --train
# python3 TauMinator_CE_calib.py --date 2023_07_04 --v 21p1 --caloClNxM 5x9 --train --pt_weighted

# #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# python3 TauMinator_CB_calib.py --date 2023_07_04 --v 21p2 --caloClNxM 5x9 --train
# python3 TauMinator_CB_calib.py --date 2023_07_04 --v 21p2 --caloClNxM 5x9 --train --pt_weighted

# python3 TauMinator_CE_calib.py --date 2023_07_04 --v 21p2 --caloClNxM 5x9 --train
# python3 TauMinator_CE_calib.py --date 2023_07_04 --v 21p2 --caloClNxM 5x9 --train --pt_weighted







