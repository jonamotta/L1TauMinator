echo ""
echo "python3 CLTW_TauIdentifier.py --v 12 --date 2022_10_04 --inTag _lTauPtCut18 --caloClNxM 5x9 --train"
python3 CLTW_TauIdentifier.py --v 12 --date 2022_10_04 --inTag _lTauPtCut18 --caloClNxM 5x9 --train
echo "** DONE"
echo "*******************************************************************************************************************************************************************"
sleep 3

echo ""
echo "python3 CLTW_TauIdentifier_pruning.py --v 12 --date 2022_10_04 --inTag _lTauPtCut18 --caloClNxM 5x9 --sparsity 0.25 --train"
python3 CLTW_TauIdentifier_pruning.py --v 12 --date 2022_10_04 --inTag _lTauPtCut18 --caloClNxM 5x9 --sparsity 0.25 --train
echo "** DONE"
echo "*******************************************************************************************************************************************************************"
sleep 3

echo ""
echo "python3 CLTW_TauIdentifier_pruning.py --v 12 --date 2022_10_04 --inTag _lTauPtCut18 --caloClNxM 5x9 --sparsity 0.5 --train"
python3 CLTW_TauIdentifier_pruning.py --v 12 --date 2022_10_04 --inTag _lTauPtCut18 --caloClNxM 5x9 --sparsity 0.5 --train
echo "** DONE"
echo "*******************************************************************************************************************************************************************"
sleep 3

echo ""
echo "python3 CLTW_TauIdentifier_pruning.py --v 12 --date 2022_10_04 --inTag _lTauPtCut18 --caloClNxM 5x9 --sparsity 0.75 --train"
python3 CLTW_TauIdentifier_pruning.py --v 12 --date 2022_10_04 --inTag _lTauPtCut18 --caloClNxM 5x9 --sparsity 0.75 --train
echo "** DONE"
echo "*******************************************************************************************************************************************************************"
sleep 3

echo ""
echo "python3 CLTW_TauQIdentifier.py --v 12 --date 2022_10_04 --inTag _lTauPtCut18 --caloClNxM 5x9 --train"
python3 CLTW_TauQIdentifier.py --v 12 --date 2022_10_04 --inTag _lTauPtCut18 --caloClNxM 5x9 --train
echo "** DONE"
echo "*******************************************************************************************************************************************************************"
sleep 3

echo ""
echo "python3 CLTW_TauQIdentifier_pruning.py --v 12 --date 2022_10_04 --inTag _lTauPtCut18 --caloClNxM 5x9 --sparsity 0.25 --train"
python3 CLTW_TauQIdentifier_pruning.py --v 12 --date 2022_10_04 --inTag _lTauPtCut18 --caloClNxM 5x9 --sparsity 0.25 --train
echo "** DONE"
echo "*******************************************************************************************************************************************************************"
sleep 3

echo ""
echo "python3 CLTW_TauQIdentifier_pruning.py --v 12 --date 2022_10_04 --inTag _lTauPtCut18 --caloClNxM 5x9 --sparsity 0.5 --train"
python3 CLTW_TauQIdentifier_pruning.py --v 12 --date 2022_10_04 --inTag _lTauPtCut18 --caloClNxM 5x9 --sparsity 0.5 --train
echo "** DONE"
echo "*******************************************************************************************************************************************************************"
sleep 3

echo ""
echo "python3 CLTW_TauQIdentifier_pruning.py --v 12 --date 2022_10_04 --inTag _lTauPtCut18 --caloClNxM 5x9 --sparsity 0.75 --train"
python3 CLTW_TauQIdentifier_pruning.py --v 12 --date 2022_10_04 --inTag _lTauPtCut18 --caloClNxM 5x9 --sparsity 0.75 --train
echo "** DONE"
echo "*******************************************************************************************************************************************************************"
sleep 3






echo ""
echo "python3 CLTW_TauCalibrator.py --v 12 --date 2022_10_04 --inTag _lTauPtCut18_uEtacut1.5 --inTagCNN _lTauPtCut18 --caloClNxM 5x9 --train"
python3 CLTW_TauCalibrator.py --v 12 --date 2022_10_04 --inTag _lTauPtCut18_uEtacut1.5 --inTagCNN _lTauPtCut18 --caloClNxM 5x9 --train
echo "** DONE"
echo "*******************************************************************************************************************************************************************"
sleep 3

echo ""
echo "python3 CLTW_TauCalibrator_pruning.py --v 12 --date 2022_10_04 --inTag _lTauPtCut18_uEtacut1.5 --inTagCNN _lTauPtCut18 --caloClNxM 5x9 --sparsity 0.25 --train"
python3 CLTW_TauCalibrator_pruning.py --v 12 --date 2022_10_04 --inTag _lTauPtCut18_uEtacut1.5 --inTagCNN _lTauPtCut18 --caloClNxM 5x9 --sparsity 0.25 --train
echo "** DONE"
echo "*******************************************************************************************************************************************************************"
sleep 3

echo ""
echo "python3 CLTW_TauCalibrator_pruning.py --v 12 --date 2022_10_04 --inTag _lTauPtCut18_uEtacut1.5 --inTagCNN _lTauPtCut18 --caloClNxM 5x9 --sparsity 0.5 --train"
python3 CLTW_TauCalibrator_pruning.py --v 12 --date 2022_10_04 --inTag _lTauPtCut18_uEtacut1.5 --inTagCNN _lTauPtCut18 --caloClNxM 5x9 --sparsity 0.5 --train
echo "** DONE"
echo "*******************************************************************************************************************************************************************"
sleep 3

echo ""
echo "python3 CLTW_TauCalibrator_pruning.py --v 12 --date 2022_10_04 --inTag _lTauPtCut18_uEtacut1.5 --inTagCNN _lTauPtCut18 --caloClNxM 5x9 --sparsity 0.75 --train"
python3 CLTW_TauCalibrator_pruning.py --v 12 --date 2022_10_04 --inTag _lTauPtCut18_uEtacut1.5 --inTagCNN _lTauPtCut18 --caloClNxM 5x9 --sparsity 0.75 --train
echo "** DONE"
echo "*******************************************************************************************************************************************************************"
sleep 3

echo ""
echo "python3 CLTW_TauQCalibrator.py --v 12 --date 2022_10_04 --inTag _lTauPtCut18_uEtacut1.5 --inTagCNN _lTauPtCut18 --caloClNxM 5x9 --train"
python3 CLTW_TauQCalibrator.py --v 12 --date 2022_10_04 --inTag _lTauPtCut18_uEtacut1.5 --inTagCNN _lTauPtCut18 --caloClNxM 5x9 --train
echo "** DONE"
echo "*******************************************************************************************************************************************************************"
sleep 3

echo ""
echo "python3 CLTW_TauQCalibrator_pruning.py --v 12 --date 2022_10_04 --inTag _lTauPtCut18_uEtacut1.5 --inTagCNN _lTauPtCut18 --caloClNxM 5x9 --sparsity 0.25 --train"
python3 CLTW_TauQCalibrator_pruning.py --v 12 --date 2022_10_04 --inTag _lTauPtCut18_uEtacut1.5 --inTagCNN _lTauPtCut18 --caloClNxM 5x9 --sparsity 0.25 --train
echo "** DONE"
echo "*******************************************************************************************************************************************************************"
sleep 3

echo ""
echo "python3 CLTW_TauQCalibrator_pruning.py --v 12 --date 2022_10_04 --inTag _lTauPtCut18_uEtacut1.5 --inTagCNN _lTauPtCut18 --caloClNxM 5x9 --sparsity 0.5 --train"
python3 CLTW_TauQCalibrator_pruning.py --v 12 --date 2022_10_04 --inTag _lTauPtCut18_uEtacut1.5 --inTagCNN _lTauPtCut18 --caloClNxM 5x9 --sparsity 0.5 --train
echo "** DONE"
echo "*******************************************************************************************************************************************************************"
sleep 3

echo ""
echo "python3 CLTW_TauQCalibrator_pruning.py --v 12 --date 2022_10_04 --inTag _lTauPtCut18_uEtacut1.5 --inTagCNN _lTauPtCut18 --caloClNxM 5x9 --sparsity 0.75 --train"
python3 CLTW_TauQCalibrator_pruning.py --v 12 --date 2022_10_04 --inTag _lTauPtCut18_uEtacut1.5 --inTagCNN _lTauPtCut18 --caloClNxM 5x9 --sparsity 0.75 --train
echo "** DONE"
echo "*******************************************************************************************************************************************************************"






echo ""
echo "python3 CL3D_TauIdentifier.py --v 12 --date 2022_10_04 --inTag _lTauPtCut18_lEtacut1.5 --train"
python3 CL3D_TauIdentifier.py --v 12 --date 2022_10_04 --inTag _lTauPtCut18_lEtacut1.5 --train
echo "** DONE"
echo "*******************************************************************************************************************************************************************"
sleep 3

echo ""
echo "python3 CL3D_TauCalibrator.py --v 12 --date 2022_10_04 --inTag _lTauPtCut18_lEtacut1.5 --train"
python3 CL3D_TauCalibrator.py --v 12 --date 2022_10_04 --inTag _lTauPtCut18_lEtacut1.5 --train
echo "** DONE"
echo "*******************************************************************************************************************************************************************"
sleep 3






echo ""
echo "python3 CL3D_TauIdentifier_DNN.py --v 12 --date 2022_10_04 --inTag _lTauPtCut18_lEtacut1.5 --train"
python3 CL3D_TauIdentifier_DNN.py --v 12 --date 2022_10_04 --inTag _lTauPtCut18_lEtacut1.5 --train
echo "** DONE"
echo "*******************************************************************************************************************************************************************"
sleep 3

echo ""
echo "python3 CL3D_TauCalibrator_DNN.py --v 12 --date 2022_10_04 --inTag _lTauPtCut18_lEtacut1.5 --train"
python3 CL3D_TauCalibrator_DNN.py --v 12 --date 2022_10_04 --inTag _lTauPtCut18_lEtacut1.5 --train
echo "** DONE"
echo "*******************************************************************************************************************************************************************"
sleep 3






echo ""
echo "python3 CL3D_TauIdentifier_QDNN.py --v 12 --date 2022_10_04 --inTag _lTauPtCut18_lEtacut1.5 --train"
python3 CL3D_TauIdentifier_QDNN.py --v 12 --date 2022_10_04 --inTag _lTauPtCut18_lEtacut1.5 --train
echo "** DONE"
echo "*******************************************************************************************************************************************************************"
sleep 3

echo ""
echo "python3 CL3D_TauCalibrator_QDNN.py --v 12 --date 2022_10_04 --inTag _lTauPtCut18_lEtacut1.5 --train"
python3 CL3D_TauCalibrator_QDNN.py --v 12 --date 2022_10_04 --inTag _lTauPtCut18_lEtacut1.5 --train
echo "** DONE"
echo "*******************************************************************************************************************************************************************"
sleep 3

