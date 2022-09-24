echo ""
echo "python3 CLTW_TauIdentifier.py --v 9 --date 2022_09_14 --inTag _lTauPtCut18 --caloClNxM 5x9 --train"
python3 CLTW_TauIdentifier.py --v 9 --date 2022_09_14 --inTag _lTauPtCut18 --caloClNxM 5x9 --train
echo "** DONE"
echo "*******************************************************************************************************************************************************************"
sleep 3

echo ""
echo "python3 CLTW_TauIdentifier_pruning.py --v 9 --date 2022_09_14 --inTag _lTauPtCut18 --caloClNxM 5x9 --sparsity 0.5 --train"
python3 CLTW_TauIdentifier_pruning.py --v 9 --date 2022_09_14 --inTag _lTauPtCut18 --caloClNxM 5x9 --sparsity 0.5 --train
echo "** DONE"
echo "*******************************************************************************************************************************************************************"
sleep 3

echo ""
echo "python3 CLTW_TauIdentifier_pruning.py --v 9 --date 2022_09_14 --inTag _lTauPtCut18 --caloClNxM 5x9 --sparsity 0.6 --train"
python3 CLTW_TauIdentifier_pruning.py --v 9 --date 2022_09_14 --inTag _lTauPtCut18 --caloClNxM 5x9 --sparsity 0.6 --train
echo "** DONE"
echo "*******************************************************************************************************************************************************************"
sleep 3

echo ""
echo "python3 CLTW_TauIdentifier_pruning.py --v 9 --date 2022_09_14 --inTag _lTauPtCut18 --caloClNxM 5x9 --sparsity 0.7 --train"
python3 CLTW_TauIdentifier_pruning.py --v 9 --date 2022_09_14 --inTag _lTauPtCut18 --caloClNxM 5x9 --sparsity 0.7 --train
echo "** DONE"
echo "*******************************************************************************************************************************************************************"
sleep 3

echo ""
echo "python3 CLTW_TauQIdentifier.py --v 9 --date 2022_09_14 --inTag _lTauPtCut18 --caloClNxM 5x9 --train"
python3 CLTW_TauQIdentifier.py --v 9 --date 2022_09_14 --inTag _lTauPtCut18 --caloClNxM 5x9 --train
echo "** DONE"
echo "*******************************************************************************************************************************************************************"
sleep 3

echo ""
echo "python3 CLTW_TauQIdentifier_pruning.py --v 9 --date 2022_09_14 --inTag _lTauPtCut18 --caloClNxM 5x9 --sparsity 0.5 --train"
python3 CLTW_TauQIdentifier_pruning.py --v 9 --date 2022_09_14 --inTag _lTauPtCut18 --caloClNxM 5x9 --sparsity 0.5 --train
echo "** DONE"
echo "*******************************************************************************************************************************************************************"
sleep 3

echo ""
echo "python3 CLTW_TauQIdentifier_pruning.py --v 9 --date 2022_09_14 --inTag _lTauPtCut18 --caloClNxM 5x9 --sparsity 0.6 --train"
python3 CLTW_TauQIdentifier_pruning.py --v 9 --date 2022_09_14 --inTag _lTauPtCut18 --caloClNxM 5x9 --sparsity 0.6 --train
echo "** DONE"
echo "*******************************************************************************************************************************************************************"
sleep 3

echo ""
echo "python3 CLTW_TauQIdentifier_pruning.py --v 9 --date 2022_09_14 --inTag _lTauPtCut18 --caloClNxM 5x9 --sparsity 0.7 --train"
python3 CLTW_TauQIdentifier_pruning.py --v 9 --date 2022_09_14 --inTag _lTauPtCut18 --caloClNxM 5x9 --sparsity 0.7 --train
echo "** DONE"
echo "*******************************************************************************************************************************************************************"
sleep 3





echo ""
echo "python3 CLTW_TauCalibrator.py --v 9 --date 2022_09_14 --inTag _lTauPtCut18_uEtacut1.5 --inTagCNN _lTauPtCut18 --caloClNxM 5x9 --train"
python3 CLTW_TauCalibrator.py --v 9 --date 2022_09_14 --inTag _lTauPtCut18_uEtacut1.5 --inTagCNN _lTauPtCut18 --caloClNxM 5x9 --train
echo "** DONE"
echo "*******************************************************************************************************************************************************************"
sleep 3

echo ""
echo "python3 CLTW_TauCalibrator_pruning.py --v 9 --date 2022_09_14 --inTag _lTauPtCut18_uEtacut1.5 --inTagCNN _lTauPtCut18 --caloClNxM 5x9 --sparsity 0.5 --train"
python3 CLTW_TauCalibrator_pruning.py --v 9 --date 2022_09_14 --inTag _lTauPtCut18_uEtacut1.5 --inTagCNN _lTauPtCut18 --caloClNxM 5x9 --sparsity 0.5 --train
echo "** DONE"
echo "*******************************************************************************************************************************************************************"
sleep 3

echo ""
echo "python3 CLTW_TauCalibrator_pruning.py --v 9 --date 2022_09_14 --inTag _lTauPtCut18_uEtacut1.5 --inTagCNN _lTauPtCut18 --caloClNxM 5x9 --sparsity 0.6 --train"
python3 CLTW_TauCalibrator_pruning.py --v 9 --date 2022_09_14 --inTag _lTauPtCut18_uEtacut1.5 --inTagCNN _lTauPtCut18 --caloClNxM 5x9 --sparsity 0.6 --train
echo "** DONE"
echo "*******************************************************************************************************************************************************************"
sleep 3

echo ""
echo "python3 CLTW_TauCalibrator_pruning.py --v 9 --date 2022_09_14 --inTag _lTauPtCut18_uEtacut1.5 --inTagCNN _lTauPtCut18 --caloClNxM 5x9 --sparsity 0.7 --train"
python3 CLTW_TauCalibrator_pruning.py --v 9 --date 2022_09_14 --inTag _lTauPtCut18_uEtacut1.5 --inTagCNN _lTauPtCut18 --caloClNxM 5x9 --sparsity 0.7 --train
echo "** DONE"
echo "*******************************************************************************************************************************************************************"
sleep 3

echo ""
echo "python3 CLTW_TauQCalibrator.py --v 9 --date 2022_09_14 --inTag _lTauPtCut18_uEtacut1.5 --inTagCNN _lTauPtCut18 --caloClNxM 5x9 --train"
python3 CLTW_TauQCalibrator.py --v 9 --date 2022_09_14 --inTag _lTauPtCut18_uEtacut1.5 --inTagCNN _lTauPtCut18 --caloClNxM 5x9 --train
echo "** DONE"
echo "*******************************************************************************************************************************************************************"
sleep 3

echo ""
echo "python3 CLTW_TauQCalibrator_pruning.py --v 9 --date 2022_09_14 --inTag _lTauPtCut18_uEtacut1.5 --inTagCNN _lTauPtCut18 --caloClNxM 5x9 --sparsity 0.5 --train"
python3 CLTW_TauQCalibrator_pruning.py --v 9 --date 2022_09_14 --inTag _lTauPtCut18_uEtacut1.5 --inTagCNN _lTauPtCut18 --caloClNxM 5x9 --sparsity 0.5 --train
echo "** DONE"
echo "*******************************************************************************************************************************************************************"
sleep 3

echo ""
echo "python3 CLTW_TauQCalibrator_pruning.py --v 9 --date 2022_09_14 --inTag _lTauPtCut18_uEtacut1.5 --inTagCNN _lTauPtCut18 --caloClNxM 5x9 --sparsity 0.6 --train"
python3 CLTW_TauQCalibrator_pruning.py --v 9 --date 2022_09_14 --inTag _lTauPtCut18_uEtacut1.5 --inTagCNN _lTauPtCut18 --caloClNxM 5x9 --sparsity 0.6 --train
echo "** DONE"
echo "*******************************************************************************************************************************************************************"
sleep 3

echo ""
echo "python3 CLTW_TauQCalibrator_pruning.py --v 9 --date 2022_09_14 --inTag _lTauPtCut18_uEtacut1.5 --inTagCNN _lTauPtCut18 --caloClNxM 5x9 --sparsity 0.7 --train"
python3 CLTW_TauQCalibrator_pruning.py --v 9 --date 2022_09_14 --inTag _lTauPtCut18_uEtacut1.5 --inTagCNN _lTauPtCut18 --caloClNxM 5x9 --sparsity 0.7 --train
echo "** DONE"
echo "*******************************************************************************************************************************************************************"
