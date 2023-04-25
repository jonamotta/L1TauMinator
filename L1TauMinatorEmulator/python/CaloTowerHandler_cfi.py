import FWCore.ParameterSet.Config as cms

CaloTowerHandler = cms.EDProducer("CaloTowerHandler",
    # l1CaloTowers = cms.InputTag("l1tTowerCalibrationProducer","L1CaloTowerCalibratedCollection"), # calibrated towers (same input as L1CaloJetProducer)
    l1CaloTowers = cms.InputTag("l1tEGammaClusterEmuProducer","L1CaloTowerCollection",""), # uncalibrated towers (same input as L1TowerCalibrator)
    hgcalTowers = cms.InputTag("l1tHGCalTowerProducer","HGCalTowerProcessor"),
    hcalDigis = cms.InputTag("simHcalTriggerPrimitiveDigis"),
    EcalEtMinForClustering = cms.double(0.),
    HcalEtMinForClustering = cms.double(0.),
    EtMinForSeeding = cms.double(2.5),
    DEBUG = cms.bool(False)
)