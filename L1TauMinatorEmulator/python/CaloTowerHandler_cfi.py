import FWCore.ParameterSet.Config as cms

CaloTowerHandler = cms.EDProducer("CaloTowerHandler",
    # l1CaloTowers = cms.InputTag("L1TowerCalibrationProducer","L1CaloTowerCalibratedCollection"), # calibrated towers
    l1CaloTowers = cms.InputTag("L1EGammaClusterEmuProducer","L1CaloTowerCollection",""), # uncalibrated towers
    HgcalTowers = cms.InputTag("hgcalTowerProducer","HGCalTowerProcessor"),
    EcalEtMinForClustering = cms.double(0.),
    HcalEtMinForClustering = cms.double(0.),
    EtMinForSeeding = cms.double(2.5),
    DEBUG = cms.bool(True)
)