import FWCore.ParameterSet.Config as cms

CaloTowerHandler = cms.EDProducer("CaloTowerHandler",
    l1CaloTowers = cms.InputTag("L1EGammaClusterEmuProducer","L1CaloTowerCollection",""), # uncalibrated towers (same input as L1CaloJetProducer)
    hgcalTowers = cms.InputTag("hgcalTowerProducer","HGCalTowerProcessor"),
    hcalDigis = cms.InputTag("simHcalTriggerPrimitiveDigis"),
    EcalEtMinForClustering = cms.double(0.),
    HcalEtMinForClustering = cms.double(0.),
    EtMinForSeeding = cms.double(2.5),
    DEBUG = cms.bool(False)
)