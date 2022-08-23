import FWCore.ParameterSet.Config as cms

Ntuplizer = cms.EDAnalyzer("Ntuplizer",
    CaloClusters9x9 = cms.InputTag("CaloTowerHandler", "l1TowerClusters9x9"),
    CaloClusters7x7 = cms.InputTag("CaloTowerHandler", "l1TowerClusters7x7"),
    CaloClusters5x5 = cms.InputTag("CaloTowerHandler", "l1TowerClusters5x5"),
    CaloClusters5x9 = cms.InputTag("CaloTowerHandler", "l1TowerClusters5x9"),
    CaloClusters5x7 = cms.InputTag("CaloTowerHandler", "l1TowerClusters5x7"),
    CaloClusters3x7 = cms.InputTag("CaloTowerHandler", "l1TowerClusters3x7"),
    CaloClusters3x5 = cms.InputTag("CaloTowerHandler", "l1TowerClusters3x5"),
    HGClusters = cms.InputTag("HGClusterHandler", "HGClustersCollection"),
    genTaus = cms.InputTag("GenHandler", "GenTausCollection"),
    genJets = cms.InputTag("GenHandler", "GenJetsCollection"),
    treeName = cms.string("L1TauMinatorTree"),
    DEBUG = cms.bool(False)
)