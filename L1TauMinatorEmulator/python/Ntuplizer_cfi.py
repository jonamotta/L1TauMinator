import FWCore.ParameterSet.Config as cms

Ntuplizer = cms.EDAnalyzer("Ntuplizer",
    CaloClusters9x9 = cms.InputTag("CaloTowerHandler", "l1TowerClusters9x9"),
    CaloClusters7x7 = cms.InputTag("CaloTowerHandler", "l1TowerClusters7x7"),
    CaloClusters5x5 = cms.InputTag("CaloTowerHandler", "l1TowerClusters5x5"),
    CaloClusters5x9 = cms.InputTag("CaloTowerHandler", "l1TowerClusters5x9"),
    HGClusters = cms.InputTag("HGClusterHandler", "HGClustersCollection"),
    genTaus = cms.InputTag("GenHandler", "GenTauCollection"),
    genJets = cms.InputTag("GenHandler", "GenJetCollection"),
    treeName = cms.string("L1TauMinatorTree"),
    DEBUG = cms.bool(True)
)