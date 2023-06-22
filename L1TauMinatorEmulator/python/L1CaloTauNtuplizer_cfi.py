import FWCore.ParameterSet.Config as cms

L1CaloTauNtuplizer = cms.EDAnalyzer("L1CaloTauNtuplizer",
    l1TowerClustersNxMCB = cms.InputTag("L1CaloTauProducer", "l1TowerClustersNxMCB"),
    l1TowerClustersNxMCE = cms.InputTag("L1CaloTauProducer", "l1TowerClustersNxMCE"),
    HGClusters = cms.InputTag("L1CaloTauProducer", "HGClustersCollection"),
    TauMinatorTaus = cms.InputTag("L1CaloTauProducer", "TauMinatorTausCollection"),
    squareTaus = cms.InputTag("l1tCaloJetProducer", "L1CaloTauCollectionBXV"),
    genTaus = cms.InputTag("GenHandlerMenu", "GenTausCollection"),
    genBJets = cms.InputTag("GenHandler", "GenBJetsCollection"),
    treeName = cms.string("L1TauMinatorTree"),
    DEBUG = cms.bool(False)
)