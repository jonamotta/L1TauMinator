import FWCore.ParameterSet.Config as cms

L1CaloTauNtuplizer = cms.EDAnalyzer("L1CaloTauNtuplizer",
    CaloClustersNxM = cms.InputTag("L1CaloTauProducer", "l1TowerClustersNxM"),
    HGClusters = cms.InputTag("L1CaloTauProducer", "HGClustersCollection"),
    minatedTaus = cms.InputTag("L1CaloTauProducer", "TausCollection"),
    genTaus = cms.InputTag("GenHandler", "GenTausCollection"),
    treeName = cms.string("L1TauMinatorTree"),
    etaClusterDimension = cms.int32(5),
    phiClusterDimension = cms.int32(9),
    DEBUG = cms.bool(False)
)