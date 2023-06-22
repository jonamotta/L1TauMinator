import FWCore.ParameterSet.Config as cms

L1CaloTauNtuplizerProducer = cms.EDAnalyzer("L1CaloTauNtuplizerProducerTest",
    TauMinatorTaus = cms.InputTag("l1tNNCaloTauProducer", "L1NNCaloTauCollectionBXV"),
    genTaus = cms.InputTag("GenHandler", "GenTausCollection"),
    genTausMod = cms.InputTag("GenHandlerMenu", "GenTausCollection"),
    treeName = cms.string("L1TauMinatorTree"),
    DEBUG = cms.bool(False)
)