import FWCore.ParameterSet.Config as cms

L1CaloTauNtuplizerProducer = cms.EDAnalyzer("L1CaloTauNtuplizerProducer",
    TauMinatorTaus = cms.InputTag("l1tNNCaloTauProducer", "L1NNCaloTauCollectionBXV"),
    genTaus = cms.InputTag("GenHandler", "GenTausCollection"),
    genTausMod = cms.InputTag("GenHandlerMod", "GenTausCollection"),
    treeName = cms.string("L1TauMinatorTree"),
    DEBUG = cms.bool(False)
)