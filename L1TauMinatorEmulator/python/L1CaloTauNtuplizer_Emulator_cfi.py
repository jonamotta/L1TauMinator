import FWCore.ParameterSet.Config as cms

L1CaloTauNtuplizerEmulator = cms.EDAnalyzer("L1CaloTauNtuplizerEmulator",
    TauMinatorTaus = cms.InputTag("l1tNNCaloTauEmulator", "L1NNCaloTauCollectionBXV"),
    genTaus = cms.InputTag("GenHandler", "GenTausCollection"),
    genTausMod = cms.InputTag("GenHandlerMod", "GenTausCollection"),
    treeName = cms.string("L1TauMinatorTree"),
    DEBUG = cms.bool(False)
)