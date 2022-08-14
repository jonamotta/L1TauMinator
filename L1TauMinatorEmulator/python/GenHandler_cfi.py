import FWCore.ParameterSet.Config as cms

GenHandler = cms.EDProducer("GenHandler",
    GenParticles = cms.InputTag('genParticles'),
    GenJets = cms.InputTag('ak4GenJetsNoNu'),
    DEBUG = cms.bool(False)
)