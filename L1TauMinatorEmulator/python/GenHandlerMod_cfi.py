import FWCore.ParameterSet.Config as cms

GenHandlerMod = cms.EDProducer("GenHandlerMod",
    GenParticles = cms.InputTag('genParticles'),
    DEBUG = cms.bool(False)
)