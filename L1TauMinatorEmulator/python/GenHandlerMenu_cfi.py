import FWCore.ParameterSet.Config as cms

GenHandlerMenu = cms.EDProducer("GenHandlerMenu",
    GenParticles = cms.InputTag('genParticles'),
    DEBUG = cms.bool(False)
)