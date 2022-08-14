import FWCore.ParameterSet.Config as cms

HGClusterHandler = cms.EDProducer("HGClusterHandler",
    HgcalClusters=cms.InputTag("hgcalBackEndLayer2Producer","HGCalBackendLayer2Processor3DClustering"),
    DEBUG = cms.bool(False)
)