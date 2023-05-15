import FWCore.ParameterSet.Config as cms

HGClusterHandler = cms.EDProducer("HGClusterHandler",
    HgcalClusters = cms.InputTag("l1tHGCalBackEndLayer2Producer","HGCalBackendLayer2Processor3DClustering"),
    preEmId  = cms.string("hOverE < 0.3 && hOverE >= 0"),
    VsPionId = cms.PSet(
        isPUFilter = cms.bool(False),
        preselection = cms.string(""),
        method = cms.string("BDT"), # "" to be disabled, "BDT" to be enabled
        variables = cms.VPSet(
            cms.PSet(name = cms.string("fabs(eta)"), value = cms.string("abs(eta())")),
            cms.PSet(name = cms.string("eMax"), value = cms.string("eMax()")),
            cms.PSet(name = cms.string("sigmaPhiPhiTot"), value = cms.string("sigmaPhiPhiTot()")),
            cms.PSet(name = cms.string("sigmaZZ"), value = cms.string("sigmaZZ()")),
            cms.PSet(name = cms.string("layer50percent"), value = cms.string("layer50percent()")),
            cms.PSet(name = cms.string("triggerCells67percent"), value = cms.string("triggerCells67percent()")),
        ),
        weightsFile = cms.string("L1Trigger/Phase2L1ParticleFlow/data/hgcal_egID/Photon_vs_Pion_BDTweights_1116.xml.gz"),
        wp = cms.string("-0.10")
    ),
    VsPuId = cms.PSet(
        isPUFilter = cms.bool(True),
        preselection = cms.string(""),
        method = cms.string("BDT"), # "" to be disabled, "BDT" to be enabled
        variables = cms.VPSet(
            cms.PSet(name = cms.string("eMax"), value = cms.string("eMax()")),
            cms.PSet(name = cms.string("eMaxOverE"), value = cms.string("eMax()/energy()")),
            cms.PSet(name = cms.string("sigmaPhiPhiTot"), value = cms.string("sigmaPhiPhiTot()")),
            cms.PSet(name = cms.string("sigmaRRTot"), value = cms.string("sigmaRRTot()")),
            cms.PSet(name = cms.string("triggerCells90percent"), value = cms.string("triggerCells90percent()")),
        ),
        weightsFile = cms.string("L1Trigger/Phase2L1ParticleFlow/data/hgcal_egID/Photon_Pion_vs_Neutrino_BDTweights_1116.xml.gz"),
        wp = cms.string("-0.10")
    ),
    DEBUG = cms.bool(False)
)