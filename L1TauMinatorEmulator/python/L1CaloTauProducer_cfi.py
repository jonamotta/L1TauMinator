import FWCore.ParameterSet.Config as cms

L1CaloTauProducer = cms.EDProducer("L1CaloTauProducer",
    l1CaloTowers = cms.InputTag("l1tEGammaClusterEmuProducer","L1CaloTowerCollection",""), # uncalibrated towers (same input as L1TowerCalibrator)
    hgcalTowers = cms.InputTag("l1tHGCalTowerProducer","HGCalTowerProcessor"),

    HgcalClusters = cms.InputTag("l1tHGCalBackEndLayer2Producer","HGCalBackendLayer2Processor3DClustering"),
    preEmId  = cms.string("hOverE < 0.3 && hOverE >= 0"),
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

    EcalEtMinForClustering = cms.double(0.),
    HcalEtMinForClustering = cms.double(0.),
    EtMinForSeeding = cms.double(2.5),
    
    CNNmodel_CB_path = cms.string("/home/llr/cms/motta/Phase2L1T/CMSSW_12_5_2_patch1/src/L1TauMinator/L1TauMinatorEmulator/data/CNNmodel_CB.pb"),
    DNNident_CB_path = cms.string("/home/llr/cms/motta/Phase2L1T/CMSSW_12_5_2_patch1/src/L1TauMinator/L1TauMinatorEmulator/data/DNNident_CB.pb"),
    # DNNcalib_CB_path = cms.string("/home/llr/cms/motta/Phase2L1T/CMSSW_12_5_2_patch1/src/L1TauMinator/L1TauMinatorEmulator/data/DNNcalib_CB.pb"),
    DNNcalib_CB_path = cms.string("/home/llr/cms/motta/Phase2L1T/CMSSW_12_5_2_patch1/src/L1TauMinator/L1TauMinatorEmulator/data/DNNcalib_CB_ptWeighted.pb"),

    CNNmodel_CE_path = cms.string("/home/llr/cms/motta/Phase2L1T/CMSSW_12_5_2_patch1/src/L1TauMinator/L1TauMinatorEmulator/data/CNNmodel_CE.pb"),
    DNNident_CE_path = cms.string("/home/llr/cms/motta/Phase2L1T/CMSSW_12_5_2_patch1/src/L1TauMinator/L1TauMinatorEmulator/data/DNNident_CE.pb"),
    # DNNcalib_CE_path = cms.string("/home/llr/cms/motta/Phase2L1T/CMSSW_12_5_2_patch1/src/L1TauMinator/L1TauMinatorEmulator/data/DNNcalib_CE.pb"),
    DNNcalib_CE_path = cms.string("/home/llr/cms/motta/Phase2L1T/CMSSW_12_5_2_patch1/src/L1TauMinator/L1TauMinatorEmulator/data/DNNcalib_CE_ptWeighted.pb"),
    
    DEBUG = cms.bool(False)
)
