import FWCore.ParameterSet.Config as cms

L1CaloTauProducer = cms.EDProducer("L1CaloTauProducer",
    l1CaloTowers = cms.InputTag("L1EGammaClusterEmuProducer","L1CaloTowerCollection",""),
    hgcalTowers = cms.InputTag("hgcalTowerProducer","HGCalTowerProcessor"),
    HgcalClusters=cms.InputTag("hgcalBackEndLayer2Producer","HGCalBackendLayer2Processor3DClustering"),
    hcalDigis = cms.InputTag("simHcalTriggerPrimitiveDigis"),
    etaClusterDimension = cms.int32(5),
    phiClusterDimension = cms.int32(9),
    CNNfilters = cms.int32(3),
    EcalEtMinForClustering = cms.double(0.),
    HcalEtMinForClustering = cms.double(0.),
    EtMinForSeeding = cms.double(2.5),
    CLTW_CNNmodel_path = cms.string("/home/llr/cms/motta/Phase2L1T/CMSSW_12_3_0_pre4/src/L1TauMinator/L1TauMinatorEmulator/data/CLTW_CNNmodel.pb"),
    CLTW_DNNident_path = cms.string("/home/llr/cms/motta/Phase2L1T/CMSSW_12_3_0_pre4/src/L1TauMinator/L1TauMinatorEmulator/data/CLTW_DNNident.pb"),
    CLTW_DNNcalib_path = cms.string("/home/llr/cms/motta/Phase2L1T/CMSSW_12_3_0_pre4/src/L1TauMinator/L1TauMinatorEmulator/data/CLTW_DNNcalib.pb"),
    # XGBident_path = cms.string("/home/llr/cms/motta/Phase2L1T/CMSSW_12_3_0_pre4/src/L1TauMinator/L1TauMinatorEmulator/data/XGBident.model"),
    # XGBcalib_path = cms.string("/home/llr/cms/motta/Phase2L1T/CMSSW_12_3_0_pre4/src/L1TauMinator/L1TauMinatorEmulator/data/XGBcalib.model"),
    # XGBident_feats = cms.vstring("cl3d_pt", "cl3d_coreshowerlength", "cl3d_srrtot", "cl3d_srrmean", "cl3d_hoe", "cl3d_meanz"),
    # XGBcalib_feats = cms.vstring("cl3d_showerlength", "cl3d_coreshowerlength", "cl3d_abseta", "cl3d_spptot", "cl3d_srrmean", "cl3d_meanz"),
    # C1calib_params = cms.vdouble(-5.451686, 26.783195),
    # C3calib_params = cms.vdouble(51.069366, -41.85, 13.090333, -1.8184333, 0.09459969),
    CL3D_DNNident_path = cms.string("/home/llr/cms/motta/Phase2L1T/CMSSW_12_3_0_pre4/src/L1TauMinator/L1TauMinatorEmulator/data/CL3D_DNNident.pb"),
    CL3D_DNNcalib_path = cms.string("/home/llr/cms/motta/Phase2L1T/CMSSW_12_3_0_pre4/src/L1TauMinator/L1TauMinatorEmulator/data/CL3D_DNNcalib.pb"),
    CL3D_DNNident_feats = cms.vstring("cl3d_localAbsEta", "cl3d_showerlength", "cl3d_coreshowerlength", "cl3d_firstlayer", "cl3d_seetot", "cl3d_szz", "cl3d_srrtot", "cl3d_srrmean", "cl3d_hoe", "cl3d_localAbsMeanZ"),
    CL3D_DNNcalib_feats = cms.vstring("cl3d_pt", "cl3d_localAbsEta", "cl3d_showerlength", "cl3d_coreshowerlength", "cl3d_firstlayer", "cl3d_seetot", "cl3d_szz", "cl3d_srrtot", "cl3d_srrmean", "cl3d_hoe", "cl3d_localAbsMeanZ"),
    DEBUG = cms.bool(False)
)