import FWCore.ParameterSet.VarParsing as VarParsing
import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C9_cff import Phase2C9

process = cms.Process('L1TauMinatorNtuplizer',Phase2C9)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mixNoPU_cfi')
process.load('Configuration.Geometry.GeometryExtended2026D49Reco_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedHLLHC14TeV_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.DigiToRaw_cff')
process.load('Configuration.StandardSequences.RawToDigi_cff')
# process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

# re-emulation of calo tps
# process.load('L1Trigger.L1CaloTrigger.l1tEGammaCrystalsEmulatorProducer_cfi')
# process.load('L1Trigger.L1CaloTrigger.l1tTowerCalibrationProducer_cfi')
# process.load('L1Trigger.L1THGCal.hgcalTriggerPrimitives_cff')

process.load('L1TauMinator.L1TauMinatorEmulator.CaloTowerHandler_cff')
process.load('L1TauMinator.L1TauMinatorEmulator.HGClusterHandler_cff')
process.load('L1TauMinator.L1TauMinatorEmulator.GenHandler_cff')
process.load('L1TauMinator.L1TauMinatorEmulator.Ntuplizer_cff')

options = VarParsing.VarParsing ('analysis')
options.outputFile = 'Ntuple_L1TauMinator.root'
options.inputFiles = []
options.maxEvents  = -999
options.parseArguments()

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1),
    output = cms.optional.untracked.allowed(cms.int32,cms.PSet)
)
if options.maxEvents >= -1:
    process.maxEvents.input = cms.untracked.int32(options.maxEvents)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/mc/Phase2HLTTDRSummer20ReRECOMiniAOD/VBFHToTauTau_M125_14TeV_powheg_pythia8_correctedGridpack_tuneCP5/FEVT/PU200_111X_mcRun4_realistic_T15_v1-v1/280000/7771879B-1E79-E44A-897D-F27D32F0F555.root',
    ),
    secondaryFileNames = cms.untracked.vstring(),
    inputCommands = cms.untracked.vstring(
                          "keep *",
                          "drop l1tPFCandidates_*_*_*",
                          "drop l1tPFCandidatesl1tRegionalOutput_*_*_*",
                          "drop l1tEtSums_*_*_*",
                          "drop l1tHPSPFTaus_*_*_*",
                          "drop l1tPFJets_*_*_*",
                          "drop l1tPFTaus_*_*_*",
                          "drop recoCaloJets_*_*_*",
                          "drop *_*_*_RECO",
    ),

    # eventsToProcess = cms.untracked.VEventRange("1:60844"),
    # eventsToSkip = cms.untracked.VEventRange('1:224002-1:224019', '1:224021-1:max')
)
if options.inputFiles:
    process.source.fileNames = cms.untracked.vstring(options.inputFiles)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('L1TauMinatorNtuplization'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('V0')
)

# GlobalTag Info
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '125X_mcRun4_realistic_v2', '')

# Path and EndPath definitions
process.raw2digi_path     = cms.Path(process.RawToDigi)
# process.caloTpg_path      = cms.Path(process.l1tEGammaClusterEmuProducer) # re-emulation of calo tps
# process.hgcTpg_path       = cms.Path(process.L1THGCalTriggerPrimitives)   # re-emulation of calo tps
# process.caloTpgCalib_path = cms.Path(process.l1tTowerCalibrationProducer) # re-emulation of calo tps calibration
process.towerCluster_path = cms.Path(process.CaloTowerHandler_seq)
process.hgcalCluster_path = cms.Path(process.HGClusterHandler_seq)
process.generator_path    = cms.Path(process.GenHandler_seq)
process.ntuplizer_path    = cms.Path(process.Ntuplizer_seq)
# process.endjob_path       = cms.EndPath(process.endOfProcess)

# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_path, 
                                # process.caloTpg_path,      # re-emulation of calo tps
                                # process.hgcTpg_path,       # re-emulation of calo tps
                                # process.caloTpgCalib_path, # re-emulation of calo tps calibration
                                process.towerCluster_path,
                                process.hgcalCluster_path,
                                process.generator_path,
                                process.ntuplizer_path)
                                # process.endjob_path)

# Customisation of the process.
from L1Trigger.Configuration.customisePhase2 import addHcalTriggerPrimitives 
process = addHcalTriggerPrimitives(process)

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)

# Adding output file
process.TFileService=cms.Service('TFileService',fileName=cms.string(options.outputFile))