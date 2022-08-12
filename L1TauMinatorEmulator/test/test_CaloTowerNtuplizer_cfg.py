import FWCore.ParameterSet.Config as cms

from Configuration.Eras.Era_Phase2C9_cff import Phase2C9

process = cms.Process('L1REPR',Phase2C9)

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
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.load('L1TauMinator.L1TauMinatorEmulator.CaloTowerHandler_cff')
process.load('L1Trigger.L1THGCal.hgcalTriggerPrimitives_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(2),
    output = cms.optional.untracked.allowed(cms.int32,cms.PSet)
)

# Input source
process.source = cms.Source("PoolSource",
    fileNames = cms.untracked.vstring(
        '/store/mc/Phase2HLTTDRWinter20DIGI/GluGluToHHTo2B2Tau_node_SM_14TeV-madgraph-pythia8_tuneCP5/GEN-SIM-DIGI-RAW/PU200_110X_mcRun4_realistic_v3-v2/240000/0195AA7A-2336-0344-850B-9F3F4FBF1D6E.root',
    ),
    secondaryFileNames = cms.untracked.vstring(),
    inputCommands = cms.untracked.vstring(
                          "keep *",
                          "drop l1tPFCandidates_*_*_*",
                          "drop *_*_*_RECO",
    )
)

# Output definition
process.out = cms.OutputModule("PoolOutputModule",
                               fileName = cms.untracked.string('test.root')
)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('L1TauMinatorEmulator'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('V0')
)

# GlobalTag Info
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, '123X_mcRun4_realistic_v3', '')

# Tasks and Sequences definitions
from L1Trigger.L1CaloTrigger.L1EGammaCrystalsEmulatorProducer_cfi import *
L1TowerProducer_task = cms.Task(L1EGammaClusterEmuProducer)

# Path and EndPath definitions
process.raw2digi_step = cms.Path(process.RawToDigi)
process.towerProd_step = cms.Path(L1TowerProducer_task)
process.hgcl1tpg_step = cms.Path(process.hgcalTriggerPrimitives)
process.caloTower_step = cms.Path(process.CaloTowerHandler_seq)
process.hgcalCluster_step = cms.Path(process.HGCalClusterHandler_seq)
process.endjob_step = cms.EndPath(process.endOfProcess)

# Schedule definition
process.schedule = cms.Schedule(process.raw2digi_step, process.towerProd_step, process.hgcl1tpg_step, process.caloTower_step, process.hgcalCluster_step, process.endjob_step)

# Customisation of the process.
from L1Trigger.Configuration.customisePhase2 import addHcalTriggerPrimitives 
process = addHcalTriggerPrimitives(process)

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)

