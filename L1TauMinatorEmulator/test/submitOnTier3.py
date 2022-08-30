import os

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i+n]

def splitInBlocks (l, n):
    """split the list l in n blocks of equal size"""
    k = len(l) / n
    r = len(l) % n

    i = 0
    blocks = []
    while i < len(l):
        if len(blocks)<r:
            blocks.append(l[i:i+k+1])
            i += k+1
        else:
            blocks.append(l[i:i+k])
            i += k

    return blocks

##################################################################

version = "1"

filedir="/home/llr/cms/motta/Phase2L1T/CMSSW_12_3_0_pre4/src/L1TauMinator/L1TauMinatorEmulator/inputFiles/"

list_filelists = []
list_folders = []
list_njobs = []

# list_filelists.append(open(filedir+"test.txt"))
# list_folders.append("/data_CMS/cms/motta/Phase2L1T/L1TauMinatorNtuples/v"+version+"/test/")
# list_njobs.append(10)

# list_filelists.append(open(filedir+"VBFHToTauTau_M125_TuneCUETP8M1_14TeV_powheg_pythia8__Phase2HLTTDRWinter20DIGI-NoPU_110X_mcRun4_realistic_v3-v1__GEN-SIM-DIGI-RAW.txt"))
# list_folders.append("/data_CMS/cms/motta/Phase2L1T/L1TauMinatorNtuples/v"+version+"/VBFHToTauTau_M125_TuneCUETP8M1_14TeV_powheg_pythia8__Phase2HLTTDRWinter20DIGI-NoPU_110X_mcRun4_realistic_v3-v1__GEN-SIM-DIGI-RAW/")
# list_njobs.append(60)

# list_filelists.append(open(filedir+"VBFHToTauTau_M125_14TeV_powheg_pythia8_correctedGridpack_tuneCP5__Phase2HLTTDRWinter20DIGI-PU200_110X_mcRun4_realistic_v3-v3__GEN-SIM-DIGI-RAW.txt"))
# list_folders.append("/data_CMS/cms/motta/Phase2L1T/L1TauMinatorNtuples/v"+version+"/VBFHToTauTau_M125_14TeV_powheg_pythia8_correctedGridpack_tuneCP5__Phase2HLTTDRWinter20DIGI-PU200_110X_mcRun4_realistic_v3-v3__GEN-SIM-DIGI-RAW/")
# list_njobs.append(240)

# list_filelists.append(open(filedir+"GluGluToHHTo2B2Tau_node_SM_14TeV-madgraph-pythia8_tuneCP5__Phase2HLTTDRWinter20DIGI-PU200_110X_mcRun4_realistic_v3-v2__GEN-SIM-DIGI-RAW.txt"))
# list_folders.append("/data_CMS/cms/motta/Phase2L1T/L1TauMinatorNtuples/v"+version+"/GluGluToHHTo2B2Tau_node_SM_14TeV-madgraph-pythia8_tuneCP5__Phase2HLTTDRWinter20DIGI-PU200_110X_mcRun4_realistic_v3-v2__GEN-SIM-DIGI-RAW/")
# list_njobs.append(25)

# list_filelists.append(open(filedir+"QCD_Pt-15to3000_TuneCP5_Flat_14TeV-pythia8__Phase2HLTTDRWinter20DIGI-PU200_castor_110X_mcRun4_realistic_v3-v2__GEN-SIM-DIGI-RAW.txt"))
# list_folders.append("/data_CMS/cms/motta/Phase2L1T/L1TauMinatorNtuples/v"+version+"/QCD_Pt-15to3000_TuneCP5_Flat_14TeV-pythia8__Phase2HLTTDRWinter20DIGI-PU200_castor_110X_mcRun4_realistic_v3-v2__GEN-SIM-DIGI-RAW/")
# list_njobs.append(120)

# list_filelists.append(open(filedir+"MinBias_TuneCP5_14TeV-pythia8__Phase2HLTTDRWinter20DIGI-PU200_110X_mcRun4_realistic_v3-v3__GEN-SIM-DIGI-RAW.txt"))
# list_folders.append("/data_CMS/cms/motta/Phase2L1T/L1TauMinatorNtuples/v"+version+"/MinBias_TuneCP5_14TeV-pythia8__Phase2HLTTDRWinter20DIGI-PU200_110X_mcRun4_realistic_v3-v3__GEN-SIM-DIGI-RAW/")
# list_njobs.append(350)

# list_filelists.append(open(filedir+"VBFHToTauTau_M125_TuneCUETP8M1_14TeV_powheg_pythia8__Phase2HLTTDRSummer20ReRECOMiniAOD-NoPU_111X_mcRun4_realistic_T15_v1-v1__FEVT.txt"))
# list_folders.append("/data_CMS/cms/motta/Phase2L1T/L1TauMinatorNtuples/v"+version+"/VBFHToTauTau_M125_TuneCUETP8M1_14TeV_powheg_pythia8__Phase2HLTTDRSummer20ReRECOMiniAOD-NoPU_111X_mcRun4_realistic_T15_v1-v1__FEVT/")
# list_njobs.append(25)

# list_filelists.append(open(filedir+"GluGluHToTauTau_M125_14TeV_powheg_pythia8_TuneCP5__Phase2HLTTDRSummer20ReRECOMiniAOD-NoPU_111X_mcRun4_realistic_T15_v1-v1__GEN-SIM-DIGI-RAW-MINIAOD.txt"))
# list_folders.append("/data_CMS/cms/motta/Phase2L1T/L1TauMinatorNtuples/v"+version+"/GluGluHToTauTau_M125_14TeV_powheg_pythia8_TuneCP5__Phase2HLTTDRSummer20ReRECOMiniAOD-NoPU_111X_mcRun4_realistic_T15_v1-v1__GEN-SIM-DIGI-RAW-MINIAOD/")
# list_njobs.append(8)

# list_filelists.append(open(filedir+"VBFHToTauTau_M125_14TeV_powheg_pythia8_correctedGridpack_tuneCP5__Phase2HLTTDRSummer20ReRECOMiniAOD-PU200_111X_mcRun4_realistic_T15_v1-v1__FEVT.txt"))
# list_folders.append("/data_CMS/cms/motta/Phase2L1T/L1TauMinatorNtuples/v"+version+"/VBFHToTauTau_M125_14TeV_powheg_pythia8_correctedGridpack_tuneCP5__Phase2HLTTDRSummer20ReRECOMiniAOD-PU200_111X_mcRun4_realistic_T15_v1-v1__FEVT/")
# list_njobs.append(1057)

# list_filelists.append(open(filedir+"GluGluToHHTo2B2Tau_node_SM_14TeV-madgraph-pythia8_tuneCP5__Phase2HLTTDRSummer20ReRECOMiniAOD-PU200_111X_mcRun4_realistic_T15_v1-v1__GEN-SIM-DIGI-RAW-MINIAOD.txt"))
# list_folders.append("/data_CMS/cms/motta/Phase2L1T/L1TauMinatorNtuples/v"+version+"/GluGluToHHTo2B2Tau_node_SM_14TeV-madgraph-pythia8_tuneCP5__Phase2HLTTDRSummer20ReRECOMiniAOD-PU200_111X_mcRun4_realistic_T15_v1-v1__GEN-SIM-DIGI-RAW-MINIAOD/")
# list_njobs.append(49)

# list_filelists.append(open(filedir+"ZprimeToTauTau_M-500_TuneCP5_14TeV-pythia8-tauola__Phase2HLTTDRSummer20ReRECOMiniAOD-PU200_111X_mcRun4_realistic_T15_v1-v1__GEN-SIM-DIGI-RAW-MINIAOD.txt"))
# list_folders.append("/data_CMS/cms/motta/Phase2L1T/L1TauMinatorNtuples/v"+version+"/ZprimeToTauTau_M-500_TuneCP5_14TeV-pythia8-tauola__Phase2HLTTDRSummer20ReRECOMiniAOD-PU200_111X_mcRun4_realistic_T15_v1-v1__GEN-SIM-DIGI-RAW-MINIAOD/")
# list_njobs.append(47)

# list_filelists.append(open(filedir+"ZprimeToTauTau_M-1500_TuneCP5_14TeV-pythia8-tauola__Phase2HLTTDRSummer20ReRECOMiniAOD-PU200_111X_mcRun4_realistic_T15_v1-v1__GEN-SIM-DIGI-RAW-MINIAOD.txt"))
# list_folders.append("/data_CMS/cms/motta/Phase2L1T/L1TauMinatorNtuples/v"+version+"/ZprimeToTauTau_M-1500_TuneCP5_14TeV-pythia8-tauola__Phase2HLTTDRSummer20ReRECOMiniAOD-PU200_111X_mcRun4_realistic_T15_v1-v1__GEN-SIM-DIGI-RAW-MINIAOD/")
# list_njobs.append(48)

# list_filelists.append(open(filedir+"QCD_Pt-15to3000_TuneCP5_Flat_14TeV-pythia8__Phase2HLTTDRSummer20ReRECOMiniAOD-PU200_castor_111X_mcRun4_realistic_T15_v1-v1__FEVT.txt"))
# list_folders.append("/data_CMS/cms/motta/Phase2L1T/L1TauMinatorNtuples/v"+version+"/QCD_Pt-15to3000_TuneCP5_Flat_14TeV-pythia8__Phase2HLTTDRSummer20ReRECOMiniAOD-PU200_castor_111X_mcRun4_realistic_T15_v1-v1__FEVT/")
# list_njobs.append(250)

list_filelists.append(open(filedir+"MinBias_TuneCP5_14TeV-pythia8__Phase2HLTTDRSummer20ReRECOMiniAOD-PU200_withNewMB_111X_mcRun4_realistic_T15_v1_ext1-v2__FEVT.txt"))
list_folders.append("/data_CMS/cms/motta/Phase2L1T/L1TauMinatorNtuples/v"+version+"/MinBias_TuneCP5_14TeV-pythia8__Phase2HLTTDRSummer20ReRECOMiniAOD-PU200_withNewMB_111X_mcRun4_realistic_T15_v1_ext1-v2__FEVT/")
list_njobs.append(968)

os.system('mkdir -p /data_CMS/cms/motta/Phase2L1T/L1TauMinatorNtuples/v'+version)
os.system('cp listAll.sh /data_CMS/cms/motta/Phase2L1T/L1TauMinatorNtuples/v'+version)

##################################################################

os.system ('source /opt/exp_soft/cms/t3/t3setup')

for i in range(len(list_folders)):
    filelist = list_filelists[i]
    folder = list_folders[i]
    njobs = list_njobs[i]

    os.system('mkdir -p ' + folder)
    files = [f.strip() for f in filelist]
    print("Input has" , len(files) , "files")
    if njobs > len(files) : njobs = len(files)
    filelist.close()

    fileblocks = splitInBlocks (files, njobs)

    for idx, block in enumerate(fileblocks):
        outRootName = folder + '/Ntuple_' + str(idx) + '.root'
        outJobName  = folder + '/job_' + str(idx) + '.sh'
        inListName = folder + "/filelist_" + str(idx) + ".txt"
        outLogName  = folder + "/log_" + str(idx) + ".txt"

        jobfilelist = open(inListName, 'w')
        for f in block: jobfilelist.write(f+"\n")
        jobfilelist.close()

        cmsRun = "cmsRun test_Ntuplizer.py maxEvents=-1 inputFiles_load="+inListName + " outputFile="+outRootName + " >& " + outLogName

        skimjob = open (outJobName, 'w')
        skimjob.write ('#!/bin/bash\n')
        skimjob.write ('export X509_USER_PROXY=~/.t3/proxy.cert\n')
        skimjob.write ('source /cvmfs/cms.cern.ch/cmsset_default.sh\n')
        skimjob.write ('cd %s\n' % os.getcwd())
        skimjob.write ('export SCRAM_ARCH=slc6_amd64_gcc472\n')
        skimjob.write ('eval `scram r -sh`\n')
        skimjob.write (cmsRun+'\n')
        skimjob.close ()

        os.system ('chmod u+rwx ' + outJobName)
        command = ('/home/llr/cms/motta/t3submit -long \'' + outJobName +"\'")
        # command = ('/home/llr/cms/motta/t3submit -short \'' + outJobName +"\'")
        print(command)
        os.system (command)
        # break