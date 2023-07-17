from optparse import OptionParser
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

parser = OptionParser()
parser.add_option("--NtupleV",         dest="NtupleV",         default=None)
parser.add_option("--seedEtCut",       dest="seedEtCut",       default=2.5,   type=float)
parser.add_option("--etaRestriction",  dest="etaRestriction",  default=3.5,   type=float)
parser.add_option("--CB_CE_split",     dest="CB_CE_split",     default=1.5,   type=float)
parser.add_option("--clusteringEtCut", dest="clusteringEtCut", default=0.0,   type=float)
parser.add_option("--NNv",             dest="NNv",             default=None)
parser.add_option("--TauMinator",      dest="TauMinator",      default=False, action='store_true')
parser.add_option("--TauMinatorPrd",   dest="TauMinatorPrd",   default=False, action='store_true')
parser.add_option("--TauMinatorEmu",   dest="TauMinatorEmu",   default=False, action='store_true')
(options, args) = parser.parse_args()

seedEtCutTag = 'seedEtCut'+str(int(options.seedEtCut))+'p'+str(int((options.seedEtCut-int(options.seedEtCut))*10))
clusteringEtCutTag = ''
if options.clusteringEtCut != 0.0:
    clusteringEtCutTag = '_clusteringEtCut'+str(int(options.clusteringEtCut))+'p'+str(int((options.clusteringEtCut-int(options.clusteringEtCut))*10))
etaRestrictionTag = ''
if options.etaRestriction != 3.5:
    etaRestrictionTag = '_Er'+str(int(options.etaRestriction))+'p'+str(int(round((options.etaRestriction-int(options.etaRestriction))*10)))
CBCEsplitTag = '_CBCEsplit'+str(int(options.CB_CE_split))+'p'+str(int(round((options.CB_CE_split-int(options.CB_CE_split))*100)))
NNvtag = "_NN"+options.NNv

version = options.NtupleV

infile_base  = os.getcwd()+'/../inputFiles/'
user = infile_base.split('/')[5]
outfile_base = "/data_CMS/cms/"+user+"/Phase2L1T/L1TauMinatorNtuples/v"+version+"/"

list_filelists = []
list_folders = []
list_njobs = []

# list_filelists.append(open(infile_base+"test.txt"))
# list_folders.append(outfile_base+"test/")
# list_njobs.append(10)

# list_filelists.append(open(infile_base+"GluGluHToTauTau_M-125_TuneCP5_14TeV-powheg-pythia8__Phase2Fall22DRMiniAOD-PU200_125X_mcRun4_realistic_v2-v1__GEN-SIM-DIGI-RAW-MINIAOD.txt"))
# list_folders.append(outfile_base+"GluGluHToTauTau_M-125_TuneCP5_14TeV-powheg-pythia8__Phase2Fall22DRMiniAOD-PU200_125X_mcRun4_realistic_v2-v1__GEN-SIM-DIGI-RAW-MINIAOD_"+seedEtCutTag+clusteringEtCutTag+etaRestrictionTag+"/")
# list_njobs.append(350)

# list_filelists.append(open(infile_base+"VBFHToTauTau_M-125_TuneCP5_14TeV-powheg-pythia8__Phase2Fall22DRMiniAOD-PU200_125X_mcRun4_realistic_v2-v1__GEN-SIM-DIGI-RAW-MINIAOD.txt"))
# list_folders.append(outfile_base+"VBFHToTauTau_M-125_TuneCP5_14TeV-powheg-pythia8__Phase2Fall22DRMiniAOD-PU200_125X_mcRun4_realistic_v2-v1__GEN-SIM-DIGI-RAW-MINIAOD_"+seedEtCutTag+clusteringEtCutTag+etaRestrictionTag+"/")
# list_njobs.append(650)

# list_filelists.append(open(infile_base+"DYToLL_M-10To50_TuneCP5_14TeV-pythia8__Phase2Fall22DRMiniAOD-PU200_Pilot_125X_mcRun4_realistic_v2-v2__GEN-SIM-DIGI-RAW-MINIAOD.txt"))
# list_folders.append(outfile_base+"DYToLL_M-10To50_TuneCP5_14TeV-pythia8__Phase2Fall22DRMiniAOD-PU200_Pilot_125X_mcRun4_realistic_v2-v2__GEN-SIM-DIGI-RAW-MINIAOD_"+seedEtCutTag+clusteringEtCutTag+etaRestrictionTag+"/")
# list_njobs.append(650)

# list_filelists.append(open(infile_base+"DYToLL_M-50_TuneCP5_14TeV-pythia8__Phase2Fall22DRMiniAOD-PU200_125X_mcRun4_realistic_v2-v1__GEN-SIM-DIGI-RAW-MINIAOD.txt"))
# list_folders.append(outfile_base+"DYToLL_M-50_TuneCP5_14TeV-pythia8__Phase2Fall22DRMiniAOD-PU200_125X_mcRun4_realistic_v2-v1__GEN-SIM-DIGI-RAW-MINIAOD_"+seedEtCutTag+clusteringEtCutTag+etaRestrictionTag+"/")
# list_njobs.append(1580)

# list_filelists.append(open(infile_base+"GluGluToHHTo2B2Tau_node_SM_TuneCP5_14TeV-madgraph-pythia8__Phase2Fall22DRMiniAOD-PU200_125X_mcRun4_realistic_v2-v1__GEN-SIM-DIGI-RAW-MINIAOD.txt"))
# list_folders.append(outfile_base+"GluGluToHHTo2B2Tau_node_SM_TuneCP5_14TeV-madgraph-pythia8__Phase2Fall22DRMiniAOD-PU200_125X_mcRun4_realistic_v2-v1__GEN-SIM-DIGI-RAW-MINIAOD_"+seedEtCutTag+clusteringEtCutTag+etaRestrictionTag+CBCEsplitTag+NNvtag+"/")
# list_njobs.append(350)

list_filelists.append(open(infile_base+"MinBias_TuneCP5_14TeV-pythia8__Phase2Fall22DRMiniAOD-PU200_125X_mcRun4_realistic_v2-v1__GEN-SIM-DIGI-RAW-MINIAOD.txt"))
# list_filelists.append(open(infile_base+"MinBias_TMP.txt"))
list_folders.append(outfile_base+"MinBias_TuneCP5_14TeV-pythia8__Phase2Fall22DRMiniAOD-PU200_125X_mcRun4_realistic_v2-v1__GEN-SIM-DIGI-RAW-MINIAOD_"+seedEtCutTag+clusteringEtCutTag+etaRestrictionTag+CBCEsplitTag+NNvtag+"/")
list_njobs.append(12500)

os.system('mkdir -p /data_CMS/cms/'+user+'/Phase2L1T/L1TauMinatorNtuples/v'+version)
os.system('cp listAll.sh /data_CMS/cms/'+user+'/Phase2L1T/L1TauMinatorNtuples/v'+version)

##################################################################

# os.system ('source /opt/exp_soft/cms/t3/t3setup')

for i in range(len(list_folders)):
    filelist = list_filelists[i]
    folder = list_folders[i]
    njobs = list_njobs[i]

    os.system('mkdir -p ' + folder)

    os.system('cp resubmitTensorflowCrazyOut.sh '+folder)

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

        main_config = "test_Ntuplizer.py"
        if options.TauMinator: main_config = "test_L1CaloTauNtuplizer.py"
        if options.TauMinatorPrd: main_config = "test_L1CaloTauNtuplizer_Producer.py"
        if options.TauMinatorEmu: main_config = "test_L1CaloTauNtuplizer_Emulator.py"

        cmsRun = "cmsRun "+main_config+" maxEvents=-1 inputFiles_load="+inListName+" outputFile="+outRootName+" minSeedEt="+str(options.seedEtCut)+" minClusteringEt="+str(options.clusteringEtCut)+" etaRestriction="+str(options.etaRestriction)+" CBCEsplit="+str(options.CB_CE_split)+" NNv="+options.NNv+" >& "+outLogName
        # cmsRun = "cmsRun "+main_config+" maxEvents=-1 inputFiles_load="+inListName+" outputFile="+outRootName+" minSeedEt="+str(options.seedEtCut)+" >& "+outLogName

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
        # command = ('/home/llr/cms/'+user+'/t3submit -long \'' + outJobName +"\'")
        command = ('/home/llr/cms/'+user+'/t3submit -short \'' + outJobName +"\'")
        print(command)
        os.system(command)
        # break
        if idx==5000: break
