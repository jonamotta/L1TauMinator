from optparse import OptionParser
import glob
import os

parser = OptionParser()
parser.add_option("--NtupleV",         dest="NtupleV",         default=None)
parser.add_option("--v",               dest="v",               default=None)
parser.add_option("--date",            dest="date",            default=None)
parser.add_option('--caloClNxM',       dest='caloClNxM',       default="5x9")
parser.add_option("--seedEtCut",       dest="seedEtCut",       default="2p5")
parser.add_option("--clusteringEtCut", dest="clusteringEtCut", default="")
parser.add_option("--etaRestriction",  dest="etaRestriction",  default="")
parser.add_option("--CBCEsplit",       dest="CBCEsplit",       default=1.5, type=float)
parser.add_option("--uTauPtCut",       dest="uTauPtCut",       default=None,  type=int)
parser.add_option("--lTauPtCut",       dest="lTauPtCut",       default=None,  type=int)
parser.add_option("--uEtacut",         dest="uEtacut",         default=None,  type=float)
parser.add_option("--lEtacut",         dest="lEtacut",         default=None,  type=float)
parser.add_option('--doHH',            dest='doHH',            default=False, action='store_true')
parser.add_option('--doVBFH',          dest='doVBFH',          default=False, action='store_true')
parser.add_option('--doGGH',           dest='doGGH',           default=False, action='store_true')
parser.add_option('--doDY',            dest='doDY',            default=False, action='store_true')
parser.add_option('--doDYlm',          dest='doDYlm',          default=False, action='store_true')
parser.add_option('--doEmuVal',        dest='doEmuVal',        default=False, action='store_true')
parser.add_option("--no_exec",         dest="no_exec",         default=False, action='store_true')
parser.add_option("--queue",           dest="queue",           default='short')
(options, args) = parser.parse_args()
print(options)


user = os.getcwd().split('/')[5]
infile_base = '/data_CMS/cms/'+user+'/Phase2L1T/L1TauMinatorNtuples/v'+options.NtupleV
outfile_base = '/data_CMS/cms/'+user+'/Phase2L1T/'+options.date+'_v'+options.v
if options.doEmuVal: outfile_base += '/EmuValidation'
os.system('mkdir -p '+outfile_base)

inlist_folders = []
outlist_folders = []

tag = ""
if options.uTauPtCut : tag += '_CBCEsplit'+str(options.CBCEsplit) 
if options.uTauPtCut : tag += '_uTauPtCut'+str(options.uTauPtCut)
if options.lTauPtCut : tag += '_lTauPtCut'+str(options.lTauPtCut)
if options.uEtacut   : tag += '_uEtacut'+str(options.uEtacut)
if options.lEtacut   : tag += '_lEtacut'+str(options.lEtacut)

if options.doGGH:
    inlist_folders.append(infile_base+"/GluGluHToTauTau_M-125_TuneCP5_14TeV-powheg-pythia8__Phase2Fall22DRMiniAOD-PU200_125X_mcRun4_realistic_v2-v1__GEN-SIM-DIGI-RAW-MINIAOD_seedEtCut"+options.seedEtCut+options.clusteringEtCut+options.etaRestriction+"/")
    outlist_folders.append(outfile_base+"/GluGluHToTauTau_cltw"+options.caloClNxM+"_seedEtCut"+options.seedEtCut+options.clusteringEtCut+options.etaRestriction+tag+"/")

if options.doVBFH:
    inlist_folders.append(infile_base+"/VBFHToTauTau_M-125_TuneCP5_14TeV-powheg-pythia8__Phase2Fall22DRMiniAOD-PU200_125X_mcRun4_realistic_v2-v1__GEN-SIM-DIGI-RAW-MINIAOD_seedEtCut"+options.seedEtCut+options.clusteringEtCut+options.etaRestriction+"/")
    outlist_folders.append(outfile_base+"/VBFHToTauTau_cltw"+options.caloClNxM+"_seedEtCut"+options.seedEtCut+options.clusteringEtCut+options.etaRestriction+tag+"/")

if options.doDYlm:
    inlist_folders.append(infile_base+"/DYToLL_M-10To50_TuneCP5_14TeV-pythia8__Phase2Fall22DRMiniAOD-PU200_Pilot_125X_mcRun4_realistic_v2-v2__GEN-SIM-DIGI-RAW-MINIAOD_seedEtCut"+options.seedEtCut+options.clusteringEtCut+options.etaRestriction+"/")
    outlist_folders.append(outfile_base+"/DYlowmass_cltw"+options.caloClNxM+"_seedEtCut"+options.seedEtCut+options.clusteringEtCut+options.etaRestriction+tag+"/")

if options.doDY:
    inlist_folders.append(infile_base+"/DYToLL_M-50_TuneCP5_14TeV-pythia8__Phase2Fall22DRMiniAOD-PU200_125X_mcRun4_realistic_v2-v1__GEN-SIM-DIGI-RAW-MINIAOD_seedEtCut"+options.seedEtCut+options.clusteringEtCut+options.etaRestriction+"/")
    outlist_folders.append(outfile_base+"/DY_cltw"+options.caloClNxM+"_seedEtCut"+options.seedEtCut+options.clusteringEtCut+options.etaRestriction+tag+"/")

if options.doEmuVal:
    inlist_folders.append(infile_base+"/GluGluToHHTo2B2Tau_node_SM_TuneCP5_14TeV-madgraph-pythia8__Phase2Fall22DRMiniAOD-PU200_125X_mcRun4_realistic_v2-v1__GEN-SIM-DIGI-RAW-MINIAOD_seedEtCut"+options.seedEtCut+options.clusteringEtCut+options.etaRestriction+"/")
    outlist_folders.append(outfile_base+"/HHbbtautau_cltw"+options.caloClNxM+"_seedEtCut"+options.seedEtCut+options.clusteringEtCut+options.etaRestriction+tag+"/")

for i in range(len(inlist_folders)):
    infolder = inlist_folders[i]
    outfolder = outlist_folders[i]
    files = glob.glob(infolder+'/Ntuple*.root')
    files.sort()

    os.system('mkdir -p '+outfolder+'/jobs')
    os.system('mkdir -p '+outfolder+'/logs')

    for file in files:
        idx = file.split('Ntuple_')[1].split('.')[0]
        outJobName  = outfolder + '/jobs/job_' + str(idx) + '.sh'
        outLogName  = outfolder + "/logs/log_" + str(idx) + ".txt"

        script = 'Chain2Tensor.py'
        if options.doEmuVal: script = 'Chain2Tensor_EmuValidation.py'

        cmsRun = "python3 "+script+" --fin "+file+" --fout "+outfolder
        if options.caloClNxM:
            cmsRun = cmsRun + " --caloClNxM "+options.caloClNxM
        if options.CBCEsplit:
            cmsRun = cmsRun + " --CBCEsplit "+str(options.CBCEsplit)
        if options.uTauPtCut:
            cmsRun = cmsRun + " --uTauPtCut "+str(options.uTauPtCut)
        if options.lTauPtCut:
            cmsRun = cmsRun + " --lTauPtCut "+str(options.lTauPtCut)
        if options.uEtacut:
            cmsRun = cmsRun + " --uEtacut "+str(options.uEtacut)
        if options.lEtacut:
            cmsRun = cmsRun + " --lEtacut "+str(options.lEtacut)

        cmsRun = cmsRun+ " >& " + outLogName

        skimjob = open (outJobName, 'w')
        skimjob.write ('#!/bin/bash\n')
        skimjob.write ('export X509_USER_PROXY=~/.t3/proxy.cert\n')
        skimjob.write ('source /cvmfs/cms.cern.ch/cmsset_default.sh\n')
        skimjob.write ('cd %s\n' % os.getcwd())
        skimjob.write ('export SCRAM_ARCH=slc6_amd64_gcc472\n')
        skimjob.write ('eval `scram r -sh`\n')
        skimjob.write ('cd %s\n'%os.getcwd())
        skimjob.write (cmsRun+'\n')
        skimjob.close ()

        os.system ('chmod u+rwx ' + outJobName)
        command = ('/data_CMS/cms/'+user+'/CaloL1calibraton/t3submit -'+options.queue+' \'' + outJobName +"\'")
        print(command)
        if not options.no_exec:
            os.system (command)

        # break
        # if idx == "11": break