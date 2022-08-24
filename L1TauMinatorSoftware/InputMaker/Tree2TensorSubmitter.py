from optparse import OptionParser
import glob
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

#######################################################################
######################### SCRIPT BODY #################################
#######################################################################

parser = OptionParser()
# GENERAL OPTIONS
parser.add_option("--v",            dest="v",            help="Version of the iteration",                                        default=None)
parser.add_option("--date",         dest="date",         help="Date of birth of this version",                                   default=None)
parser.add_option('--caloClNxM',    dest='caloClNxM',    help='Which shape of CaloCluster to use?',                              default="9x9")
# TTREE READING OPTIONS
parser.add_option('--doHH',         dest='doHH',         help='Read the HH samples?',                       action='store_true', default=False)
parser.add_option('--doQCD',        dest='doQCD',        help='Read the QCD samples?',                      action='store_true', default=False)
parser.add_option('--doVBFH',       dest='doVBFH',       help='Read the VBF H samples?',                    action='store_true', default=False)
parser.add_option('--doMinBias',    dest='doMinBias',    help='Read the Minbias samples?',                  action='store_true', default=False)
parser.add_option('--doZp500',      dest='doZp500',      help='Read the Zp500 samples?',                    action='store_true', default=False)
parser.add_option('--doZp1500',     dest='doZp1500',     help='Read the Zp1500 samples?',                   action='store_true', default=False)
parser.add_option('--doTestRun',    dest='doTestRun',    help='Do test run with reduced number of events?', action='store_true', default=False)
# TENSORIZATION OPTIONS
parser.add_option("--outTag",       dest="outTag",                            default="")
parser.add_option("--uJetPtCut",    dest="uJetPtCut",                         default=None)
parser.add_option("--lJetPtCut",    dest="lJetPtCut",                         default=None)
parser.add_option("--uTauPtCut",    dest="uTauPtCut",                         default=None)
parser.add_option("--lTauPtCut",    dest="lTauPtCut",                         default=None)
parser.add_option("--etacut",       dest="etacut",                            default=None)
parser.add_option('--doTens4Calib', dest='doTens4Calib', action='store_true', default=False)
parser.add_option('--doTens4Ident', dest='doTens4Ident', action='store_true', default=False)
(options, args) = parser.parse_args()

if not options.date or not options.v:
    print('** ERROR : no version and date specified --> no output folder specified')
    print('** EXITING')
    exit()

if not options.doTens4Calib and not options.doTens4Ident:
    print('** ERROR : no tensorization need specified')
    print('** EXITING')
    exit()

if not options.doHH and not options.doQCD and not options.doVBFH and not options.doMinBias and not options.doZp500 and not options.doZp1500 and not options.doTestRun:
    print('** ERROR : no matching dataset specified. What do you want to do (doHH, doQCD, doVBFH, doMinBias, doZp500, doZp1500, doTestRun)?')
    print('** EXITING')
    exit()

##################### DEFINE INPUTS AND OUTPUTS ####################
version = "0"
indir  = '/data_CMS/cms/motta/Phase2L1T/L1TauMinatorNtuples/v'+version
outdir = '/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v

if options.doHH:
    indir  += '/GluGluToHHTo2B2Tau_node_SM_14TeV-madgraph-pythia8_tuneCP5__Phase2HLTTDRSummer20ReRECOMiniAOD-PU200_111X_mcRun4_realistic_T15_v1-v1__GEN-SIM-DIGI-RAW-MINIAOD'
    outdir += '/GluGluToHHTo2B2Tau_node_SM_14TeV-madgraph-pythia8_tuneCP5__Phase2HLTTDRSummer20ReRECOMiniAOD-PU200_111X_mcRun4_realistic_T15_v1-v1__GEN-SIM-DIGI-RAW-MINIAOD__batches'

elif options.doVBFH:
    indir  += '/VBFHToTauTau_M125_14TeV_powheg_pythia8_correctedGridpack_tuneCP5__Phase2HLTTDRSummer20ReRECOMiniAOD-PU200_111X_mcRun4_realistic_T15_v1-v1__FEVT'
    outdir += '/VBFHToTauTau_M125_14TeV_powheg_pythia8_correctedGridpack_tuneCP5__Phase2HLTTDRSummer20ReRECOMiniAOD-PU200_111X_mcRun4_realistic_T15_v1-v1__FEVT__batches'

elif options.doQCD:
    indir  += '/QCD_Pt-15to3000_TuneCP5_Flat_14TeV-pythia8__Phase2HLTTDRSummer20ReRECOMiniAOD-PU200_castor_111X_mcRun4_realistic_T15_v1-v1__FEVT'
    outdir += '/QCD_Pt-15to3000_TuneCP5_Flat_14TeV-pythia8__Phase2HLTTDRSummer20ReRECOMiniAOD-PU200_castor_111X_mcRun4_realistic_T15_v1-v1__FEVT__batches'

elif options.doMinBias:
    indir  += '/MinBias_TuneCP5_14TeV-pythia8__Phase2HLTTDRSummer20ReRECOMiniAOD-PU200_withNewMB_111X_mcRun4_realistic_T15_v1_ext1-v2__FEVT'
    outdir += '/MinBias_TuneCP5_14TeV-pythia8__Phase2HLTTDRSummer20ReRECOMiniAOD-PU200_withNewMB_111X_mcRun4_realistic_T15_v1_ext1-v2__FEVT__batches'

elif options.doZp500:
    indir  += '/ZprimeToTauTau_M-500_TuneCP5_14TeV-pythia8-tauola__Phase2HLTTDRSummer20ReRECOMiniAOD-PU200_111X_mcRun4_realistic_T15_v1-v1__GEN-SIM-DIGI-RAW-MINIAOD'
    outdir += '/ZprimeToTauTau_M-500_TuneCP5_14TeV-pythia8-tauola__Phase2HLTTDRSummer20ReRECOMiniAOD-PU200_111X_mcRun4_realistic_T15_v1-v1__GEN-SIM-DIGI-RAW-MINIAOD__batches'

elif options.doZp1500:
    indir  += '/ZprimeToTauTau_M-1500_TuneCP5_14TeV-pythia8-tauola__Phase2HLTTDRSummer20ReRECOMiniAOD-PU200_111X_mcRun4_realistic_T15_v1-v1__GEN-SIM-DIGI-RAW-MINIAOD'
    outdir += '/ZprimeToTauTau_M-1500_TuneCP5_14TeV-pythia8-tauola__Phase2HLTTDRSummer20ReRECOMiniAOD-PU200_111X_mcRun4_realistic_T15_v1-v1__GEN-SIM-DIGI-RAW-MINIAOD__batches'

elif options.doTestRun:
    indir  += '/test'
    outdir += '/test__batches'

os.system('mkdir -p '+outdir+'/L1Clusters')
os.system('mkdir -p '+outdir+'/GenObjects')
os.system('mkdir -p '+outdir+'/TensorizedInputs_'+options.caloClNxM+options.outTag)

jobsdir = outdir+'/jobs/jobs_'+options.caloClNxM+options.outTag
os.system('mkdir -p '+jobsdir)

# list Ntuples
InFiles = []
files = glob.glob(indir+'/Ntuple*.root')
for file in files:
    InFiles.append(file)
InFiles.sort()

for i, infile in enumerate(InFiles[:]):
    print(infile)

    tag = infile.split('/Ntuple')[1].split('.r')[0]

    outJobName  = jobsdir + '/job' + tag + '.sh'
    outLogName  = jobsdir + "/log" + tag + ".txt"

    cmsRun = 'python Tree2Tensor.py'
    # GENERAL OPTIONS
    cmsRun += ' --infile '  + infile
    cmsRun += ' --outdir ' + outdir
    cmsRun += ' --caloClNxM ' + options.caloClNxM
    # TTREE READING OPTIONS
    if options.doHH:      cmsRun += ' --doHH'
    if options.doQCD:     cmsRun += ' --doQCD'
    if options.doVBFH:    cmsRun += ' --doVBFH'
    if options.doMinBias: cmsRun += ' --doMinBias'
    if options.doZp500:   cmsRun += ' --doZp500'
    if options.doZp1500:  cmsRun += ' --doZp1500'
    if options.doTestRun: cmsRun += ' --doTestRun'
    # TENSORIZATION OPTIONS
    cmsRun += ' --infileTag ' + tag
    if options.outTag != "": cmsRun += ' --outTag '    + options.outTag
    if options.outTag:       cmsRun += ' --outTag '    + options.outTag
    if options.uJetPtCut:    cmsRun += ' --uJetPtCut ' + options.uJetPtCut
    if options.lJetPtCut:    cmsRun += ' --lJetPtCut ' + options.lJetPtCut
    if options.uTauPtCut:    cmsRun += ' --uTauPtCut ' + options.uTauPtCut
    if options.lTauPtCut:    cmsRun += ' --lTauPtCut ' + options.lTauPtCut
    if options.etacut:       cmsRun += ' --etacut '    + options.etacut
    if options.doTens4Calib: cmsRun += ' --doTens4Calib'
    if options.doTens4Ident: cmsRun += ' --doTens4Ident'
    cmsRun += ' >& ' + outLogName

    skimjob = open(outJobName, 'w')
    skimjob.write('#!/bin/bash\n')
    skimjob.write('export X509_USER_PROXY=~/.t3/proxy.cert\n')
    skimjob.write('module use /opt/exp_soft/vo.llr.in2p3.fr/modulefiles_el7\n')
    skimjob.write('module load python/3.7.0\n')
    skimjob.write('cd %s\n'%os.getcwd())
    skimjob.write(cmsRun+'\n')
    skimjob.close()

    os.system ('chmod u+rwx ' + outJobName)
    command = ('/home/llr/cms/motta/t3submit -short \'' + outJobName +"\'")
    # command = ('/home/llr/cms/evernazza/t3submit -short \'' + outJobName +"\'")
    print(command)
    os.system (command)
    # if i == 2: break
