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

#######################################################################
######################### SCRIPT BODY #################################
#######################################################################

parser = OptionParser()
parser.add_option("--v",          dest="v",                              default=None)
parser.add_option("--date",       dest="date",                           default=None)
parser.add_option("--outTag",     dest="outTag",                         default="")
parser.add_option("--uJetPtCut",  dest="uJetPtCut",                      default=False)
parser.add_option("--lJetPtCut",  dest="lJetPtCut",                      default=False)
parser.add_option("--uTauPtCut",  dest="uTauPtCut",                      default=False)
parser.add_option("--lTauPtCut",  dest="lTauPtCut",                      default=False)
parser.add_option("--etacut",     dest="etacut",                         default=False)
parser.add_option('--caloClNxM',  dest='caloClNxM',                      default="9x9")
parser.add_option('--doHH',       dest='doHH',      action='store_true', default=False)
parser.add_option('--doQCD',      dest='doQCD',     action='store_true', default=False)
parser.add_option('--doVBFH',     dest='doVBFH',    action='store_true', default=False)
parser.add_option('--doMinBias',  dest='doMinBias', action='store_true', default=False)
parser.add_option('--doTestRun',  dest='doTestRun', action='store_true', default=False)
(options, args) = parser.parse_args()

if not options.date or not options.v:
    print('** ERROR : no version and date specified --> no output folder specified')
    print('** EXITING')
    exit()

if not options.doHH and not options.doQCD and not options.doVBFH and not options.doMinBias and not options.doTestRun:
    print('** ERROR : no matching dataset specified. What do you want to do (doHH, doQCD, doVBFH, doMinBias, doTestRun)?')
    print('** EXITING')
    exit()

##################### DEFINE INPUTS AND OUTPUTS ####################
indir  = '/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v
outdir = '/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v

if options.doHH:
    indir  += '/GluGluToHHTo2B2Tau_node_SM_14TeV-madgraph-pythia8_tuneCP5__Phase2HLTTDRWinter20DIGI-PU200_110X_mcRun4_realistic_v3-v2__GEN-SIM-DIGI-RAW__batches'
    outdir += '/TensorizedInputs'+options.outTag

elif options.doVBFH:
    indir  += '/VBFHToTauTau_M125_14TeV_powheg_pythia8_correctedGridpack_tuneCP5__Phase2HLTTDRWinter20DIGI-PU200_110X_mcRun4_realistic_v3-v3__GEN-SIM-DIGI-RAW__batches'
    outdir += '/TensorizedInputs'+options.outTag

elif options.doQCD:
    indir  += '/QCD_Pt-15to3000_TuneCP5_Flat_14TeV-pythia8__Phase2HLTTDRWinter20DIGI-PU200_castor_110X_mcRun4_realistic_v3-v2__GEN-SIM-DIGI-RAW__batches'
    outdir += '/TensorizedInputs'+options.outTag

elif options.doMinBias:
    indir  += '/MinBias_TuneCP5_14TeV-pythia8__Phase2HLTTDRWinter20DIGI-PU200_110X_mcRun4_realistic_v3-v3__GEN-SIM-DIGI-RAW__batches'
    outdir += '/TensorizedInputs'+options.outTag

elif options.doTestRun:
    indir  += '/test__batches'
    outdir += '/TensorizedInputs'+options.outTag


os.system('mkdir -p ' + outdir)
tags = [tag.strip() for tag in open(indir+'/tagsFile.txt', 'r')]
njobs = len(tags)
print("Input has" , len(tags) , "files", "-->", len(tags), "jobs")
taglist.close()

for idx, tag in enumerate(tags):
    #print(idx, tag)

    outJobName  = folder + '/job_' + str(idx) + '.sh'
    outLogName  = folder + "/log_" + str(idx) + ".txt"

    command = 'python Tensorizer.py --fin '

















