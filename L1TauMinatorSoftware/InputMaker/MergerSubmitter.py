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

# read the batched input tensors to the NN and merge them
parser = OptionParser()
parser.add_option("--v",            dest="v",                                 default=None)
parser.add_option("--date",         dest="date",                              default=None)
parser.add_option("--inTag",        dest="inTag",                             default="")
parser.add_option("--outTag",       dest="outTag",                            default="")
parser.add_option('--caloClNxM',    dest='caloClNxM',                         default="9x9")
parser.add_option('--doHH',         dest='doHH',         action='store_true', default=False)
parser.add_option('--doQCD',        dest='doQCD',        action='store_true', default=False)
parser.add_option('--doVBFH',       dest='doVBFH',       action='store_true', default=False)
parser.add_option('--doMinBias',    dest='doMinBias',    action='store_true', default=False)
parser.add_option('--doZp500',      dest='doZp500',      action='store_true', default=False)
parser.add_option('--doZp1500',     dest='doZp1500',     action='store_true', default=False)
parser.add_option('--doTestRun',    dest='doTestRun',    action='store_true', default=False)
parser.add_option('--doTens4Calib', dest='doTens4Calib', action='store_true', default=None)
parser.add_option('--doTens4Ident', dest='doTens4Ident', action='store_true', default=None)
(options, args) = parser.parse_args()


jobsdir = '/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNN'
if options.doTens4Calib: jobsdir += 'Calibrator'
if options.doTens4Ident: jobsdir += 'Identifier'
jobsdir += options.caloClNxM+'Training'+options.outTag
os.system('mkdir -p '+jobsdir)

outJobName  = jobsdir + '/job.sh'
outLogName  = jobsdir + '/log.txt'

cmsRun = 'python Merger.py'
cmsRun += ' --v '+options.v
cmsRun += ' --date '+options.date
cmsRun += ' --inTag '+options.inTag
cmsRun += ' --outTag '+options.outTag
cmsRun += ' --caloClNxM '+options.caloClNxM
if options.doHH          : cmsRun += ' --doHH'
if options.doQCD         : cmsRun += ' --doQCD'
if options.doVBFH        : cmsRun += ' --doVBFH'
if options.doMinBias     : cmsRun += ' --doMinBias'
if options.doZp500       : cmsRun += ' --doZp500'
if options.doZp1500      : cmsRun += ' --doZp1500'
if options.doTestRun     : cmsRun += ' --doTestRun'
if options.doTens4Calib  : cmsRun += ' --doTens4Calib'
if options.doTens4Ident  : cmsRun += ' --doTens4Ident'
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