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

filedir="/home/llr/cms/motta/Phase2L1T/CMSSW_12_3_0_pre4/src/L1TauMinator/L1TauMinatorEmulator/inputFiles/"

list_filelists = []
list_folders = []
list_njobs = []

list_filelists.append(open(filedir+"test.txt"))
# list_folders.append("/data_cms_upgrade/motta/Phase2L1T_SKIMS/test/")
list_folders.append("/data_CMS/cms/motta/Phase2L1T_SKIMS/test/")
list_njobs.append(100)

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

        cmsRun = "cmsRun test_Ntuplizer.py maxEvents=1 inputFiles_load="+inListName + " outputFile="+outRootName + " >& " + outLogName

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
        # command = ('/home/llr/cms/motta/t3submit -long \'' + outJobName +"\'")
        command = ('/home/llr/cms/motta/t3submit -short \'' + outJobName +"\'")
        print(command)
        os.system (command)
        # break