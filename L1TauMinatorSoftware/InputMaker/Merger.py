from optparse import OptionParser
import numpy as np
import random
import glob
import os

#######################################################################
######################### SCRIPT BODY #################################
#######################################################################

if __name__ == "__main__" :

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


    if not options.date or not options.v:
        print('** ERROR : no version and date specified --> no output folder specified')
        print('** EXITING')
        exit()

    if not options.doTens4Calib and not options.doTens4Ident:
        print('** ERROR : no merging need specified')
        print('** EXITING')
        exit()

    if not options.doHH and not options.doQCD and not options.doVBFH and not options.doMinBias and not options.doZp500 and not options.doZp1500 and not options.doTestRun:
        print('** ERROR : no matching dataset specified. What do you want to do (doHH, doQCD, doVBFH, doMinBias, doTestRun)?')
        print('** EXITING')
        exit()

    ##################### DEFINE INPUTS AND OUTPUTS ####################
    indir  = '/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v

    indirs = []
    taglists = []

    if options.doTens4Calib: splitter = 'X_CNN_Calibrator'
    if options.doTens4Ident: splitter = 'X_CNN_Identifier'

    if options.doHH:
        tmp = indir+'/GluGluToHHTo2B2Tau_node_SM_14TeV-madgraph-pythia8_tuneCP5__Phase2HLTTDRSummer20ReRECOMiniAOD-PU200_111X_mcRun4_realistic_T15_v1-v1__GEN-SIM-DIGI-RAW-MINIAOD__batches/TensorizedInputs_'+options.caloClNxM+options.inTag
        indirs.append(tmp)

        taglist = []
        files = glob.glob(tmp+'/'+splitter+'*.npz')
        for file in files:
            tag = '_'+file.split(splitter)[1].split('_')[1].split('.')[0]
            taglist.append(tag)
        taglists.append(taglist)

    if options.doVBFH:
        tmp = indir+'/VBFHToTauTau_M125_14TeV_powheg_pythia8_correctedGridpack_tuneCP5__Phase2HLTTDRSummer20ReRECOMiniAOD-PU200_111X_mcRun4_realistic_T15_v1-v1__FEVT__batches/TensorizedInputs_'+options.caloClNxM+options.inTag
        indirs.append(tmp)
        
        taglist = []
        files = glob.glob(tmp+'/'+splitter+'*.npz')
        for file in files:
            tag = '_'+file.split(splitter)[1].split('_')[1].split('.')[0]
            taglist.append(tag)
        taglists.append(taglist)

    if options.doQCD:
        tmp = indir+'/QCD_Pt-15to3000_TuneCP5_Flat_14TeV-pythia8__Phase2HLTTDRSummer20ReRECOMiniAOD-PU200_castor_111X_mcRun4_realistic_T15_v1-v1__FEVT__batches/TensorizedInputs_'+options.caloClNxM+options.inTag
        indirs.append(tmp)

        taglist = []
        files = glob.glob(tmp+'/'+splitter+'*.npz')
        for file in files:
            tag = '_'+file.split(splitter)[1].split('_')[1].split('.')[0]
            taglist.append(tag)
        taglists.append(taglist)

    if options.doMinBias:
        tmp = indir+'/MinBias_TuneCP5_14TeV-pythia8__Phase2HLTTDRSummer20ReRECOMiniAOD-PU200_withNewMB_111X_mcRun4_realistic_T15_v1_ext1-v2__FEVT__batches/TensorizedInputs_'+options.caloClNxM+options.inTag
        indirs.append(tmp)

        taglist = []
        files = glob.glob(tmp+'/'+splitter+'*.npz')
        for file in files:
            tag = '_'+file.split(splitter)[1].split('_')[1].split('.')[0]
            taglist.append(tag)
        taglists.append(taglist)

    if options.doZp500:
        tmp = indir+'/ZprimeToTauTau_M-500_TuneCP5_14TeV-pythia8-tauola__Phase2HLTTDRSummer20ReRECOMiniAOD-PU200_111X_mcRun4_realistic_T15_v1-v1__GEN-SIM-DIGI-RAW-MINIAOD__batches/TensorizedInputs_'+options.caloClNxM+options.inTag
        indirs.append(tmp)

        taglist = []
        files = glob.glob(tmp+'/'+splitter+'*.npz')
        for file in files:
            tag = '_'+file.split(splitter)[1].split('_')[1].split('.')[0]
            taglist.append(tag)
        taglists.append(taglist)

    if options.doZp1500:
        tmp = indir+'/ZprimeToTauTau_M-1500_TuneCP5_14TeV-pythia8-tauola__Phase2HLTTDRSummer20ReRECOMiniAOD-PU200_111X_mcRun4_realistic_T15_v1-v1__GEN-SIM-DIGI-RAW-MINIAOD__batches/TensorizedInputs_'+options.caloClNxM+options.inTag
        indirs.append(tmp)

        taglist = []
        files = glob.glob(tmp+'/'+splitter+'*.npz')
        for file in files:
            tag = '_'+file.split(splitter)[1].split('_')[1].split('.')[0]
            taglist.append(tag)
        taglists.append(taglist)

    if options.doTestRun:
        tmp = indir+'/test__batches/TensorizedInputs_'+options.caloClNxM+options.inTag
        indirs.append(tmp)

        taglist = []
        files = glob.glob(tmp+'/'+splitter+'*.npz')
        for file in files:
            tag = '_'+file.split(splitter)[1].split('_')[1].split('.')[0]
            taglist.append(tag)
        taglists.append(taglist)


    outdir = indir + '/TauCNN'
    if options.doTens4Calib: outdir += 'Calibrator'
    if options.doTens4Ident: outdir += 'Identifier'
    outdir += options.caloClNxM+'Training'+options.outTag
    os.system('mkdir -p '+outdir)


    readFrom = {
        'inputsCalibratorCNN'    : '/X_CNN_Calibrator'+options.caloClNxM,
        'inputsCalibratorDense'  : '/X_Dense_Calibrator'+options.caloClNxM,
        'inputsIdentifierCNN'    : '/X_CNN_Identifier'+options.caloClNxM,
        'inputsIdentifierDense'  : '/X_Dense_Identifier'+options.caloClNxM,
        'targetsCalibrator'      : '/Y_Calibrator'+options.caloClNxM,
        'targetsIdentifier'      : '/Y_Identifier'+options.caloClNxM
    }

    X1sToConcatenate = []
    X2sToConcatenate = []
    YsToConcatenate = []

    for i_fold, taglist in enumerate(taglists):
        for idx, tag in enumerate(taglist):
            tag = tag.strip()
            if not idx%10: print('reading batch', idx)
            try:
                # DEBUG
                # print(tag)
                # print(indirs[i_fold]+readFrom['inputsCalibrator']+tag+'.npz')
                # print(np.load(indirs[i_fold]+readFrom['inputsCalibrator']+tag+'.npz', allow_pickle=True)['arr_0'])
                # exit()

                if options.doTens4Calib:
                    X1sToConcatenate.append(np.load(indirs[i_fold]+readFrom['inputsCalibratorCNN']+tag+'.npz', allow_pickle=True)['arr_0'])
                    X2sToConcatenate.append(np.load(indirs[i_fold]+readFrom['inputsCalibratorDense']+tag+'.npz', allow_pickle=True)['arr_0'])
                    YsToConcatenate.append(np.load(indirs[i_fold]+readFrom['targetsCalibrator']+tag+'.npz', allow_pickle=True)['arr_0'])

                elif options.doTens4Ident:
                    X1sToConcatenate.append(np.load(indirs[i_fold]+readFrom['inputsIdentifierCNN']+tag+'.npz', allow_pickle=True)['arr_0'])
                    X2sToConcatenate.append(np.load(indirs[i_fold]+readFrom['inputsIdentifierDense']+tag+'.npz', allow_pickle=True)['arr_0'])
                    YsToConcatenate.append(np.load(indirs[i_fold]+readFrom['targetsIdentifier']+tag+'.npz', allow_pickle=True)['arr_0'])


            except FileNotFoundError:
                # DEBUG
                print('** INFO: towers'+tag+' not found --> skipping')
                continue

            # if idx==50: break

    # shuffle the single batches to make the dataset mroe homogeneous and not dependent on concatenation order
    mixer = list(zip(X1sToConcatenate, X2sToConcatenate, YsToConcatenate))
    random.shuffle(mixer)
    X1sToConcatenate, X2sToConcatenate, YsToConcatenate = zip(*mixer)

    # concatenate batches in single tensors
    X1 = np.concatenate(X1sToConcatenate)
    X2 = np.concatenate(X2sToConcatenate)
    Y = np.concatenate(YsToConcatenate)

    ## DEBUG
    print(len(X1))
    print(len(X2))
    print(len(Y))

    if options.doTens4Calib:
        np.savez_compressed(outdir+'/X_CNN_'+options.caloClNxM+'_forCalibrator.npz', X1)
        np.savez_compressed(outdir+'/X_Dense_'+options.caloClNxM+'_forCalibrator.npz', X2)
        np.savez_compressed(outdir+'/Y'+options.caloClNxM+'_forCalibrator.npz', Y)

    elif options.doTens4Ident:
        np.savez_compressed(outdir+'/X_CNN_'+options.caloClNxM+'_forIdentifier.npz', X1)
        np.savez_compressed(outdir+'/X_Dense_'+options.caloClNxM+'_forIdentifier.npz', X2)
        np.savez_compressed(outdir+'/Y'+options.caloClNxM+'_forIdentifier.npz', Y)
