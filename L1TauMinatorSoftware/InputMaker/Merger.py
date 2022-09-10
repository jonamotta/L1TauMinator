from optparse import OptionParser
import pandas as pd
import numpy as np
import random
import math
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
    parser.add_option('--caloClNxM',    dest='caloClNxM',                         default="5x9")
    parser.add_option('--doHGCAL',      dest='doHGCAL',      action='store_true', default=False)
    parser.add_option('--doCALO',       dest='doCALO',       action='store_true', default=False)
    parser.add_option('--doHH',         dest='doHH',         action='store_true', default=False)
    parser.add_option('--doQCD',        dest='doQCD',        action='store_true', default=False)
    parser.add_option('--doVBFH',       dest='doVBFH',       action='store_true', default=False)
    parser.add_option('--doMinBias',    dest='doMinBias',    action='store_true', default=False)
    parser.add_option('--doZp500',      dest='doZp500',      action='store_true', default=False)
    parser.add_option('--doZp1500',     dest='doZp1500',     action='store_true', default=False)
    parser.add_option('--doTestRun',    dest='doTestRun',    action='store_true', default=False)
    parser.add_option('--doTens4Calib', dest='doTens4Calib', action='store_true', default=False)
    parser.add_option('--doTens4Ident', dest='doTens4Ident', action='store_true', default=False)
    parser.add_option('--doTens4Rate',  dest='doTens4Rate',  action='store_true', default=False)
    (options, args) = parser.parse_args()


    ##################### DEFINE INPUTS AND OUTPUTS ####################
    indir  = '/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v

    indirs = []
    taglists = []

    if options.doCALO:
        if options.doTens4Calib: splitter = 'X_CNN_Calibrator'
        if options.doTens4Ident: splitter = 'X_CNN_Identifier'
        if options.doTens4Rate:  splitter = 'X_CNN_Rate'
        tensorsFolder = '/TensorizedInputs_'+options.caloClNxM+options.inTag

    if options.doHGCAL:
        if options.doTens4Calib: splitter = 'X_BDT_Calibrator'
        if options.doTens4Ident: splitter = 'X_BDT_Identifier'
        if options.doTens4Rate:  splitter = 'X_BDT_Rate'
        tensorsFolder = '/PickledInputs'+options.inTag

    if options.doHH:
        tmp = indir+'/GluGluToHHTo2B2Tau_node_SM_14TeV-madgraph-pythia8_tuneCP5__Phase2HLTTDRSummer20ReRECOMiniAOD-PU200_111X_mcRun4_realistic_T15_v1-v1__GEN-SIM-DIGI-RAW-MINIAOD__batches/'+tensorsFolder
        indirs.append(tmp)

        taglist = []
        if options.doCALO:  files = glob.glob(tmp+'/'+splitter+'*.npz')
        if options.doHGCAL: files = glob.glob(tmp+'/'+splitter+'*.pkl')
        for file in files:
            tag = '_'+file.split(splitter)[1].split('_')[1].split('.')[0]
            taglist.append(tag)
        taglists.append(taglist)

    if options.doVBFH:
        tmp = indir+'/VBFHToTauTau_M125_14TeV_powheg_pythia8_correctedGridpack_tuneCP5__Phase2HLTTDRSummer20ReRECOMiniAOD-PU200_111X_mcRun4_realistic_T15_v1-v1__FEVT__batches/'+tensorsFolder
        indirs.append(tmp)
        
        taglist = []
        if options.doCALO:  files = glob.glob(tmp+'/'+splitter+'*.npz')
        if options.doHGCAL: files = glob.glob(tmp+'/'+splitter+'*.pkl')
        for file in files:
            tag = '_'+file.split(splitter)[1].split('_')[1].split('.')[0]
            taglist.append(tag)
        taglists.append(taglist)

    if options.doQCD:
        tmp = indir+'/QCD_Pt-15to3000_TuneCP5_Flat_14TeV-pythia8__Phase2HLTTDRSummer20ReRECOMiniAOD-PU200_castor_111X_mcRun4_realistic_T15_v1-v1__FEVT__batches/'+tensorsFolder
        indirs.append(tmp)

        taglist = []
        if options.doCALO:  files = glob.glob(tmp+'/'+splitter+'*.npz')
        if options.doHGCAL: files = glob.glob(tmp+'/'+splitter+'*.pkl')
        for file in files:
            tag = '_'+file.split(splitter)[1].split('_')[1].split('.')[0]
            taglist.append(tag)
        taglists.append(taglist)

    if options.doMinBias:
        # tmp = indir+'/MinBias_TuneCP5_14TeV-pythia8__Phase2HLTTDRSummer20ReRECOMiniAOD-PU200_withNewMB_111X_mcRun4_realistic_T15_v1_ext1-v2__FEVT__batches/'+tensorsFolder
        tmp = indir+'/tmpRate__batches/'+tensorsFolder
        indirs.append(tmp)

        taglist = []
        if options.doCALO:  files = glob.glob(tmp+'/'+splitter+'*.npz')
        if options.doHGCAL: files = glob.glob(tmp+'/'+splitter+'*.pkl')
        for file in files:
            tag = '_'+file.split(splitter)[1].split('_')[1].split('.')[0]
            taglist.append(tag)
        taglists.append(taglist)

    if options.doZp500:
        tmp = indir+'/ZprimeToTauTau_M-500_TuneCP5_14TeV-pythia8-tauola__Phase2HLTTDRSummer20ReRECOMiniAOD-PU200_111X_mcRun4_realistic_T15_v1-v1__GEN-SIM-DIGI-RAW-MINIAOD__batches/'+tensorsFolder
        indirs.append(tmp)

        taglist = []
        if options.doCALO:  files = glob.glob(tmp+'/'+splitter+'*.npz')
        if options.doHGCAL: files = glob.glob(tmp+'/'+splitter+'*.pkl')
        for file in files:
            tag = '_'+file.split(splitter)[1].split('_')[1].split('.')[0]
            taglist.append(tag)
        taglists.append(taglist)

    if options.doZp1500:
        tmp = indir+'/ZprimeToTauTau_M-1500_TuneCP5_14TeV-pythia8-tauola__Phase2HLTTDRSummer20ReRECOMiniAOD-PU200_111X_mcRun4_realistic_T15_v1-v1__GEN-SIM-DIGI-RAW-MINIAOD__batches/'+tensorsFolder
        indirs.append(tmp)

        taglist = []
        if options.doCALO:  files = glob.glob(tmp+'/'+splitter+'*.npz')
        if options.doHGCAL: files = glob.glob(tmp+'/'+splitter+'*.pkl')
        for file in files:
            tag = '_'+file.split(splitter)[1].split('_')[1].split('.')[0]
            taglist.append(tag)
        taglists.append(taglist)

    if options.doTestRun:
        tmp = indir+'/test__batches/'+tensorsFolder
        indirs.append(tmp)

        taglist = []
        if options.doCALO:  files = glob.glob(tmp+'/'+splitter+'*.npz')
        if options.doHGCAL: files = glob.glob(tmp+'/'+splitter+'*.pkl')
        for file in files:
            tag = '_'+file.split(splitter)[1].split('_')[1].split('.')[0]
            taglist.append(tag)
        taglists.append(taglist)

    readFrom = {
        'inputsCalibratorCNN'    : '/X_CNN_Calibrator'+options.caloClNxM,
        'inputsCalibratorDense'  : '/X_Dense_Calibrator'+options.caloClNxM,
        'inputsIdentifierCNN'    : '/X_CNN_Identifier'+options.caloClNxM,
        'inputsIdentifierDense'  : '/X_Dense_Identifier'+options.caloClNxM,
        'inputsRateCNN'          : '/X_CNN_Rate'+options.caloClNxM,
        'inputsRateDense'        : '/X_Dense_Rate'+options.caloClNxM,
        'targetsCalibrator'      : '/Y_Calibrator'+options.caloClNxM,
        'targetsIdentifier'      : '/Y_Identifier'+options.caloClNxM,
        'targetsRate'            : '/Y_Rate'+options.caloClNxM,
        # -----------------------
        'inputsCalibratorBDT'    : '/X_BDT_Calibrator',
        'inputsIdentifierBDT'    : '/X_BDT_Identifier',
        'inputsRateBDT'          : '/X_BDT_Rate'
    }

    X1sToConcatenate = []
    X2sToConcatenate = []
    YsToConcatenate = []

    XsToConcatenate = []

    for i_fold, taglist in enumerate(taglists):
        for idx, tag in enumerate(taglist):
            tag = tag.strip()
            if not idx%10: print('reading batch', idx)
            try:
                # DEBUG
                # print(tag)
                # print(indirs[i_fold]+readFrom['inputsCalibratorBDT']+tag+'.npz')
                # print(np.load(indirs[i_fold]+readFrom['inputsCalibratorBDT']+tag+'.npz', allow_pickle=True)['arr_0'])
                # exit()

                if options.doTens4Calib:
                    if options.doCALO:
                        X1sToConcatenate.append(np.load(indirs[i_fold]+readFrom['inputsCalibratorCNN']+tag+'.npz', allow_pickle=True)['arr_0'])
                        X2sToConcatenate.append(np.load(indirs[i_fold]+readFrom['inputsCalibratorDense']+tag+'.npz', allow_pickle=True)['arr_0'])
                        YsToConcatenate.append(np.load(indirs[i_fold]+readFrom['targetsCalibrator']+tag+'.npz', allow_pickle=True)['arr_0'])

                    if options.doHGCAL:
                        XsToConcatenate.append(pd.read_pickle(indirs[i_fold]+readFrom['inputsCalibratorBDT']+tag+'.pkl'))


                elif options.doTens4Ident:
                    if options.doCALO:
                        X1sToConcatenate.append(np.load(indirs[i_fold]+readFrom['inputsIdentifierCNN']+tag+'.npz', allow_pickle=True)['arr_0'])
                        X2sToConcatenate.append(np.load(indirs[i_fold]+readFrom['inputsIdentifierDense']+tag+'.npz', allow_pickle=True)['arr_0'])
                        YsToConcatenate.append(np.load(indirs[i_fold]+readFrom['targetsIdentifier']+tag+'.npz', allow_pickle=True)['arr_0'])

                    if options.doHGCAL:
                        XsToConcatenate.append(pd.read_pickle(indirs[i_fold]+readFrom['inputsIdentifierBDT']+tag+'.pkl'))


                elif options.doTens4Rate:
                    if options.doCALO:
                        X1sToConcatenate.append(np.load(indirs[i_fold]+readFrom['inputsRateCNN']+tag+'.npz', allow_pickle=True)['arr_0'])
                        X2sToConcatenate.append(np.load(indirs[i_fold]+readFrom['inputsRateDense']+tag+'.npz', allow_pickle=True)['arr_0'])
                        YsToConcatenate.append(np.load(indirs[i_fold]+readFrom['targetsRate']+tag+'.npz', allow_pickle=True)['arr_0'])

                    if options.doHGCAL:
                        XsToConcatenate.append(pd.read_pickle(indirs[i_fold]+readFrom['inputsRateBDT']+tag+'.pkl'))

            except FileNotFoundError:
                # DEBUG
                print('** INFO: towers'+tag+' not found --> skipping')
                continue

            # uncomment if you wnat to have a smaller dataset tha the full one
            # if options.doTens4Calib and idx==400: break
            # if options.doTens4Ident and idx==200: break

    if not options.doTens4Rate:
        if options.doCALO:
            # shuffle the single batches to make the dataset mroe homogeneous and not dependent on concatenation order
            mixer = list(zip(X1sToConcatenate, X2sToConcatenate, YsToConcatenate))
            random.shuffle(mixer)
            X1sToConcatenate, X2sToConcatenate, YsToConcatenate = zip(*mixer)

            dp = int(math.ceil(len(X1sToConcatenate)/4*3))

            # concatenate batches in single tensors
            X1_train = np.concatenate(X1sToConcatenate[:dp])
            X2_train = np.concatenate(X2sToConcatenate[:dp])
            Y_train  = np.concatenate(YsToConcatenate[:dp])

            X1_valid = np.concatenate(X1sToConcatenate[dp:])
            X2_valid = np.concatenate(X2sToConcatenate[dp:])
            Y_valid  = np.concatenate(YsToConcatenate[dp:])

            ## DEBUG
            print('shape X1_train =', X1_train.shape)
            print('shape X2_train =', X2_train.shape)
            print('shape Y_train =', Y_train.shape)

            print('shape X1_valid =', X1_valid.shape)
            print('shape X2_valid =', X2_valid.shape)
            print('shape Y_valid =', Y_valid.shape)

        if options.doHGCAL:
            dp = int(math.ceil(len(XsToConcatenate)/4*3))

            # concatenate batches in single tensors
            X_train = pd.concat(XsToConcatenate[:dp], axis=0)
            X_valid = pd.concat(XsToConcatenate[dp:], axis=0)

            ## DEBUG
            print('shape X_train =', X_train.shape)
            print('shape X_valid =', X_valid.shape)

    else:
        if options.doCALO:
            X1 = np.concatenate(X1sToConcatenate)
            X2 = np.concatenate(X2sToConcatenate)
            Y  = np.concatenate(YsToConcatenate)

            ## DEBUG
            print('shape X1 =', X1.shape)
            print('shape X2 =', X2.shape)
            print('shape Y =', Y.shape)


        if options.doHGCAL:
            # concatenate batches in single tensors
            X = pd.concat(XsToConcatenate, axis=0)

            ## DEBUG
            print('shape X =', X.shape)

    if options.doTens4Calib:
        if options.doCALO:
            np.savez_compressed(indir+'/TauCNNCalibrator'+options.caloClNxM+'Training'+options.outTag+'/X_CNN_'+options.caloClNxM+'_forCalibrator.npz', X1_train)
            np.savez_compressed(indir+'/TauCNNCalibrator'+options.caloClNxM+'Training'+options.outTag+'/X_Dense_'+options.caloClNxM+'_forCalibrator.npz', X2_train)
            np.savez_compressed(indir+'/TauCNNCalibrator'+options.caloClNxM+'Training'+options.outTag+'/Y_'+options.caloClNxM+'_forCalibrator.npz', Y_train)

            np.savez_compressed(indir+'/TauCNNCalibrator'+options.caloClNxM+'Training'+options.outTag+'/X_CNN_'+options.caloClNxM+'_forEvaluator.npz', X1_valid)
            np.savez_compressed(indir+'/TauCNNCalibrator'+options.caloClNxM+'Training'+options.outTag+'/X_Dense_'+options.caloClNxM+'_forEvaluator.npz', X2_valid)
            np.savez_compressed(indir+'/TauCNNCalibrator'+options.caloClNxM+'Training'+options.outTag+'/Y_'+options.caloClNxM+'_forEvaluator.npz', Y_valid)

        if options.doHGCAL:
            X_train.to_pickle(indir+'/TauBDTCalibratorTraining'+options.outTag+'/X_Calib_BDT_forCalibrator.pkl')
            X_valid.to_pickle(indir+'/TauBDTCalibratorTraining'+options.outTag+'/X_Calib_BDT_forEvaluator.pkl')

    elif options.doTens4Ident:
        if options.doCALO:
            np.savez_compressed(indir+'/TauCNNIdentifier'+options.caloClNxM+'Training'+options.outTag+'/X_CNN_'+options.caloClNxM+'_forIdentifier.npz', X1_train)
            np.savez_compressed(indir+'/TauCNNIdentifier'+options.caloClNxM+'Training'+options.outTag+'/X_Dense_'+options.caloClNxM+'_forIdentifier.npz', X2_train)
            np.savez_compressed(indir+'/TauCNNIdentifier'+options.caloClNxM+'Training'+options.outTag+'/Y_'+options.caloClNxM+'_forIdentifier.npz', Y_train)

            np.savez_compressed(indir+'/TauCNNIdentifier'+options.caloClNxM+'Training'+options.outTag+'/X_CNN_'+options.caloClNxM+'_forEvaluator.npz', X1_valid)
            np.savez_compressed(indir+'/TauCNNIdentifier'+options.caloClNxM+'Training'+options.outTag+'/X_Dense_'+options.caloClNxM+'_forEvaluator.npz', X2_valid)
            np.savez_compressed(indir+'/TauCNNIdentifier'+options.caloClNxM+'Training'+options.outTag+'/Y_'+options.caloClNxM+'_forEvaluator.npz', Y_valid)

        if options.doHGCAL:
            X_train.to_pickle(indir+'/TauBDTIdentifierTraining'+options.outTag+'/X_Ident_BDT_forIdentifier.pkl')
            X_valid.to_pickle(indir+'/TauBDTIdentifierTraining'+options.outTag+'/X_Ident_BDT_forEvaluator.pkl')

    elif options.doTens4Rate:
        if options.doCALO:
            np.savez_compressed(indir+'/TauMinatorRateEvaluator_'+options.caloClNxM+'_CL3D'+options.outTag+'/X_Rate_CNN_'+options.caloClNxM+'.npz', X1)
            np.savez_compressed(indir+'/TauMinatorRateEvaluator_'+options.caloClNxM+'_CL3D'+options.outTag+'/X_Rate_Dense_'+options.caloClNxM+'.npz', X2)
            np.savez_compressed(indir+'/TauMinatorRateEvaluator_'+options.caloClNxM+'_CL3D'+options.outTag+'/Y_Rate_'+options.caloClNxM+'.npz', Y)

        if options.doHGCAL:
            X.to_pickle(indir+'/TauMinatorRateEvaluator_'+options.caloClNxM+'_CL3D'+options.outTag+'/X_Rate_BDT.pkl')
