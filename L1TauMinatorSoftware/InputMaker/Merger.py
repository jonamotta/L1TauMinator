from optparse import OptionParser
import numpy as np
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

    if not options.doHH and not options.doQCD and not options.doVBFH and not options.doMinBias and not options.doTestRun:
        print('** ERROR : no matching dataset specified. What do you want to do (doHH, doQCD, doVBFH, doMinBias, doTestRun)?')
        print('** EXITING')
        exit()

    ##################### DEFINE INPUTS AND OUTPUTS ####################
    indir  = '/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v

    indirs = []
    taglists = []

    if options.doHH:
        indirs.append(indir+'/GluGluToHHTo2B2Tau_node_SM_14TeV-madgraph-pythia8_tuneCP5__Phase2HLTTDRWinter20DIGI-PU200_110X_mcRun4_realistic_v3-v2__GEN-SIM-DIGI-RAW__batches/TensorizedInputs_'+options.caloClNxM+options.inTag)
        taglists.append(open(indir+'/GluGluToHHTo2B2Tau_node_SM_14TeV-madgraph-pythia8_tuneCP5__Phase2HLTTDRWinter20DIGI-PU200_110X_mcRun4_realistic_v3-v2__GEN-SIM-DIGI-RAW__batches/tagsFile.txt'))

    if options.doVBFH:
        indirs.append(indir+'/VBFHToTauTau_M125_14TeV_powheg_pythia8_correctedGridpack_tuneCP5__Phase2HLTTDRWinter20DIGI-PU200_110X_mcRun4_realistic_v3-v3__GEN-SIM-DIGI-RAW__batches/TensorizedInputs_'+options.caloClNxM+options.inTag)
        taglists.append(open(indir+'/VBFHToTauTau_M125_14TeV_powheg_pythia8_correctedGridpack_tuneCP5__Phase2HLTTDRWinter20DIGI-PU200_110X_mcRun4_realistic_v3-v3__GEN-SIM-DIGI-RAW__batches/tagsFile.txt'))

    if options.doQCD:
        indirs.append(indir+'/QCD_Pt-15to3000_TuneCP5_Flat_14TeV-pythia8__Phase2HLTTDRWinter20DIGI-PU200_castor_110X_mcRun4_realistic_v3-v2__GEN-SIM-DIGI-RAW__batches/TensorizedInputs_'+options.caloClNxM+options.inTag)
        taglists.append(open(indir+'/QCD_Pt-15to3000_TuneCP5_Flat_14TeV-pythia8__Phase2HLTTDRWinter20DIGI-PU200_castor_110X_mcRun4_realistic_v3-v2__GEN-SIM-DIGI-RAW__batches/tagsFile.txt'))

    if options.doMinBias:
        indirs.append(indir+'/MinBias_TuneCP5_14TeV-pythia8__Phase2HLTTDRWinter20DIGI-PU200_110X_mcRun4_realistic_v3-v3__GEN-SIM-DIGI-RAW__batches/TensorizedInputs_'+options.caloClNxM+options.inTag)
        taglists.append(open(indir+'/MinBias_TuneCP5_14TeV-pythia8__Phase2HLTTDRWinter20DIGI-PU200_110X_mcRun4_realistic_v3-v3__GEN-SIM-DIGI-RAW__batches/tagsFile.txt'))

    if options.doTestRun:
        indirs.append(indir+'/test__batches/TensorizedInputs_'+options.caloClNxM+options.inTag)
        taglists.append(open(indir+'/test__batches/tagsFile.txt'))


    outdir = indir + '/TauCNN'
    if options.doTens4Calib: outdir += 'Calibrator_training'
    if options.doTens4Ident: outdir += 'Identifier_training'
    outdir += options.outTag
    os.system('mkdir -p '+outdir)


    readFrom = {
        'inputsCalibrator'  : '/X_Calibrator'+options.caloClNxM,
        'inputsIdentifier'  : '/X_Identifier'+options.caloClNxM,
        'targetsCalibrator' : '/Y_Calibrator'+options.caloClNxM,
        'targetsIdentifier' : '/Y_Identifier'+options.caloClNxM
    }


    XsToConcatenate = []
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
                    XsToConcatenate.append(np.load(indirs[i_fold]+readFrom['inputsCalibrator']+tag+'.npz', allow_pickle=True)['arr_0'])
                    YsToConcatenate.append(np.load(indirs[i_fold]+readFrom['targetsCalibrator']+tag+'.npz', allow_pickle=True)['arr_0'])

                elif options.doTens4Ident:
                    XsToConcatenate.append(np.load(indirs[i_fold]+readFrom['inputsIdentifier']+tag+'.npz', allow_pickle=True)['arr_0'])
                    YsToConcatenate.append(np.load(indirs[i_fold]+readFrom['targetsIdentifier']+tag+'.npz', allow_pickle=True)['arr_0'])


            except FileNotFoundError:
                # DEBUG
                print('** INFO: towers'+tag+' not found --> skipping')
                continue

    X = np.concatenate(XsToConcatenate)
    Y = np.concatenate(YsToConcatenate)

    ## DEBUG
    print(len(X))
    print(len(Y))

    if options.doTens4Calib:
        np.savez_compressed(outdir+'/X4Calibrator.npz', X)
        np.savez_compressed(outdir+'/Y4Calibrator.npz', Y)

    elif options.doTens4Ident:
        np.savez_compressed(outdir+'/X4Identifier.npz', X)
        np.savez_compressed(outdir+'/Y4Identifier.npz', Y)
