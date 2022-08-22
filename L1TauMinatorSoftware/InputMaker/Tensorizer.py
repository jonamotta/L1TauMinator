from optparse import OptionParser
import pandas as pd
import numpy as np
import math
import os


def TensorizeForIdentification(dfFlatTowClus, dfFlatGenTaus, dfFlatGenJets, uJetPtCut, lJetPtCut, uTauPtCut, lTauPtCut, etacut, NxM):
    if len(dfFlatTowClus) == 0:
        print('** WARNING : no data to be tensorized for identification here')
        return

    dfGenTaus = dfFlatGenTaus.copy(deep=True)
    dfGenJets = dfFlatGenJets.copy(deep=True)
    dfTowClus = dfFlatTowClus.copy(deep=True)

    # get clusters' shape dimensions
    N = int(NxM.split('x')[0])
    M = int(NxM.split('x')[1])

    # Select only hadronic taus
    dfGenTaus = dfGenTaus[dfGenTaus['tau_DM'] >= 0]

    # Apply cut on jet pt
    if uJetPtCut:
        dfGenJets = dfGenJets[dfGenJets['jet_pt'] <= float(uJetPtCut)]
    if lJetPtCut:
        dfGenJets = dfGenJets[dfGenJets['jet_pt'] >= float(lJetPtCut)]

    # Apply cut on tau pt
    if uTauPtCut:
        dfGenTaus = dfGenTaus[dfGenTaus['tau_pt'] <= float(uTauPtCut)]
    if lTauPtCut:
        dfGenTaus = dfGenTaus[dfGenTaus['tau_pt'] >= float(lTauPtCut)]

    # Apply cut on tau/jet eta
    if etacut:
        dfGenTaus = dfGenTaus[dfGenTaus['tau_eta'] <= float(etacut)]
        dfGenJets = dfGenJets[dfGenJets['jet_eta'] <= float(etacut)]

    # save unique identifier
    dfGenTaus['uniqueId'] = 'tau_'+dfGenTaus['event'].astype(str)+'_'+dfGenTaus['tau_Idx'].astype(str)
    dfGenJets['uniqueId'] = 'jet_'+dfGenJets['event'].astype(str)+'_'+dfGenJets['jet_Idx'].astype(str)
    dfCluPU  = dfTowClus[(dfTowClus['cl_tauMatchIdx']==-99) & (dfTowClus['cl_jetMatchIdx']==-99)].copy(deep=True)
    dfCluPU['uniqueId'] = 'pu_'+dfCluPU['event'].astype(str)+'_'+dfCluPU.index.astype(str)

    # join the taus and the clusters datasets -> this creates all the possible combination of clusters and jets/taus for each event
    # important that dfFlatET is joined to dfFlatEJ and not viceversa --> this because dfFlatEJ contains the safe jets to be used and the safe event numbers
    dfGenTaus.set_index('event', inplace=True)
    dfGenJets.set_index('event', inplace=True)
    dfTowClus.set_index('event', inplace=True)
    dfCluTau = dfGenTaus.join(dfTowClus, on='event', how='left', rsuffix='_joined', sort=False)
    dfCluJet = dfGenJets.join(dfTowClus, on='event', how='left', rsuffix='_joined', sort=False)

    # split dataframes between signal, qcd and pu
    features = ['uniqueId','cl_towerIeta','cl_towerIphi','cl_towerIem','cl_towerIhad','cl_towerEgIet'] #,'cl_towerNeg']
    dfCluTau = dfCluTau[dfCluTau['tau_Idx'] == dfCluTau['cl_tauMatchIdx']][features]
    dfCluJet = dfCluJet[dfCluJet['jet_Idx'] == dfCluJet['cl_jetMatchIdx']][features]
    dfCluPU = dfCluPU[features].copy(deep=True)

    # put ID label for signal and backgorund
    dfCluTau['targetId'] = 1
    dfCluJet['targetId'] = 0
    dfCluPU['targetId']  = 0

    # shuffle the rows so that no possible order gets learned
    dfCluTau = dfCluTau.sample(frac=1).copy(deep=True)
    dfCluJet = dfCluJet.sample(frac=1).copy(deep=True)
    dfCluPU  = dfCluPU.sample(frac=1).copy(deep=True)

    # get roughly the same amount of signal ad background to have a more balanced dataset
    # dfCluJet = dfCluJet.head(dfCluTau.shape[0])
    # dfCluPU  = dfCluPU.head(dfCluTau.shape[0])

    # concatenate and shuffle
    dfCluTauJetPu = pd.concat([dfCluTau, dfCluJet, dfCluPU], axis=0)
    dfCluTauJetPu  = dfCluTauJetPu.sample(frac=1).copy(deep=True)

    # make the input tensors for the neural network
    dfCluTauJetPu.set_index('uniqueId',inplace=True)

    XL = []
    YL = []
    for i, idx in enumerate(dfCluTauJetPu.index):
        # progress
        if i%100 == 0:
            print(i/len(dfCluTauJetPu.index)*100, '%')

        # for some reason some events have some problems with some barrel towers getting ieta=-1016 and iphi=-962 --> skip out-of-shape TowerClusters
        if len(dfCluTauJetPu.cl_towerIeta.loc[idx]) != N*M: continue

        # features for the NN
        xl = []
        for j in range(N*M):
            xl.append(dfCluTauJetPu.cl_towerIeta.loc[idx][j])
            xl.append(dfCluTauJetPu.cl_towerIphi.loc[idx][j])
            xl.append(dfCluTauJetPu.cl_towerIem.loc[idx][j])
            xl.append(dfCluTauJetPu.cl_towerIhad.loc[idx][j])
            xl.append(dfCluTauJetPu.cl_towerEgIet.loc[idx][j])
            # xl.append(dfCluTauJetPu.cl_towerNeg.loc[idx][j])
        x = np.array(xl).reshape(N,M,5)
        
        # target of the NN
        yl = []
        yl.append(dfCluTauJetPu.targetId.loc[idx])
        y = np.array(yl)

        # inputs to the NN
        XL.append(x)
        YL.append(y)

    # tensorize the lists
    X = np.array(XL)
    Y = np.array(YL)
    
    # save .npz files with tensor formatted datasets
    np.savez_compressed(saveTo['inputsIdentifier'], X)
    np.savez_compressed(saveTo['targetsIdentifier'], Y)


def TensorizeForCalibration(dfFlatTowClus, dfFlatGenTaus, uTauPtCut, lTauPtCut, etacut, NxM):
    if len(dfFlatTowClus) == 0 or len(dfFlatGenTaus) == 0:
        print('** WARNING : no data to be tensorized for calibration here')
        return

    dfGenTaus = dfFlatGenTaus.copy(deep=True)
    dfTowClus = dfFlatTowClus.copy(deep=True)

    # get clusters' shape dimensions
    N = int(NxM.split('x')[0])
    M = int(NxM.split('x')[1])

    # Select only hadronic taus
    dfGenTaus = dfGenTaus[dfGenTaus['tau_DM'] >= 0]

    # Apply cut on tau pt
    if uTauPtCut:
        dfGenTaus = dfGenTaus[dfGenTaus['tau_pt'] <= float(uTauPtCut)]
    if lTauPtCut:
        dfGenTaus = dfGenTaus[dfGenTaus['tau_pt'] >= float(lTauPtCut)]

    # Apply cut on tau eta
    if etacut:
        dfGenTaus = dfGenTaus[dfGenTaus['tau_eta'] <= float(etacut)]

    # transform pt in hardware units
    dfGenTaus['tau_hwPt'] = dfGenTaus['tau_pt'].copy(deep=True) * 2
    dfGenTaus['tau_hwVisPt'] = dfGenTaus['tau_visPt'].copy(deep=True) * 2
    dfGenTaus['tau_hwVisPtEm'] = dfGenTaus['tau_visPtEm'].copy(deep=True) * 2
    dfGenTaus['tau_hwVisPtHad'] = dfGenTaus['tau_visPtHad'].copy(deep=True) * 2

    # save unique identifier
    dfGenTaus['uniqueId'] = 'tau_'+dfGenTaus['event'].astype(str)+'_'+dfGenTaus['tau_Idx'].astype(str)

    # keep only the clusters that are matched to a tau
    dfTowClus = dfTowClus[dfTowClus['cl_tauMatchIdx'] >= 0]

    # join the taus and the clusters datasets -> this creates all the possible combination of clusters and taus for each event
    # important that dfFlatET is joined to dfFlatEJ and not viceversa --> this because dfFlatEJ contains the safe jets to be used and the safe event numbers
    dfGenTaus.set_index('event', inplace=True)
    dfTowClus.set_index('event', inplace=True)
    dfCluTau = dfGenTaus.join(dfTowClus, on='event', how='left', rsuffix='_joined', sort=False)

    # keep only the good matches between taus and clusters
    dfCluTau = dfCluTau[dfCluTau['tau_Idx'] == dfCluTau['cl_tauMatchIdx']]

    # shuffle the rows so that no possible order gets learned
    dfCluTau = dfCluTau.sample(frac=1).copy(deep=True)

    # make the input tensors for the neural network
    dfCluTau.set_index('uniqueId',inplace=True)
    XL = []
    YL = []
    for i, idx in enumerate(dfCluTau.index):
        # progress
        if i%100 == 0:
            print(i/len(dfCluTau.index)*100, '%')

        # for some reason some events have some problems with some barrel towers getting ieta=-1016 and iphi=-962 --> skip out-of-shape TowerClusters
        if len(dfCluTau.cl_towerIeta.loc[idx]) != N*M: continue

        # features for the NN
        xl = []
        for j in range(N*M):
            xl.append(dfCluTau.cl_towerIeta.loc[idx][j])
            xl.append(dfCluTau.cl_towerIphi.loc[idx][j])
            xl.append(dfCluTau.cl_towerIem.loc[idx][j])
            xl.append(dfCluTau.cl_towerIhad.loc[idx][j])
            xl.append(dfCluTau.cl_towerEgIet.loc[idx][j])
            # xl.append(dfCluTau.cl_towerNeg.loc[idx][j])
        x = np.array(xl).reshape(N,M,5)
        
        # targets of the NN
        yl = []
        yl.append(dfCluTau.tau_hwVisPt.loc[idx])
        yl.append(dfCluTau.tau_hwVisPtEm.loc[idx])
        yl.append(dfCluTau.tau_hwVisPtHad.loc[idx])
        y = np.array(yl)

        # inputs to the NN
        XL.append(x)
        YL.append(y)

    # tensorize the lists
    X = np.array(XL)
    Y = np.array(YL)

    # save .npz files with tensor formatted datasets
    np.savez_compressed(saveTo['inputsCalibrator'], X)
    np.savez_compressed(saveTo['targetsCalibrator'], Y)


#######################################################################
######################### SCRIPT BODY #################################
#######################################################################

if __name__ == "__main__" :

    parser = OptionParser()
    parser.add_option("--indir",        dest="indir",                             default=None)
    parser.add_option("--outdir",       dest="outdir",                            default=None)
    parser.add_option("--tag",          dest="tag",                               default=None)
    parser.add_option("--uJetPtCut",    dest="uJetPtCut",                         default=None)
    parser.add_option("--lJetPtCut",    dest="lJetPtCut",                         default=None)
    parser.add_option("--uTauPtCut",    dest="uTauPtCut",                         default=None)
    parser.add_option("--lTauPtCut",    dest="lTauPtCut",                         default=None)
    parser.add_option("--etacut",       dest="etacut",                            default=None)
    parser.add_option('--caloClNxM',    dest='caloClNxM',                         default=None)
    parser.add_option('--doTens4Calib', dest='doTens4Calib', action='store_true', default=False)
    parser.add_option('--doTens4Ident', dest='doTens4Ident', action='store_true', default=False)
    (options, args) = parser.parse_args()

    readFrom = {
        'TowClus' : options.indir+'/L1Clusters/TowClus'+options.caloClNxM+options.tag+'.pkl',
        'GenTaus' : options.indir+'/GenObjects/GenTaus'+options.tag+'.pkl',
        'GenJets' : options.indir+'/GenObjects/GenJets'+options.tag+'.pkl'
    }

    saveTo = {
        'inputsCalibrator'  : options.outdir+'/X_Calibrator'+options.caloClNxM+options.tag+'.npz',
        'inputsIdentifier'  : options.outdir+'/X_Identifier'+options.caloClNxM+options.tag+'.npz',
        'targetsCalibrator' : options.outdir+'/Y_Calibrator'+options.caloClNxM+options.tag+'.npz',
        'targetsIdentifier' : options.outdir+'/Y_Identifier'+options.caloClNxM+options.tag+'.npz'
    }

    dfTowClus = pd.read_pickle(readFrom['TowClus'])
    dfGenTaus = pd.read_pickle(readFrom['GenTaus'])
    dfGenJets = pd.read_pickle(readFrom['GenJets'])

    if not options.doTens4Calib and not options.doTens4Ident:
        print('** ERROR : no tensorization need specified')
        print('** EXITING')
        exit()

    if options.doTens4Calib:
        print('** INFO : doing tensorization for calibration')
        TensorizeForCalibration(dfTowClus, dfGenTaus, options.uTauPtCut, options.lTauPtCut, options.etacut, options.caloClNxM)
    if options.doTens4Ident:
        print('** INFO : doing tensorization for identification')
        TensorizeForIdentification(dfTowClus, dfGenTaus, dfGenJets, options.uJetPtCut, options.lJetPtCut, options.uTauPtCut, options.lTauPtCut, options.etacut, options.caloClNxM)

    print('** INFO : ALL DONE!')