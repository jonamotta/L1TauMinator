from optparse import OptionParser
from itertools import chain
import pandas as pd
import numpy as np
import argparse
import uproot3
import glob
import sys
import os

def inputQuantizer(inputE, inputPrecision):
    if inputPrecision: return min( np.floor(inputE/inputPrecision), 511 ) * inputPrecision
    else:              return inputE


def TensorizeForClNxMRate(dfFlatTowClus, uEtacut, lEtacut, NxM, inputPrecision):
    if len(dfFlatTowClus) == 0:
        print('** WARNING : no data to be tensorized for calibration here')
        return

    dfTowClus = dfFlatTowClus.copy(deep=True)

    # get clusters' shape dimensions
    N = int(NxM.split('x')[0])
    M = int(NxM.split('x')[1])

    # Apply cut on tau eta
    if uEtacut:
        dfTowClus = dfTowClus[abs(dfTowClus['cl_seedEta']) <= float(uEtacut)]
    if lEtacut:
        dfTowClus = dfTowClus[abs(dfTowClus['cl_seedEta']) >= float(lEtacut)]

    # save unique identifier
    dfTowClus['uniqueId'] = dfTowClus['event'].astype(str)+'_'+dfTowClus.index.astype(str)

    # shuffle the rows so that no possible order gets learned
    dfTowClus = dfTowClus.sample(frac=1).copy(deep=True)

    # make uniqueId the index
    dfTowClus.set_index('uniqueId',inplace=True)

    # make the input tensors for the neural network
    X1L = []
    X2L = []
    YL = []
    for i, idx in enumerate(dfTowClus.index):
        # progress
        if i%100 == 0:
            print(i/len(dfTowClus.index)*100, '%')

        # for some reason some events have some problems with some barrel towers getting ieta=-1016 and iphi=-962 --> skip out-of-shape TowerClusters
        if len(dfTowClus.cl_towerHad.loc[idx]) != N*M: continue

        # features of the Dense NN
        x2l = []
        x2l.append(dfTowClus.cl_seedEta.loc[idx])
        x2l.append(dfTowClus.cl_seedPhi.loc[idx])
        x2 = np.array(x2l)

        # features for the CNN
        x1l = []
        for j in range(N*M):
            x1l.append(inputQuantizer(dfTowClus.cl_towerEgEt.loc[idx][j], inputPrecision))
            x1l.append(inputQuantizer(dfTowClus.cl_towerEm.loc[idx][j], inputPrecision))
            x1l.append(inputQuantizer(dfTowClus.cl_towerHad.loc[idx][j], inputPrecision))
        x1 = np.array(x1l).reshape(N,M,3)
        
        # "targets" of the NN
        yl = []
        yl.append(dfTowClus.cl_seedEta.loc[idx])
        yl.append(dfTowClus.cl_seedPhi.loc[idx])
        yl.append(dfTowClus.event.loc[idx])
        y = np.array(yl)

        # inputs to the NN
        X1L.append(x1)
        X2L.append(x2)
        YL.append(y)

    # tensorize the lists
    X1 = np.array(X1L)
    X2 = np.array(X2L)
    Y = np.array(YL)

    print(X1.shape)
    print(X2.shape)
    print(Y.shape)

    # save .npz files with tensor formatted datasets
    np.savez_compressed(saveTensTo['inputsRateCNN'], X1)
    np.savez_compressed(saveTensTo['inputsRateDense'], X2)
    np.savez_compressed(saveTensTo['targetsRate'], Y)


def TensorizeForClNxMIdentification(dfFlatTowClus, dfFlatGenTaus, dfFlatGenJets, uJetPtCut, lJetPtCut, uTauPtCut, lTauPtCut, uEtacut, lEtacut, NxM, inputPrecision):
    if len(dfFlatTowClus) == 0:
        print('** WARNING : no data to be tensorized for identification here')
        return

    dfGenTaus = dfFlatGenTaus.copy(deep=True)
    dfGenJets = dfFlatGenJets.copy(deep=True)
    dfTowClus = dfFlatTowClus.copy(deep=True)

    # compute absolute eta of teh seeds
    dfTowClus['cl_absSeedIeta'] =  abs(dfTowClus['cl_seedIeta']).astype(int)

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
        dfGenTaus = dfGenTaus[dfGenTaus['tau_visPt'] <= float(uTauPtCut)]
    if lTauPtCut:
        dfGenTaus = dfGenTaus[dfGenTaus['tau_visPt'] >= float(lTauPtCut)]

    # Apply cut on tau eta
    if uEtacut:
        dfGenTaus = dfGenTaus[abs(dfGenTaus['tau_eta']) <= float(uEtacut)]
        dfGenJets = dfGenJets[abs(dfGenJets['jet_eta']) <= float(uEtacut)]
        dfTowClus = dfTowClus[abs(dfTowClus['cl_seedEta']) <= float(uEtacut)]
    if lEtacut:
        dfGenTaus = dfGenTaus[abs(dfGenTaus['tau_eta']) >= float(lEtacut)]
        dfGenJets = dfGenJets[abs(dfGenJets['jet_eta']) >= float(lEtacut)]
        dfTowClus = dfTowClus[abs(dfTowClus['cl_seedEta']) >= float(lEtacut)]

    # save unique identifier
    dfGenTaus['uniqueId'] = 'tau_'+dfGenTaus['event'].astype(str)+'_'+dfGenTaus['tau_Idx'].astype(str)
    dfGenJets['uniqueId'] = 'jet_'+dfGenJets['event'].astype(str)+'_'+dfGenJets['jet_Idx'].astype(str)
    dfCluPU  = dfTowClus[(dfTowClus['cl_tauMatchIdx']==-99) & (dfTowClus['cl_jetMatchIdx']==-99)].copy(deep=True)
    dfCluPU['uniqueId'] = 'pu_'+dfCluPU['event'].astype(str)+'_'+dfCluPU.index.astype(str)

    # join the taus and the clusters datasets -> this creates all the possible combination of clusters and jets/taus for each event
    # important that dfTowClus is joined to dfGen* and not viceversa --> this because dfGen* contains the safe jets to be used and the safe event numbers
    dfGenTaus.set_index('event', inplace=True)
    dfGenJets.set_index('event', inplace=True)
    dfTowClus.set_index('event', inplace=True)
    dfCluTau = dfGenTaus.join(dfTowClus, on='event', how='left', rsuffix='_joined', sort=False)
    dfCluJet = dfGenJets.join(dfTowClus, on='event', how='left', rsuffix='_joined', sort=False)

    # remove NaN entries due to missing tau/jet-clu matches
    dfCluTau.dropna(axis=0, how='any', inplace=True)
    dfCluJet.dropna(axis=0, how='any', inplace=True)

    # make sure these columns are ints and not floats
    dfCluTau[['cl_absSeedIeta', 'cl_seedIeta', 'cl_seedIphi']] = dfCluTau[['cl_absSeedIeta', 'cl_seedIeta', 'cl_seedIphi']].astype(int)
    dfCluJet[['cl_absSeedIeta', 'cl_seedIeta', 'cl_seedIphi']] = dfCluJet[['cl_absSeedIeta', 'cl_seedIeta', 'cl_seedIphi']].astype(int)

    # split dataframes between signal, qcd and pu
    # features = ['uniqueId', 'cl_absSeedIeta', 'cl_seedIphi', 'cl_towerEgEt', 'cl_towerEm', 'cl_towerHad']
    features = ['uniqueId', 'cl_seedEta', 'cl_seedPhi', 'cl_towerEgEt', 'cl_towerEm', 'cl_towerHad']
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
    if dfCluTau.shape[0] != 0:
        dfCluJet = dfCluJet.head(dfCluTau.shape[0])
        dfCluPU  = dfCluPU.head(dfCluTau.shape[0])

    # concatenate and shuffle
    dfCluTauJetPu = pd.concat([dfCluTau, dfCluJet, dfCluPU], axis=0)
    dfCluTauJetPu  = dfCluTauJetPu.sample(frac=1).copy(deep=True)

    # make uniqueId the index
    dfCluTauJetPu.set_index('uniqueId',inplace=True)

    # make the input tensors for the neural network
    X1L = []
    X2L = []
    YL = []
    for i, idx in enumerate(dfCluTauJetPu.index):
        # progress
        if i%100 == 0:
            print(i/len(dfCluTauJetPu.index)*100, '%')

        # for some reason some events have some problems with some barrel towers getting ieta=-1016 and iphi=-962 --> skip out-of-shape TowerClusters
        if len(dfCluTauJetPu.cl_towerHad.loc[idx]) != N*M: continue

        # features of the Dense NN
        x2l = []
        x2l.append(dfCluTauJetPu.cl_seedEta.loc[idx])
        x2l.append(dfCluTauJetPu.cl_seedPhi.loc[idx])
        x2 = np.array(x2l)

        # features for the CNN
        x1l = []
        for j in range(N*M):
            x1l.append(inputQuantizer(dfCluTauJetPu.cl_towerEgEt.loc[idx][j], inputPrecision))
            x1l.append(inputQuantizer(dfCluTauJetPu.cl_towerEm.loc[idx][j], inputPrecision))
            x1l.append(inputQuantizer(dfCluTauJetPu.cl_towerHad.loc[idx][j], inputPrecision))
        x1 = np.array(x1l).reshape(N,M,3)
        
        # target of the NN
        yl = []
        yl.append(dfCluTauJetPu.targetId.loc[idx])
        y = np.array(yl)

        # inputs to the NN
        X1L.append(x1)
        X2L.append(x2)
        YL.append(y)

    # tensorize the lists
    X1 = np.array(X1L)
    X2 = np.array(X2L)
    Y = np.array(YL)
    
    print(X1.shape)
    print(X2.shape)
    print(Y.shape)

    # save .npz files with tensor formatted datasets
    np.savez_compressed(saveTensTo['inputsIdentifierCNN'], X1)
    np.savez_compressed(saveTensTo['inputsIdentifierDense'], X2)
    np.savez_compressed(saveTensTo['targetsIdentifier'], Y)


def TensorizeForClNxMCoTraining(dfFlatTowClus, dfFlatGenTaus, dfFlatGenJets, uJetPtCut, lJetPtCut, uTauPtCut, lTauPtCut, uEtacut, lEtacut, NxM, inputPrecision):
    if len(dfFlatTowClus) == 0:
        print('** WARNING : no data to be tensorized for identification here')
        return

    dfGenTaus = dfFlatGenTaus.copy(deep=True)
    dfGenJets = dfFlatGenJets.copy(deep=True)
    dfTowClus = dfFlatTowClus.copy(deep=True)

    # compute absolute eta of teh seeds
    dfTowClus['cl_absSeedIeta'] =  abs(dfTowClus['cl_seedIeta']).astype(int)

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
        dfGenTaus = dfGenTaus[dfGenTaus['tau_visPt'] <= float(uTauPtCut)]
    if lTauPtCut:
        dfGenTaus = dfGenTaus[dfGenTaus['tau_visPt'] >= float(lTauPtCut)]

    # Apply cut on tau eta
    if uEtacut:
        dfGenTaus = dfGenTaus[abs(dfGenTaus['tau_eta']) <= float(uEtacut)]
        dfGenJets = dfGenJets[abs(dfGenJets['jet_eta']) <= float(uEtacut)]
        dfTowClus = dfTowClus[abs(dfTowClus['cl_seedEta']) <= float(uEtacut)]
    if lEtacut:
        dfGenTaus = dfGenTaus[abs(dfGenTaus['tau_eta']) >= float(lEtacut)]
        dfGenJets = dfGenJets[abs(dfGenJets['jet_eta']) >= float(lEtacut)]
        dfTowClus = dfTowClus[abs(dfTowClus['cl_seedEta']) >= float(lEtacut)]

    # save unique identifier
    dfGenTaus['uniqueId'] = 'tau_'+dfGenTaus['event'].astype(str)+'_'+dfGenTaus['tau_Idx'].astype(str)
    dfGenJets['uniqueId'] = 'jet_'+dfGenJets['event'].astype(str)+'_'+dfGenJets['jet_Idx'].astype(str)
    dfCluPU  = dfTowClus[(dfTowClus['cl_tauMatchIdx']==-99) & (dfTowClus['cl_jetMatchIdx']==-99)].copy(deep=True)
    dfCluPU['uniqueId'] = 'pu_'+dfCluPU['event'].astype(str)+'_'+dfCluPU.index.astype(str)

    # join the taus and the clusters datasets -> this creates all the possible combination of clusters and jets/taus for each event
    # important that dfTowClus is joined to dfGen* and not viceversa --> this because dfGen* contains the safe jets to be used and the safe event numbers
    dfGenTaus.set_index('event', inplace=True)
    dfGenJets.set_index('event', inplace=True)
    dfTowClus.set_index('event', inplace=True)
    dfCluTau = dfGenTaus.join(dfTowClus, on='event', how='left', rsuffix='_joined', sort=False)
    dfCluJet = dfGenJets.join(dfTowClus, on='event', how='left', rsuffix='_joined', sort=False)

    # remove NaN entries due to missing tau/jet-clu matches
    dfCluTau.dropna(axis=0, how='any', inplace=True)
    dfCluJet.dropna(axis=0, how='any', inplace=True)

    # create fake taus columns to facilitate concatenation and target tensorization
    dfCluPU['tau_visPt'] = -99.9 ; dfCluPU['tau_visEta'] = -99.9 ; dfCluPU['tau_visPhi'] = -99.9 ; dfCluPU['tau_DM'] = -99.9
    dfCluJet['tau_visPt'] = -99.9 ; dfCluJet['tau_visEta'] = -99.9 ; dfCluJet['tau_visPhi'] = -99.9 ; dfCluJet['tau_DM'] = -99.9

    # make sure these columns are ints and not floats
    dfCluTau[['cl_absSeedIeta', 'cl_seedIeta', 'cl_seedIphi']] = dfCluTau[['cl_absSeedIeta', 'cl_seedIeta', 'cl_seedIphi']].astype(int)
    dfCluJet[['cl_absSeedIeta', 'cl_seedIeta', 'cl_seedIphi']] = dfCluJet[['cl_absSeedIeta', 'cl_seedIeta', 'cl_seedIphi']].astype(int)

    # split dataframes between signal, qcd and pu
    # features = ['uniqueId', 'cl_absSeedIeta', 'cl_seedIphi', 'cl_towerEgEt', 'cl_towerEm', 'cl_towerHad']
    dfCluTau.reset_index(inplace=True)
    dfCluJet.reset_index(inplace=True)
    features = ['uniqueId', 'event', 'cl_seedEta', 'cl_seedPhi', 'cl_towerEgEt', 'cl_towerEm', 'cl_towerHad', 'tau_visPt', 'tau_visEta', 'tau_visPhi', 'tau_DM']
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
    if dfCluTau.shape[0] != 0:
        dfCluJet = dfCluJet.head(dfCluTau.shape[0])
        dfCluPU  = dfCluPU.head(dfCluTau.shape[0])

    # concatenate and shuffle
    dfCluTauJetPu = pd.concat([dfCluTau, dfCluJet, dfCluPU], axis=0)
    dfCluTauJetPu  = dfCluTauJetPu.sample(frac=1).copy(deep=True)

    # make uniqueId the index
    dfCluTauJetPu.set_index('uniqueId',inplace=True)

    # make the input tensors for the neural network
    X1L = []
    X2L = []
    YL = []
    for i, idx in enumerate(dfCluTauJetPu.index):
        # progress
        if i%100 == 0:
            print(i/len(dfCluTauJetPu.index)*100, '%')

        # for some reason some events have some problems with some barrel towers getting ieta=-1016 and iphi=-962 --> skip out-of-shape TowerClusters
        if len(dfCluTauJetPu.cl_towerHad.loc[idx]) != N*M: continue

        # features of the Dense NN
        x2l = []
        x2l.append(dfCluTauJetPu.cl_seedEta.loc[idx])
        x2l.append(dfCluTauJetPu.cl_seedPhi.loc[idx])
        x2 = np.array(x2l)

        # features for the CNN
        x1l = []
        for j in range(N*M):
            x1l.append(inputQuantizer(dfCluTauJetPu.cl_towerEgEt.loc[idx][j], inputPrecision))
            x1l.append(inputQuantizer(dfCluTauJetPu.cl_towerEm.loc[idx][j], inputPrecision))
            x1l.append(inputQuantizer(dfCluTauJetPu.cl_towerHad.loc[idx][j], inputPrecision))
        x1 = np.array(x1l).reshape(N,M,3)
        
        # target of the NN
        yl = []
        yl.append(dfCluTauJetPu.targetId.loc[idx])
        yl.append(dfCluTauJetPu.tau_visPt.loc[idx])
        yl.append(dfCluTauJetPu.tau_visEta.loc[idx])
        yl.append(dfCluTauJetPu.tau_visPhi.loc[idx])
        yl.append(dfCluTauJetPu.tau_DM.loc[idx])
        yl.append(dfCluTauJetPu.event.loc[idx])
        y = np.array(yl)

        # inputs to the NN
        X1L.append(x1)
        X2L.append(x2)
        YL.append(y)

    # tensorize the lists
    X1 = np.array(X1L)
    X2 = np.array(X2L)
    Y = np.array(YL)
    
    print(X1.shape)
    print(X2.shape)
    print(Y.shape)

    # save .npz files with tensor formatted datasets
    np.savez_compressed(saveTensTo['inputsCoTrainingCNN'], X1)
    np.savez_compressed(saveTensTo['inputsCoTrainingDense'], X2)
    np.savez_compressed(saveTensTo['targetsCoTraining'], Y)


def TensorizeForClNxMCalibration(dfFlatTowClus, dfFlatGenTaus, uTauPtCut, lTauPtCut, uEtacut, lEtacut, NxM, inputPrecision):
    if len(dfFlatTowClus) == 0 or len(dfFlatGenTaus) == 0:
        print('** WARNING : no data to be tensorized for calibration here')
        return

    dfGenTaus = dfFlatGenTaus.copy(deep=True)
    dfTowClus = dfFlatTowClus.copy(deep=True)

    # compute absolute eta of the seeds
    dfTowClus['cl_absSeedIeta'] =  abs(dfTowClus['cl_seedIeta']).astype(int)

    # get clusters' shape dimensions
    N = int(NxM.split('x')[0])
    M = int(NxM.split('x')[1])

    # Select only hadronic taus
    dfGenTaus = dfGenTaus[dfGenTaus['tau_DM'] >= 0]

    # Apply cut on tau pt
    if uTauPtCut:
        dfGenTaus = dfGenTaus[dfGenTaus['tau_visPt'] <= float(uTauPtCut)]
    if lTauPtCut:
        dfGenTaus = dfGenTaus[dfGenTaus['tau_visPt'] >= float(lTauPtCut)]

    # Apply cut on tau eta
    if uEtacut:
        dfGenTaus = dfGenTaus[abs(dfGenTaus['tau_eta']) <= float(uEtacut)]
        dfTowClus = dfTowClus[abs(dfTowClus['cl_seedEta']) <= float(uEtacut)]
    if lEtacut:
        dfGenTaus = dfGenTaus[abs(dfGenTaus['tau_eta']) >= float(lEtacut)]
        dfTowClus = dfTowClus[abs(dfTowClus['cl_seedEta']) >= float(lEtacut)]

    # save unique identifier
    dfGenTaus['uniqueId'] = 'tau_'+dfGenTaus['event'].astype(str)+'_'+dfGenTaus['tau_Idx'].astype(str)

    # keep only the clusters that are matched to a tau
    dfTowClus = dfTowClus[dfTowClus['cl_tauMatchIdx'] >= 0]

    # join the taus and the clusters datasets -> this creates all the possible combination of clusters and taus for each event
    # important that dfTowClus is joined to dfGenTaus and not viceversa --> this because dfGenTaus contains the safe jets to be used and the safe event numbers
    dfGenTaus.set_index('event', inplace=True)
    dfTowClus.set_index('event', inplace=True)
    dfCluTau = dfGenTaus.join(dfTowClus, on='event', how='left', rsuffix='_joined', sort=False)

    # remove NaN entries due to missing tau/jet-clu matches
    dfCluTau.dropna(axis=0, how='any', inplace=True)

    # make sure these columns are ints and not floats
    dfCluTau[['cl_absSeedIeta', 'cl_seedIeta', 'cl_seedIphi']] = dfCluTau[['cl_absSeedIeta', 'cl_seedIeta', 'cl_seedIphi']].astype(int)

    # keep only the good matches between taus and clusters
    dfCluTau = dfCluTau[dfCluTau['tau_Idx'] == dfCluTau['cl_tauMatchIdx']]

    # shuffle the rows so that no possible order gets learned
    dfCluTau = dfCluTau.sample(frac=1).copy(deep=True)

    # make uniqueId the index
    dfCluTau.reset_index(inplace=True)
    dfCluTau.set_index('uniqueId', inplace=True)

    # make the input tensors for the neural network
    X1L = []
    X2L = []
    YL = []
    for i, idx in enumerate(dfCluTau.index):
        # progress
        if i%100 == 0:
            print(i/len(dfCluTau.index)*100, '%')

        # for some reason some events have some problems with some barrel towers getting ieta=-1016 and iphi=-962 --> skip out-of-shape TowerClusters
        if len(dfCluTau.cl_towerHad.loc[idx]) != N*M: continue

        # features of the Dense NN
        x2l = []
        x2l.append(dfCluTau.cl_seedEta.loc[idx])
        x2l.append(dfCluTau.cl_seedPhi.loc[idx])
        x2 = np.array(x2l)

        # features for the CNN
        x1l = []
        for j in range(N*M):
            x1l.append(inputQuantizer(dfCluTau.cl_towerEgEt.loc[idx][j], inputPrecision))
            x1l.append(inputQuantizer(dfCluTau.cl_towerEm.loc[idx][j], inputPrecision))
            x1l.append(inputQuantizer(dfCluTau.cl_towerHad.loc[idx][j], inputPrecision))
        x1 = np.array(x1l).reshape(N,M,3)
        
        # targets of the NN
        yl = []
        yl.append(dfCluTau.tau_visPt.loc[idx])
        yl.append(dfCluTau.tau_visEta.loc[idx])
        yl.append(dfCluTau.tau_visPhi.loc[idx])
        yl.append(dfCluTau.tau_DM.loc[idx])
        yl.append(dfCluTau.event.loc[idx])
        y = np.array(yl)

        # inputs to the NN
        X1L.append(x1)
        X2L.append(x2)
        YL.append(y)

    # tensorize the lists
    X1 = np.array(X1L)
    X2 = np.array(X2L)
    Y = np.array(YL)

    print(X1.shape)
    print(X2.shape)
    print(Y.shape)

    # save .npz files with tensor formatted datasets
    np.savez_compressed(saveTensTo['inputsCalibratorCNN'], X1)
    np.savez_compressed(saveTensTo['inputsCalibratorDense'], X2)
    np.savez_compressed(saveTensTo['targetsCalibrator'], Y)


def TensorizeForCl3dRate(dfFlatHGClus, uEtacut, lEtacut, inputPrecision):
    if len(dfFlatHGClus) == 0:
        print('** WARNING : no data to be tensorized for calibration here')
        return

    dfHGClus = dfFlatHGClus.copy(deep=True)

    # Apply cut on tau eta
    if uEtacut:
        dfHGClus = dfHGClus[abs(dfHGClus['cl3d_eta']) <= float(uEtacut)]
    if lEtacut:
        dfHGClus = dfHGClus[abs(dfHGClus['cl3d_eta']) >= float(lEtacut)]

    # save unique identifier
    dfHGClus['uniqueId'] = dfHGClus['event'].astype(str)+'_'+dfHGClus.index.astype(str)

    # shuffle the rows so that no possible order gets learned
    dfHGClus = dfHGClus.sample(frac=1).copy(deep=True)

    # make uniqueId the index
    dfHGClus.set_index('uniqueId',inplace=True)

    # save .pkl file with formatted datasets
    dfHGClus.to_pickle(saveTensTo['inputsRateBDT'])


def TensorizeForCl3dIdentification(dfFlatHGClus, dfFlatGenTaus, dfFlatGenJets, uJetPtCut, lJetPtCut, uTauPtCut, lTauPtCut, uEtacut, lEtacut, inputPrecision):
    if len(dfFlatHGClus) == 0:
        print('** WARNING : no data to be tensorized for identification here')
        return

    dfGenTaus = dfFlatGenTaus.copy(deep=True)
    dfGenJets = dfFlatGenJets.copy(deep=True)
    dfHGClus = dfFlatHGClus.copy(deep=True)

    # Select only hadronic taus
    dfGenTaus = dfGenTaus[dfGenTaus['tau_DM'] >= 0]

    # Apply cut on jet pt
    if uJetPtCut:
        dfGenJets = dfGenJets[dfGenJets['jet_pt'] <= float(uJetPtCut)]
    if lJetPtCut:
        dfGenJets = dfGenJets[dfGenJets['jet_pt'] >= float(lJetPtCut)]

    # Apply cut on tau pt
    if uTauPtCut:
        dfGenTaus = dfGenTaus[dfGenTaus['tau_visPt'] <= float(uTauPtCut)]
    if lTauPtCut:
        dfGenTaus = dfGenTaus[dfGenTaus['tau_visPt'] >= float(lTauPtCut)]

    # Apply cut on tau eta
    if uEtacut:
        dfGenTaus = dfGenTaus[abs(dfGenTaus['tau_eta']) <= float(uEtacut)]
        dfGenJets = dfGenJets[abs(dfGenJets['jet_eta']) <= float(uEtacut)]
        dfHGClus  = dfHGClus[abs(dfHGClus['cl3d_eta']) <= float(uEtacut)]
    if lEtacut:
        dfGenTaus = dfGenTaus[abs(dfGenTaus['tau_eta']) >= float(lEtacut)]
        dfGenJets = dfGenJets[abs(dfGenJets['jet_eta']) >= float(lEtacut)]
        dfHGClus  = dfHGClus[abs(dfHGClus['cl3d_eta']) >= float(lEtacut)]

    # save unique identifier
    dfGenTaus['uniqueId'] = 'tau_'+dfGenTaus['event'].astype(str)+'_'+dfGenTaus['tau_Idx'].astype(str)
    dfGenJets['uniqueId'] = 'jet_'+dfGenJets['event'].astype(str)+'_'+dfGenJets['jet_Idx'].astype(str)
    dfCluPU  = dfHGClus[(dfHGClus['cl3d_tauMatchIdx']==-99) & (dfHGClus['cl3d_jetMatchIdx']==-99)].copy(deep=True)
    dfCluPU['uniqueId'] = 'pu_'+dfCluPU['event'].astype(str)+'_'+dfCluPU.index.astype(str)

    # join the taus and the clusters datasets -> this creates all the possible combination of clusters and jets/taus for each event
    # important that dfHGClus is joined to dfGen* and not viceversa --> this because dfGen* contains the safe jets to be used and the safe event numbers
    dfGenTaus.set_index('event', inplace=True)
    dfGenJets.set_index('event', inplace=True)
    dfHGClus.set_index('event', inplace=True)
    dfCluTau = dfGenTaus.join(dfHGClus, on='event', how='left', rsuffix='_joined', sort=False)
    dfCluJet = dfGenJets.join(dfHGClus, on='event', how='left', rsuffix='_joined', sort=False)

    # remove NaN entries due to missing tau/jet-clu matches
    dfCluTau.dropna(axis=0, how='any', inplace=True)
    dfCluJet.dropna(axis=0, how='any', inplace=True)

    # split dataframes between signal, qcd and pu
    dfCluTau = dfCluTau[dfCluTau['tau_Idx'] == dfCluTau['cl3d_tauMatchIdx']]
    dfCluJet = dfCluJet[dfCluJet['jet_Idx'] == dfCluJet['cl3d_jetMatchIdx']]

    # put ID label for signal and backgorund
    dfCluTau['targetId'] = 1
    dfCluJet['targetId'] = 0
    dfCluPU['targetId']  = 0

    # shuffle the rows so that no possible order gets learned
    dfCluTau = dfCluTau.sample(frac=1).copy(deep=True)
    dfCluJet = dfCluJet.sample(frac=1).copy(deep=True)
    dfCluPU  = dfCluPU.sample(frac=1).copy(deep=True)

    # get roughly the same amount of signal ad background to have a more balanced dataset
    if dfCluTau.shape[0] != 0:
        dfCluJet = dfCluJet.head(dfCluTau.shape[0])
        dfCluPU  = dfCluPU.head(dfCluTau.shape[0]*2)

    # concatenate and shuffle
    dfCluTauJetPu = pd.concat([dfCluTau, dfCluJet, dfCluPU], axis=0)
    dfCluTauJetPu  = dfCluTauJetPu.sample(frac=1).copy(deep=True)

    # make uniqueId the index
    dfCluTauJetPu.set_index('uniqueId',inplace=True)

    # save .pkl file with formatted datasets
    dfCluTauJetPu.to_pickle(saveTensTo['inputsIdentifierBDT'])


def TensorizeForCl3dCalibration(dfFlatHGClus, dfFlatGenTaus, uTauPtCut, lTauPtCut, uEtacut, lEtacut, inputPrecision):
    if len(dfFlatHGClus) == 0 or len(dfFlatGenTaus) == 0:
        print('** WARNING : no data to be tensorized for calibration here')
        return

    dfGenTaus = dfFlatGenTaus.copy(deep=True)
    dfHGClus  = dfFlatHGClus.copy(deep=True)

    # Select only hadronic taus
    dfGenTaus = dfGenTaus[dfGenTaus['tau_DM'] >= 0]

    # Apply cut on tau pt
    if uTauPtCut:
        dfGenTaus = dfGenTaus[dfGenTaus['tau_visPt'] <= float(uTauPtCut)]
    if lTauPtCut:
        dfGenTaus = dfGenTaus[dfGenTaus['tau_visPt'] >= float(lTauPtCut)]

    # Apply cut on tau eta
    if uEtacut:
        dfGenTaus = dfGenTaus[abs(dfGenTaus['tau_eta']) <= float(uEtacut)]
        dfHGClus  = dfHGClus[abs(dfHGClus['cl3d_eta']) <= float(uEtacut)]
    if lEtacut:
        dfGenTaus = dfGenTaus[abs(dfGenTaus['tau_eta']) >= float(lEtacut)]
        dfHGClus  = dfHGClus[abs(dfHGClus['cl3d_eta']) >= float(lEtacut)]

    # save unique identifier
    dfGenTaus['uniqueId'] = 'tau_'+dfGenTaus['event'].astype(str)+'_'+dfGenTaus['tau_Idx'].astype(str)

    # keep only the clusters that are matched to a tau
    dfHGClus = dfHGClus[dfHGClus['cl3d_tauMatchIdx'] >= 0]

    # join the taus and the clusters datasets -> this creates all the possible combination of clusters and taus for each event
    # important that dfHGClus is joined to dfGenTaus and not viceversa --> this because dfGenTaus contains the safe jets to be used and the safe event numbers
    dfGenTaus.set_index('event', inplace=True)
    dfHGClus.set_index('event', inplace=True)
    dfCluTau = dfGenTaus.join(dfHGClus, on='event', how='left', rsuffix='_joined', sort=False)

    # remove NaN entries due to missing tau/jet-clu matches
    dfCluTau.dropna(axis=0, how='any', inplace=True)

    # keep only the good matches between taus and clusters
    dfCluTau = dfCluTau[dfCluTau['tau_Idx'] == dfCluTau['cl3d_tauMatchIdx']]

    # shuffle the rows so that no possible order gets learned
    dfCluTau = dfCluTau.sample(frac=1).copy(deep=True)

    # make uniqueId the index
    dfCluTau.reset_index(inplace=True)
    dfCluTau.set_index('uniqueId',inplace=True)

    # save .pkl file with formatted datasets
    dfCluTau.to_pickle(saveTensTo['inputsCalibratorBDT'])


def TensorizeForTauMinatorPerformance(dfFlatTowClus, dfFlatHGClus, dfFlatGenTaus, uTauPtCut, lTauPtCut, uEtacut, lEtacut,  NxM, inputPrecision):
    if len(dfFlatTowClus) == 0 or len(dfFlatHGClus) == 0 or len(dfFlatGenTaus) == 0:
        print('** WARNING : no data to be tensorized for calibration here')
        return

    dfGenTaus = dfFlatGenTaus.copy(deep=True)
    dfTowClus = dfFlatTowClus.copy(deep=True)
    dfHGClus  = dfFlatHGClus.copy(deep=True)

    # compute absolute eta of the seeds
    dfTowClus['cl_absSeedIeta'] =  abs(dfTowClus['cl_seedIeta']).astype(int)

    # get clusters' shape dimensions
    N = int(NxM.split('x')[0])
    M = int(NxM.split('x')[1])

    # Apply cut on tau pt
    if uTauPtCut:
        dfGenTaus = dfGenTaus[dfGenTaus['tau_visPt'] <= float(uTauPtCut)]
    if lTauPtCut:
        dfGenTaus = dfGenTaus[dfGenTaus['tau_visPt'] >= float(lTauPtCut)]

    # Apply cut on tau eta
    if uEtacut:
        dfGenTaus = dfGenTaus[abs(dfGenTaus['tau_eta']) <= float(uEtacut)]
        dfTowClus = dfTowClus[abs(dfTowClus['cl_seedEta']) <= float(uEtacut)]
        dfHGClus  = dfHGClus[abs(dfHGClus['cl3d_eta']) <= float(uEtacut)]
    if lEtacut:
        dfGenTaus = dfGenTaus[abs(dfGenTaus['tau_eta']) >= float(lEtacut)]
        dfTowClus = dfTowClus[abs(dfTowClus['cl_seedEta']) >= float(lEtacut)]
        dfHGClus  = dfHGClus[abs(dfHGClus['cl3d_eta']) >= float(lEtacut)]

    # save unique identifier
    dfGenTaus['uniqueId'] = 'tau_'+dfGenTaus['event'].astype(str)+'_'+dfGenTaus['tau_Idx'].astype(str)

    # keep only the clusters that are matched to a tau
    dfTowClus = dfTowClus[dfTowClus['cl_tauMatchIdx'] >= 0]
    dfHGClus = dfHGClus[dfHGClus['cl3d_tauMatchIdx'] >= 0]

    # join the taus and the clusters datasets -> this creates all the possible combination of clusters and taus for each event
    # important that dfHGClus is joined to dfGenTaus and not viceversa --> this because dfGenTaus contains the safe jets to be used and the safe event numbers
    dfGenTaus.set_index('event', inplace=True)
    dfTowClus.set_index('event', inplace=True)
    dfHGClus.set_index('event', inplace=True)
    dfCLTWTau = dfGenTaus.join(dfTowClus, on='event', how='left', rsuffix='_joined', sort=False)
    dfCL3DTau = dfGenTaus.join(dfHGClus, on='event', how='left', rsuffix='_joined', sort=False)

    # remove NaN entries due to missing tau/jet-clu matches
    dfCLTWTau.dropna(axis=0, how='any', inplace=True)
    dfCL3DTau.dropna(axis=0, how='any', inplace=True)

    # keep only the good matches between taus and clusters
    dfCLTWTau = dfCLTWTau[dfCLTWTau['tau_Idx'] == dfCLTWTau['cl_tauMatchIdx']]
    dfCL3DTau = dfCL3DTau[dfCL3DTau['tau_Idx'] == dfCL3DTau['cl3d_tauMatchIdx']]

    # make uniqueId the index
    dfCLTWTau.reset_index(inplace=True)
    dfCL3DTau.reset_index(inplace=True)
    dfCLTWTau.set_index('uniqueId',inplace=True)
    dfCL3DTau.set_index('uniqueId',inplace=True)

    # make the input tensors for the neural network
    X1L = []
    X2L = []
    YL = []
    for i, idx in enumerate(dfCLTWTau.index):
        # progress
        if i%100 == 0:
            print(i/len(dfCLTWTau.index)*100, '%')

        # for some reason some events have some problems with some barrel towers getting ieta=-1016 and iphi=-962 --> skip out-of-shape TowerClusters
        if len(dfCLTWTau.cl_towerHad.loc[idx]) != N*M: continue

        # features of the Dense NN
        x2l = []
        x2l.append(dfCLTWTau.cl_seedEta.loc[idx])
        x2l.append(dfCLTWTau.cl_seedPhi.loc[idx])
        x2 = np.array(x2l)

        # features for the CNN
        x1l = []
        for j in range(N*M):
            x1l.append(inputQuantizer(dfCLTWTau.cl_towerEgEt.loc[idx][j], inputPrecision))
            x1l.append(inputQuantizer(dfCLTWTau.cl_towerEm.loc[idx][j], inputPrecision))
            x1l.append(inputQuantizer(dfCLTWTau.cl_towerHad.loc[idx][j], inputPrecision))
        x1 = np.array(x1l).reshape(N,M,3)
        
        # targets of the NN
        yl = []
        yl.append(dfCLTWTau.tau_visPt.loc[idx])
        yl.append(dfCLTWTau.tau_visEta.loc[idx])
        yl.append(dfCLTWTau.tau_visPhi.loc[idx])
        yl.append(dfCLTWTau.tau_DM.loc[idx])
        yl.append(dfCLTWTau.event.loc[idx])
        y = np.array(yl)

        # inputs to the NN
        X1L.append(x1)
        X2L.append(x2)
        YL.append(y)

    # tensorize the lists
    X1 = np.array(X1L)
    X2 = np.array(X2L)
    Y = np.array(YL)

    print(X1.shape)
    print(X2.shape)
    print(Y.shape)

    # save .npz files with tensor formatted datasets
    np.savez_compressed(saveTensTo['inputsMinatorCNN'], X1)
    np.savez_compressed(saveTensTo['inputsMinatorDense'], X2)
    np.savez_compressed(saveTensTo['targetsMinator'], Y)

    # save .pkl file with formatted datasets
    dfCL3DTau.to_pickle(saveTensTo['inputsMinatorBDT'])


def TensorizeForTauMinatorRate(dfFlatTowClus, dfFlatHGClus, uEtacut, lEtacut,  NxM, inputPrecision):
    if len(dfFlatTowClus) == 0 or len(dfFlatHGClus) == 0:
        print('** WARNING : no data to be tensorized for calibration here')
        return

    dfTowClus = dfFlatTowClus.copy(deep=True)
    dfHGClus = dfFlatHGClus.copy(deep=True)

    # get clusters' shape dimensions
    N = int(NxM.split('x')[0])
    M = int(NxM.split('x')[1])

    # Apply cut on tau eta
    if uEtacut:
        dfTowClus = dfTowClus[abs(dfTowClus['cl_seedEta']) <= float(uEtacut)]
        dfHGClus = dfHGClus[abs(dfHGClus['cl3d_eta']) <= float(uEtacut)]
    if lEtacut:
        dfTowClus = dfTowClus[abs(dfTowClus['cl_seedEta']) >= float(lEtacut)]
        dfHGClus = dfHGClus[abs(dfHGClus['cl3d_eta']) >= float(lEtacut)]

    # save unique identifier
    dfTowClus['uniqueId'] = dfTowClus['event'].astype(str)+'_'+dfTowClus.index.astype(str)
    dfHGClus['uniqueId'] = dfHGClus['event'].astype(str)+'_'+dfHGClus.index.astype(str)

    # make uniqueId the index
    dfTowClus.set_index('uniqueId',inplace=True)
    dfHGClus.set_index('uniqueId',inplace=True)

    # make the input tensors for the neural network
    X1L = []
    X2L = []
    YL = []
    for i, idx in enumerate(dfTowClus.index):
        # progress
        if i%100 == 0:
            print(i/len(dfTowClus.index)*100, '%')

        # for some reason some events have some problems with some barrel towers getting ieta=-1016 and iphi=-962 --> skip out-of-shape TowerClusters
        if len(dfTowClus.cl_towerHad.loc[idx]) != N*M: continue

        # features of the Dense NN
        x2l = []
        x2l.append(dfTowClus.cl_seedEta.loc[idx])
        x2l.append(dfTowClus.cl_seedPhi.loc[idx])
        x2 = np.array(x2l)

        # features for the CNN
        x1l = []
        for j in range(N*M):
            x1l.append(inputQuantizer(dfTowClus.cl_towerEgEt.loc[idx][j], inputPrecision))
            x1l.append(inputQuantizer(dfTowClus.cl_towerEm.loc[idx][j], inputPrecision))
            x1l.append(inputQuantizer(dfTowClus.cl_towerHad.loc[idx][j], inputPrecision))
        x1 = np.array(x1l).reshape(N,M,3)
        
        # "targets" of the NN
        yl = []
        yl.append(dfTowClus.cl_seedEta.loc[idx])
        yl.append(dfTowClus.cl_seedPhi.loc[idx])
        yl.append(dfTowClus.event.loc[idx])
        y = np.array(yl)

        # inputs to the NN
        X1L.append(x1)
        X2L.append(x2)
        YL.append(y)

    # tensorize the lists
    X1 = np.array(X1L)
    X2 = np.array(X2L)
    Y = np.array(YL)

    print(X1.shape)
    print(X2.shape)
    print(Y.shape)

    # save .npz files with tensor formatted datasets
    np.savez_compressed(saveTensTo['inputsMinatorRateCNN'], X1)
    np.savez_compressed(saveTensTo['inputsMinatorRateDense'], X2)
    np.savez_compressed(saveTensTo['targetsMinatorRate'], Y)

    # save .pkl file with formatted datasets
    dfHGClus.to_pickle(saveTensTo['inputsMinatorRateBDT'])

#######################################################################
######################### SCRIPT BODY #################################
#######################################################################

if __name__ == "__main__" :

    parser = OptionParser()
    # GENERAL OPTIONS
    parser.add_option("--infile",         dest="infile",                                                                               default=None)
    parser.add_option("--outdir",         dest="outdir",                                                                               default=None)
    parser.add_option('--caloClNxM',      dest='caloClNxM',    help='Which shape of CaloCluster to use?',                              default="5x9")
    # INPUTS PREPARATION OPTIONS
    parser.add_option('--doHGCAL',        dest='doHGCAL',      help='Do HGCAL inputs preparation?',               action='store_true', default=False)
    parser.add_option('--doCALO',         dest='doCALO',       help='Do CALO inputs preparation?',                action='store_true', default=False)
    # TTREE READING OPTIONS
    parser.add_option('--doHH',           dest='doHH',         help='Read the HH samples?',                       action='store_true', default=False)
    parser.add_option('--doQCD',          dest='doQCD',        help='Read the QCD samples?',                      action='store_true', default=False)
    parser.add_option('--doVBFH',         dest='doVBFH',       help='Read the VBF H samples?',                    action='store_true', default=False)
    parser.add_option('--doMinBias',      dest='doMinBias',    help='Read the Minbias samples?',                  action='store_true', default=False)
    parser.add_option('--doZp500',        dest='doZp500',      help='Read the Minbias samples?',                  action='store_true', default=False)
    parser.add_option('--doZp1500',       dest='doZp1500',     help='Read the Minbias samples?',                  action='store_true', default=False)
    parser.add_option('--doTestRun',      dest='doTestRun',    help='Do test run with reduced number of events?', action='store_true', default=False)
    # TENSORIZATION OPTIONS
    parser.add_option("--infileTag",      dest="infileTag",                           default=None)
    parser.add_option("--outTag",         dest="outTag",                              default="")
    parser.add_option("--inputPrecision", dest="inputPrecision", type=float,          default=None)
    parser.add_option("--uJetPtCut",      dest="uJetPtCut",                           default=None)
    parser.add_option("--lJetPtCut",      dest="lJetPtCut",                           default=None)
    parser.add_option("--uTauPtCut",      dest="uTauPtCut",                           default=None)
    parser.add_option("--lTauPtCut",      dest="lTauPtCut",                           default=None)
    parser.add_option("--uEtacut",        dest="uEtacut",                             default=None)
    parser.add_option("--lEtacut",        dest="lEtacut",                             default=None)
    parser.add_option('--doTens4Calib',   dest='doTens4Calib',   action='store_true', default=False)
    parser.add_option('--doTens4Ident',   dest='doTens4Ident',   action='store_true', default=False)
    parser.add_option('--doTens4Minator', dest='doTens4Minator', action='store_true', default=False)
    parser.add_option('--doTens4Rate',    dest='doTens4Rate',    action='store_true', default=False)
    parser.add_option('--doTens4Cotraining', dest='doTens4Cotraining', action='store_true', default=False)
    (options, args) = parser.parse_args()

    print(options)

    ##################### DEFINE INPUTS AND OUTPUTS ####################
    Infile  = options.infile

    key = 'Ntuplizer/L1TauMinatorTree'
    branches_event  = ['EventNumber']
    branches_gentau = ['tau_Idx', 'tau_eta', 'tau_phi', 'tau_pt', 'tau_e', 'tau_m', 'tau_visEta', 'tau_visPhi', 'tau_visPt', 'tau_visE', 'tau_visM', 'tau_visPtEm', 'tau_visPtHad', 'tau_visEEm', 'tau_visEHad', 'tau_DM']
    branches_genjet = ['jet_Idx', 'jet_eta', 'jet_phi', 'jet_pt', 'jet_e', 'jet_eEm', 'jet_eHad', 'jet_eInv']
    branches_cl3d   = ['cl3d_pt', 'cl3d_energy', 'cl3d_eta', 'cl3d_phi', 'cl3d_showerlength', 'cl3d_coreshowerlength', 'cl3d_firstlayer', 'cl3d_seetot', 'cl3d_seemax', 'cl3d_spptot', 'cl3d_sppmax', 'cl3d_szz', 'cl3d_srrtot', 'cl3d_srrmax', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz', 'cl3d_quality', 'cl3d_tauMatchIdx', 'cl3d_jetMatchIdx']
    NxM = options.caloClNxM
    branches_clNxM = ['cl'+NxM+'_barrelSeeded', 'cl'+NxM+'_nHits', 'cl'+NxM+'_seedIeta', 'cl'+NxM+'_seedIphi', 'cl'+NxM+'_seedEta', 'cl'+NxM+'_seedPhi', 'cl'+NxM+'_isBarrel', 'cl'+NxM+'_isOverlap', 'cl'+NxM+'_isEndcap', 'cl'+NxM+'_tauMatchIdx', 'cl'+NxM+'_jetMatchIdx', 'cl'+NxM+'_totalEm', 'cl'+NxM+'_totalHad', 'cl'+NxM+'_totalEt', 'cl'+NxM+'_totalEgEt', 'cl'+NxM+'_totalIem', 'cl'+NxM+'_totalIhad', 'cl'+NxM+'_totalIet', 'cl'+NxM+'_totalEgIet', 'cl'+NxM+'_towerEta', 'cl'+NxM+'_towerPhi', 'cl'+NxM+'_towerEm', 'cl'+NxM+'_towerHad', 'cl'+NxM+'_towerEt', 'cl'+NxM+'_towerIeta', 'cl'+NxM+'_towerIphi', 'cl'+NxM+'_towerIem', 'cl'+NxM+'_towerIhad', 'cl'+NxM+'_towerIet', 'cl'+NxM+'_nEGs', 'cl'+NxM+'_towerEgEt', 'cl'+NxM+'_towerEgIet', 'cl'+NxM+'_towerNeg']
    # branches_clNxM = ['cl'+NxM+'_barrelSeeded', 'cl'+NxM+'_nHits', 'cl'+NxM+'_seedIeta', 'cl'+NxM+'_seedIphi', 'cl'+NxM+'_isBarrel', 'cl'+NxM+'_isOverlap', 'cl'+NxM+'_isEndcap', 'cl'+NxM+'_tauMatchIdx', 'cl'+NxM+'_jetMatchIdx', 'cl'+NxM+'_totalIem', 'cl'+NxM+'_totalIhad', 'cl'+NxM+'_totalIet', 'cl'+NxM+'_towerIeta', 'cl'+NxM+'_towerIphi', 'cl'+NxM+'_towerIem', 'cl'+NxM+'_towerIhad', 'cl'+NxM+'_towerIet', 'cl'+NxM+'_towerEgIet']

    if options.doTens4Rate: branches_clNxM = ['cl'+NxM+'_seedEta', 'cl'+NxM+'_seedPhi', 'cl'+NxM+'_towerEm', 'cl'+NxM+'_towerHad', 'cl'+NxM+'_towerEt', 'cl'+NxM+'_towerEgEt']

    # define the two paths where to store the hdf5 files
    saveDfsTo = {
        'HGClus'  : options.outdir+'/L1Clusters/HGClus',
        'TowClus' : options.outdir+'/L1Clusters/TowClus'+NxM,
        'GenTaus' : options.outdir+'/GenObjects/GenTaus',
        'GenJets' : options.outdir+'/GenObjects/GenJets'
    }

    ##################### READ THE TTREES ####################
    print(' ** INFO : reading trees')
    TTree = uproot3.open(Infile)[key]

    if not options.doTens4Rate:
        arr_event  = TTree.arrays(branches_event)
        arr_gentau = TTree.arrays(branches_gentau)
        arr_genjet = TTree.arrays(branches_genjet)
        arr_cl3d   = TTree.arrays(branches_cl3d)
        arr_clNxM  = TTree.arrays(branches_clNxM)

    else:
        arr_event  = TTree.arrays(branches_event)
        arr_clNxM  = TTree.arrays(branches_clNxM)
        arr_cl3d   = TTree.arrays(branches_cl3d)


    ## DEBUG STUFF
        # print('*******************************************************')
        # bNxM = NxM.encode('utf-8')
        # # get clusters' shape dimensions
        # N = int(NxM.split('x')[0])
        # M = int(NxM.split('x')[1])
        # print(len(arr_event[b'EventNumber']))
        # print(len(arr_gentau[b'tau_eta']))
        # for i in range(len(arr_gentau[b'tau_eta'])):
        #     print('    ->', len(arr_gentau[b'tau_eta'][i]))
        #     for j in range(len(arr_gentau[b'tau_eta'][i])):
        #         print('        - idx', arr_gentau[b'tau_Idx'][i][j],' eta', arr_gentau[b'tau_eta'][i][j], 'phi', arr_gentau[b'tau_phi'][i][j], 'pt', arr_gentau[b'tau_pt'][i][j])
        # print(len(arr_genjet[b'jet_eta']))
        # for i in range(len(arr_genjet[b'jet_eta'])):
        #     print('    ->', len(arr_genjet[b'jet_eta'][i]))
        # print(len(arr_cl3d[b'cl3d_pt']))
        # for i in range(len(arr_cl3d[b'cl3d_pt'])):
        #     print('    ->', len(arr_cl3d[b'cl3d_pt'][i]))
        # print(len(arr_clNxM[b'cl'+bNxM+b'_nHits']))
        # for i in range(len(arr_clNxM[b'cl'+bNxM+b'_nHits'])):
        #     print('    ->', len(arr_clNxM[b'cl'+bNxM+b'_nHits'][i]))
        # for i in range(len(arr_clNxM[b'cl'+bNxM+b'_towerEta'][0])):
        #     print('        ->', len(arr_clNxM[b'cl'+bNxM+b'_towerEta'][0][i]))

        # evtN = arr_event[b'EventNumber']
        # ietas = arr_clNxM[b'cl'+bNxM+b'_towerIeta']
        # seedEtas = arr_clNxM[b'cl'+bNxM+b'_seedIeta']
        # seedPhis = arr_clNxM[b'cl'+bNxM+b'_seedIphi']
        # for i in range(len(evtN)):
        #     for j in range(len(ietas[i])):
        #         if len(ietas[i][j]) != N*M:
        #             print('    ** WARNING - EVT', evtN[i],'**    ->', len(ietas[i][j]), '    -> seedEta ', seedEtas[i][j], 'seedPhi', seedPhis[i][j])
        # print('*******************************************************')

    if not options.doTens4Rate:
        df_event  = pd.DataFrame(arr_event)
        df_gentau = pd.DataFrame(arr_gentau)
        df_genjet = pd.DataFrame(arr_genjet)
        df_cl3d   = pd.DataFrame(arr_cl3d)
        df_clNxM  = pd.DataFrame(arr_clNxM)

        dfHGClus  = pd.concat([df_event, df_cl3d], axis=1)
        dfTowClus = pd.concat([df_event, df_clNxM], axis=1)
        dfGenTaus = pd.concat([df_event, df_gentau], axis=1)
        dfGenJets = pd.concat([df_event, df_genjet], axis=1)
    
    else:
        df_event  = pd.DataFrame(arr_event)
        df_clNxM  = pd.DataFrame(arr_clNxM)
        df_cl3d   = pd.DataFrame(arr_cl3d)

        dfHGClus  = pd.concat([df_event, df_cl3d], axis=1)
        dfTowClus = pd.concat([df_event, df_clNxM], axis=1)

    ## uncomment for fast tests
    dfHGClus = dfHGClus.head(100).copy(deep=True)
    dfTowClus = dfTowClus.head(100).copy(deep=True)
    # dfGenTaus = dfGenTaus.head(100).copy(deep=True)
    # dfGenJets = dfGenJets.head(100).copy(deep=True)


    ##################### FLATTEN THE TTREES ####################
    print(' ** INFO : flattening trees')
    if not options.doTens4Rate:
        # flatten out the jets dataframe
        dfFlatGenTaus = pd.DataFrame({
            'event'        : np.repeat(dfGenTaus[b'EventNumber'].values, dfGenTaus[b'tau_eta'].str.len()), # event IDs are copied to keep proper track of what is what
            'tau_Idx'      : list(chain.from_iterable(dfGenTaus[b'tau_Idx'])),
            'tau_eta'      : list(chain.from_iterable(dfGenTaus[b'tau_eta'])),
            'tau_phi'      : list(chain.from_iterable(dfGenTaus[b'tau_phi'])),
            'tau_pt'       : list(chain.from_iterable(dfGenTaus[b'tau_pt'])),
            'tau_e'        : list(chain.from_iterable(dfGenTaus[b'tau_e'])),
            'tau_m'        : list(chain.from_iterable(dfGenTaus[b'tau_m'])),
            'tau_visEta'   : list(chain.from_iterable(dfGenTaus[b'tau_visEta'])),
            'tau_visPhi'   : list(chain.from_iterable(dfGenTaus[b'tau_visPhi'])),
            'tau_visPt'    : list(chain.from_iterable(dfGenTaus[b'tau_visPt'])),
            'tau_visE'     : list(chain.from_iterable(dfGenTaus[b'tau_visE'])),
            'tau_visM'     : list(chain.from_iterable(dfGenTaus[b'tau_visM'])),
            'tau_visPtEm'  : list(chain.from_iterable(dfGenTaus[b'tau_visPtEm'])),
            'tau_visPtHad' : list(chain.from_iterable(dfGenTaus[b'tau_visPtHad'])),
            'tau_visEEm'   : list(chain.from_iterable(dfGenTaus[b'tau_visEEm'])),
            'tau_visEHad'  : list(chain.from_iterable(dfGenTaus[b'tau_visEHad'])),
            'tau_DM'       : list(chain.from_iterable(dfGenTaus[b'tau_DM']))
            })

        # flatten out the jets dataframe
        dfFlatGenJets = pd.DataFrame({
            'event'    : np.repeat(dfGenJets[b'EventNumber'].values, dfGenJets[b'jet_eta'].str.len()), # event IDs are copied to keep proper track of what is what
            'jet_Idx'  : list(chain.from_iterable(dfGenJets[b'jet_Idx'])),
            'jet_eta'  : list(chain.from_iterable(dfGenJets[b'jet_eta'])),
            'jet_phi'  : list(chain.from_iterable(dfGenJets[b'jet_phi'])),
            'jet_pt'   : list(chain.from_iterable(dfGenJets[b'jet_pt'])),
            'jet_e'    : list(chain.from_iterable(dfGenJets[b'jet_e'])),
            'jet_eEm'  : list(chain.from_iterable(dfGenJets[b'jet_eEm'])),
            'jet_eHad' : list(chain.from_iterable(dfGenJets[b'jet_eHad'])),
            'jet_eInv' : list(chain.from_iterable(dfGenJets[b'jet_eInv']))
            })

        # flatten out the hgcal clusters dataframe
        dfFlatHGClus = pd.DataFrame({
            'event'                 : np.repeat(dfHGClus[b'EventNumber'].values, dfHGClus[b'cl3d_eta'].str.len()), # event IDs are copied to keep proper track of what is what
            'cl3d_pt'               : list(chain.from_iterable(dfHGClus[b'cl3d_pt'])),
            'cl3d_energy'           : list(chain.from_iterable(dfHGClus[b'cl3d_energy'])),
            'cl3d_eta'              : list(chain.from_iterable(dfHGClus[b'cl3d_eta'])),
            'cl3d_phi'              : list(chain.from_iterable(dfHGClus[b'cl3d_phi'])),
            'cl3d_showerlength'     : list(chain.from_iterable(dfHGClus[b'cl3d_showerlength'])),
            'cl3d_coreshowerlength' : list(chain.from_iterable(dfHGClus[b'cl3d_coreshowerlength'])),
            'cl3d_firstlayer'       : list(chain.from_iterable(dfHGClus[b'cl3d_firstlayer'])),
            'cl3d_seetot'           : list(chain.from_iterable(dfHGClus[b'cl3d_seetot'])),
            'cl3d_seemax'           : list(chain.from_iterable(dfHGClus[b'cl3d_seemax'])),
            'cl3d_spptot'           : list(chain.from_iterable(dfHGClus[b'cl3d_spptot'])),
            'cl3d_sppmax'           : list(chain.from_iterable(dfHGClus[b'cl3d_sppmax'])),
            'cl3d_szz'              : list(chain.from_iterable(dfHGClus[b'cl3d_szz'])),
            'cl3d_srrtot'           : list(chain.from_iterable(dfHGClus[b'cl3d_srrtot'])),
            'cl3d_srrmax'           : list(chain.from_iterable(dfHGClus[b'cl3d_srrmax'])),
            'cl3d_srrmean'          : list(chain.from_iterable(dfHGClus[b'cl3d_srrmean'])),
            'cl3d_hoe'              : list(chain.from_iterable(dfHGClus[b'cl3d_hoe'])),
            'cl3d_meanz'            : list(chain.from_iterable(dfHGClus[b'cl3d_meanz'])),
            'cl3d_quality'          : list(chain.from_iterable(dfHGClus[b'cl3d_quality'])),
            'cl3d_tauMatchIdx'      : list(chain.from_iterable(dfHGClus[b'cl3d_tauMatchIdx'])),
            'cl3d_jetMatchIdx'      : list(chain.from_iterable(dfHGClus[b'cl3d_jetMatchIdx']))
            })

        bNxM = NxM.encode('utf-8')
        # flatten out the tower clusters dataframe
        dfFlatTowClus = pd.DataFrame({
            'event'           : np.repeat(dfTowClus[b'EventNumber'].values, dfTowClus[b'cl'+bNxM+b'_seedIeta'].str.len()), # event IDs are copied to keep proper track of what is what
            'cl_barrelSeeded' : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_barrelSeeded'])),
            'cl_nHits'        : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_nHits'])),
            'cl_nEGs'         : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_nEGs'])),
            'cl_seedIeta'     : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_seedIeta'])),
            'cl_seedIphi'     : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_seedIphi'])),
            'cl_seedEta'      : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_seedEta'])),
            'cl_seedPhi'      : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_seedPhi'])),
            'cl_isBarrel'     : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_isBarrel'])),
            'cl_isOverlap'    : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_isOverlap'])),
            'cl_isEndcap'     : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_isEndcap'])),
            'cl_tauMatchIdx'  : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_tauMatchIdx'])),
            'cl_jetMatchIdx'  : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_jetMatchIdx'])),
            'cl_totalEm'      : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_totalEm'])),
            'cl_totalHad'     : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_totalHad'])),
            'cl_totalEt'      : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_totalEt'])),
            'cl_totalEgEt'    : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_totalEgEt'])),
            'cl_totalIem'     : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_totalIem'])),
            'cl_totalIhad'    : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_totalIhad'])),
            'cl_totalIet'     : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_totalIet'])),
            'cl_totalEgIet'   : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_totalEgIet'])),
            'cl_towerEta'     : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_towerEta'])),
            'cl_towerPhi'     : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_towerPhi'])),
            'cl_towerEm'      : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_towerEm'])),
            'cl_towerHad'     : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_towerHad'])),
            'cl_towerEt'      : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_towerEt'])),
            'cl_towerEgEt'    : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_towerEgEt'])),
            'cl_towerIeta'    : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_towerIeta'])),
            'cl_towerIphi'    : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_towerIphi'])),
            'cl_towerIem'     : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_towerIem'])),
            'cl_towerIhad'    : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_towerIhad'])),
            'cl_towerIet'     : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_towerIet'])),
            'cl_towerEgIet'   : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_towerEgIet'])),
            'cl_towerNeg'     : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_towerNeg'])),
            })

    else:
        # flatten out the hgcal clusters dataframe
        dfFlatHGClus = pd.DataFrame({
            'event'                 : np.repeat(dfHGClus[b'EventNumber'].values, dfHGClus[b'cl3d_eta'].str.len()), # event IDs are copied to keep proper track of what is what
            'cl3d_pt'               : list(chain.from_iterable(dfHGClus[b'cl3d_pt'])),
            'cl3d_energy'           : list(chain.from_iterable(dfHGClus[b'cl3d_energy'])),
            'cl3d_eta'              : list(chain.from_iterable(dfHGClus[b'cl3d_eta'])),
            'cl3d_phi'              : list(chain.from_iterable(dfHGClus[b'cl3d_phi'])),
            'cl3d_showerlength'     : list(chain.from_iterable(dfHGClus[b'cl3d_showerlength'])),
            'cl3d_coreshowerlength' : list(chain.from_iterable(dfHGClus[b'cl3d_coreshowerlength'])),
            'cl3d_firstlayer'       : list(chain.from_iterable(dfHGClus[b'cl3d_firstlayer'])),
            'cl3d_seetot'           : list(chain.from_iterable(dfHGClus[b'cl3d_seetot'])),
            'cl3d_seemax'           : list(chain.from_iterable(dfHGClus[b'cl3d_seemax'])),
            'cl3d_spptot'           : list(chain.from_iterable(dfHGClus[b'cl3d_spptot'])),
            'cl3d_sppmax'           : list(chain.from_iterable(dfHGClus[b'cl3d_sppmax'])),
            'cl3d_szz'              : list(chain.from_iterable(dfHGClus[b'cl3d_szz'])),
            'cl3d_srrtot'           : list(chain.from_iterable(dfHGClus[b'cl3d_srrtot'])),
            'cl3d_srrmax'           : list(chain.from_iterable(dfHGClus[b'cl3d_srrmax'])),
            'cl3d_srrmean'          : list(chain.from_iterable(dfHGClus[b'cl3d_srrmean'])),
            'cl3d_hoe'              : list(chain.from_iterable(dfHGClus[b'cl3d_hoe'])),
            'cl3d_meanz'            : list(chain.from_iterable(dfHGClus[b'cl3d_meanz'])),
            'cl3d_quality'          : list(chain.from_iterable(dfHGClus[b'cl3d_quality'])),
            'cl3d_tauMatchIdx'      : list(chain.from_iterable(dfHGClus[b'cl3d_tauMatchIdx'])),
            'cl3d_jetMatchIdx'      : list(chain.from_iterable(dfHGClus[b'cl3d_jetMatchIdx']))
            })

        bNxM = NxM.encode('utf-8')
        # flatten out the tower clusters dataframe
        dfFlatTowClus = pd.DataFrame({
            'event'           : np.repeat(dfTowClus[b'EventNumber'].values, dfTowClus[b'cl'+bNxM+b'_seedEta'].str.len()), # event IDs are copied to keep proper track of what is what
            'cl_seedEta'      : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_seedEta'])),
            'cl_seedPhi'      : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_seedPhi'])),
            'cl_towerEm'      : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_towerEm'])),
            'cl_towerHad'     : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_towerHad'])),
            'cl_towerEt'      : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_towerEt'])),
            'cl_towerEgEt'    : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_towerEgEt'])),
            })


    ##################### SAVE TO FILE ####################
    print(' ** INFO : saving DFs')

    if not options.doTens4Rate:
        dfFlatHGClus.to_pickle(saveDfsTo['HGClus']+options.infileTag+'.pkl')
        dfFlatTowClus.to_pickle(saveDfsTo['TowClus']+options.infileTag+'.pkl')
        dfFlatGenTaus.to_pickle(saveDfsTo['GenTaus']+options.infileTag+'.pkl')
        dfFlatGenJets.to_pickle(saveDfsTo['GenJets']+options.infileTag+'.pkl')
    else:
        dfFlatTowClus.to_pickle(saveDfsTo['TowClus']+options.infileTag+'.pkl')
        dfFlatHGClus.to_pickle(saveDfsTo['HGClus']+options.infileTag+'.pkl')


    ##################### TENSORIZE FOR NN ####################

    if options.outTag != "":
        outTag = options.outTag
    else:
        outTag = ""
        if options.uJetPtCut : outTag += '_uJetPtCut'+options.uJetPtCut
        if options.lJetPtCut : outTag += '_lJetPtCut'+options.lJetPtCut
        if options.uTauPtCut : outTag += '_uTauPtCut'+options.uTauPtCut
        if options.lTauPtCut : outTag += '_lTauPtCut'+options.lTauPtCut
        if options.uEtacut   : outTag += '_uEtacut'+options.uEtacut
        if options.lEtacut   : outTag += '_lEtacut'+options.lEtacut

    saveTensTo = {
        'inputsCalibratorCNN'    : options.outdir+'/TensorizedInputs_'+options.caloClNxM+outTag+'/X_CNN_Calibrator'+options.caloClNxM+options.infileTag+'.npz',
        'inputsCalibratorDense'  : options.outdir+'/TensorizedInputs_'+options.caloClNxM+outTag+'/X_Dense_Calibrator'+options.caloClNxM+options.infileTag+'.npz',
        'inputsIdentifierCNN'    : options.outdir+'/TensorizedInputs_'+options.caloClNxM+outTag+'/X_CNN_Identifier'+options.caloClNxM+options.infileTag+'.npz',
        'inputsIdentifierDense'  : options.outdir+'/TensorizedInputs_'+options.caloClNxM+outTag+'/X_Dense_Identifier'+options.caloClNxM+options.infileTag+'.npz',
        'inputsRateCNN'          : options.outdir+'/TensorizedInputs_'+options.caloClNxM+outTag+'/X_CNN_Rate'+options.caloClNxM+options.infileTag+'.npz',
        'inputsRateDense'        : options.outdir+'/TensorizedInputs_'+options.caloClNxM+outTag+'/X_Dense_Rate'+options.caloClNxM+options.infileTag+'.npz',
        'targetsCalibrator'      : options.outdir+'/TensorizedInputs_'+options.caloClNxM+outTag+'/Y_Calibrator'+options.caloClNxM+options.infileTag+'.npz',
        'targetsIdentifier'      : options.outdir+'/TensorizedInputs_'+options.caloClNxM+outTag+'/Y_Identifier'+options.caloClNxM+options.infileTag+'.npz',
        'targetsRate'            : options.outdir+'/TensorizedInputs_'+options.caloClNxM+outTag+'/Y_Rate'+options.caloClNxM+options.infileTag+'.npz',
        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------
        'inputsCoTrainingCNN'    : options.outdir+'/TensorizedInputs_'+options.caloClNxM+outTag+'/X_CNN_CoTraining'+options.caloClNxM+options.infileTag+'.npz',
        'inputsCoTrainingDense'  : options.outdir+'/TensorizedInputs_'+options.caloClNxM+outTag+'/X_Dense_CoTraining'+options.caloClNxM+options.infileTag+'.npz',
        'targetsCoTraining'      : options.outdir+'/TensorizedInputs_'+options.caloClNxM+outTag+'/Y_CoTraining'+options.caloClNxM+options.infileTag+'.npz',
        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------
        'inputsCalibratorBDT'    : options.outdir+'/PickledInputs'+outTag+'/X_BDT_Calibrator'+options.infileTag+'.pkl',
        'inputsIdentifierBDT'    : options.outdir+'/PickledInputs'+outTag+'/X_BDT_Identifier'+options.infileTag+'.pkl',
        'inputsRateBDT'          : options.outdir+'/PickledInputs'+outTag+'/X_BDT_Rate'+options.infileTag+'.pkl',
        # -----------------------------------------------------------------------------------------------------------------------------------------------------------------
        'inputsMinatorCNN'       : options.outdir+'/MinatorPerformanceInputs_'+options.caloClNxM+outTag+'/X_CNN_Minator'+options.caloClNxM+options.infileTag+'.npz',
        'inputsMinatorDense'     : options.outdir+'/MinatorPerformanceInputs_'+options.caloClNxM+outTag+'/X_Dense_Minator'+options.caloClNxM+options.infileTag+'.npz',
        'targetsMinator'         : options.outdir+'/MinatorPerformanceInputs_'+options.caloClNxM+outTag+'/Y_Minator'+options.caloClNxM+options.infileTag+'.npz',
        'inputsMinatorBDT'       : options.outdir+'/MinatorPerformanceInputs_'+options.caloClNxM+outTag+'/X_BDT_Minator'+options.infileTag+'.pkl',
        'inputsMinatorRateCNN'   : options.outdir+'/MinatorRateInputs_'+options.caloClNxM+outTag+'/X_CNN_Rate'+options.caloClNxM+options.infileTag+'.npz',
        'inputsMinatorRateDense' : options.outdir+'/MinatorRateInputs_'+options.caloClNxM+outTag+'/X_Dense_Rate'+options.caloClNxM+options.infileTag+'.npz',
        'targetsMinatorRate'     : options.outdir+'/MinatorRateInputs_'+options.caloClNxM+outTag+'/Y_Rate'+options.caloClNxM+options.infileTag+'.npz',
        'inputsMinatorRateBDT'   : options.outdir+'/MinatorRateInputs_'+options.caloClNxM+outTag+'/X_BDT_Rate'+options.infileTag+'.pkl',
    }

    if options.doTens4Cotraining:
        print('** INFO : doing tensorization for co-training')
        TensorizeForClNxMCoTraining(dfFlatTowClus, dfFlatGenTaus, dfFlatGenJets, options.uJetPtCut, options.lJetPtCut, options.uTauPtCut, options.lTauPtCut, options.uEtacut, options.lEtacut, options.caloClNxM, options.inputPrecision)

    if options.doTens4Calib:
        print('** INFO : doing tensorization for calibration')
        if options.doCALO:  TensorizeForClNxMCalibration(dfFlatTowClus, dfFlatGenTaus, options.uTauPtCut, options.lTauPtCut, options.uEtacut, options.lEtacut, options.caloClNxM, options.inputPrecision)
        if options.doHGCAL: TensorizeForCl3dCalibration(dfFlatHGClus, dfFlatGenTaus, options.uTauPtCut, options.lTauPtCut, options.uEtacut, options.lEtacut, options.inputPrecision)

    if options.doTens4Ident:
        print('** INFO : doing tensorization for identification')
        if options.doCALO:  TensorizeForClNxMIdentification(dfFlatTowClus, dfFlatGenTaus, dfFlatGenJets, options.uJetPtCut, options.lJetPtCut, options.uTauPtCut, options.lTauPtCut, options.uEtacut, options.lEtacut, options.caloClNxM, options.inputPrecision)
        if options.doHGCAL: TensorizeForCl3dIdentification(dfFlatHGClus, dfFlatGenTaus, dfFlatGenJets, options.uJetPtCut, options.lJetPtCut, options.uTauPtCut, options.lTauPtCut, options.uEtacut, options.lEtacut, options.inputPrecision)

    if options.doTens4Minator:
        print('** INFO : doing tensorization for TauMinator performance evaluation')
        TensorizeForTauMinatorPerformance(dfFlatTowClus, dfFlatHGClus, dfFlatGenTaus, options.uTauPtCut, options.lTauPtCut, options.uEtacut, options.lEtacut, options.caloClNxM, options.inputPrecision)

    if options.doTens4Rate:
        print('** INFO : doing tensorization for rate evaluation')
        if options.doCALO:    TensorizeForClNxMRate(dfFlatTowClus, options.uEtacut, options.lEtacut, options.caloClNxM, options.inputPrecision)
        elif options.doHGCAL: TensorizeForCl3dRate(dfFlatHGClus, options.uEtacut, options.lEtacut, options.inputPrecision)
        else:                 TensorizeForTauMinatorRate(dfFlatTowClus, dfFlatHGClus, options.uEtacut, options.lEtacut, options.caloClNxM, options.inputPrecision)

    print('** INFO : ALL DONE!')