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


def Tensorize(dfFlatTowClus, dfFlatHGClus, uEtacut, lEtacut,  NxM, inputPrecision):
    if len(dfFlatTowClus) == 0 or len(dfFlatHGClus) == 0:
        print('** WARNING : no data to be tensorized for calibration here')
        return

    dfTowClus = dfFlatTowClus.copy(deep=True)
    dfHGClus = dfFlatHGClus.copy(deep=True)

    dfHGClus['cl3d_abseta'] = abs(dfHGClus['cl3d_eta'])

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
        
        # "targets" of the NN -> the ones produced by CMSSW thta have to match the predictions
        yl = []
        yl.append(dfTowClus.cl_IDscore.loc[idx])
        yl.append(dfTowClus.cl_calibPt.loc[idx])
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
    np.savez_compressed(saveTensTo['inputsCNN'], X1)
    np.savez_compressed(saveTensTo['inputsDNN'], X2)
    np.savez_compressed(saveTensTo['targetsCNN+DNN'], Y)

    # save .pkl file with formatted datasets
    dfHGClus.to_pickle(saveTensTo['inputsBDT'])


#######################################################################
######################### SCRIPT BODY #################################
#######################################################################

if __name__ == "__main__" :

    parser = OptionParser()
    # GENERAL OPTIONS
    parser.add_option("--infile",         dest="infile",                                                                               default=None)
    parser.add_option("--outdir",         dest="outdir",                                                                               default=None)
    parser.add_option('--caloClNxM',      dest='caloClNxM',    help='Which shape of CaloCluster to use?',                              default="5x9")
    # TENSORIZATION OPTIONS
    parser.add_option("--inputPrecision", dest="inputPrecision", type=float,          default=None)
    parser.add_option("--uJetPtCut",      dest="uJetPtCut",                           default=None)
    parser.add_option("--lJetPtCut",      dest="lJetPtCut",                           default=None)
    parser.add_option("--uTauPtCut",      dest="uTauPtCut",                           default=None)
    parser.add_option("--lTauPtCut",      dest="lTauPtCut",                           default=None)
    parser.add_option("--uEtacut",        dest="uEtacut",                             default=None)
    parser.add_option("--lEtacut",        dest="lEtacut",                             default=None)
    (options, args) = parser.parse_args()

    print(options)

    ##################### DEFINE INPUTS AND OUTPUTS ####################
    Infile  = options.infile

    key = 'L1CaloTauNtuplizer/L1TauMinatorTree'
    branches_event  = ['EventNumber']
    branches_gentau = ['tau_Idx', 'tau_eta', 'tau_phi', 'tau_pt', 'tau_e', 'tau_m', 'tau_visEta', 'tau_visPhi', 'tau_visPt', 'tau_visE', 'tau_visM', 'tau_visPtEm', 'tau_visPtHad', 'tau_visEEm', 'tau_visEHad', 'tau_DM']
    branches_cl3d   = ['cl3d_pt', 'cl3d_eta', 'cl3d_showerlength', 'cl3d_coreshowerlength', 'cl3d_spptot', 'cl3d_srrtot', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz', 'cl3d_tauMatchIdx', 'cl3d_calibPt', 'cl3d_IDscore']
    NxM = options.caloClNxM
    branches_clNxM = ['cl'+NxM+'_seedEta', 'cl'+NxM+'_seedPhi', 'cl'+NxM+'_tauMatchIdx', 'cl'+NxM+'_towerEm', 'cl'+NxM+'_towerHad', 'cl'+NxM+'_towerEgEt', 'cl'+NxM+'_calibPt', 'cl'+NxM+'_IDscore']
    branches_l1tau  = ['l1tau_pt', 'l1tau_eta', 'l1tau_phi', 'l1tau_clusterIdx', 'l1tau_isBarrel', 'l1tau_isEndcap', 'l1tau_IDscore', 'l1tau_tauMatchIdx']

    # define the two paths where to store the hdf5 files
    saveDfsTo = {
        'HGClus'  : options.outdir+'/DFs/HGClus',
        'TowClus' : options.outdir+'/DFs/TowClus'+NxM,
        'GenTaus' : options.outdir+'/DFs/GenTaus',
        'L1Taus' : options.outdir+'/DFs/GenTaus',
    }

    outTag = ""
    if options.uJetPtCut : outTag += '_uJetPtCut'+options.uJetPtCut
    if options.lJetPtCut : outTag += '_lJetPtCut'+options.lJetPtCut
    if options.uTauPtCut : outTag += '_uTauPtCut'+options.uTauPtCut
    if options.lTauPtCut : outTag += '_lTauPtCut'+options.lTauPtCut
    if options.uEtacut   : outTag += '_uEtacut'+options.uEtacut
    if options.lEtacut   : outTag += '_lEtacut'+options.lEtacut

    os.system('mkdir -p '+options.outdir+'/DFs')
    os.system('mkdir -p '+options.outdir+'/Tensorized_'+options.caloClNxM+outTag)

    ##################### READ THE TTREES ####################
    print(' ** INFO : reading trees')
    TTree = uproot3.open(Infile)[key]

    arr_event  = TTree.arrays(branches_event)
    arr_gentau = TTree.arrays(branches_gentau)
    arr_l1tau  = TTree.arrays(branches_l1tau)
    arr_cl3d   = TTree.arrays(branches_cl3d)
    arr_clNxM  = TTree.arrays(branches_clNxM)

    df_event  = pd.DataFrame(arr_event)
    df_gentau = pd.DataFrame(arr_gentau)
    df_l1tau  = pd.DataFrame(arr_l1tau)
    df_cl3d   = pd.DataFrame(arr_cl3d)
    df_clNxM  = pd.DataFrame(arr_clNxM)

    dfHGClus  = pd.concat([df_event, df_cl3d], axis=1)
    dfTowClus = pd.concat([df_event, df_clNxM], axis=1)
    dfGenTaus = pd.concat([df_event, df_gentau], axis=1)
    dfL1Taus  = pd.concat([df_event, df_l1tau], axis=1)
    

    ##################### FLATTEN THE TTREES ####################
    print(' ** INFO : flattening trees')
    # flatten out the taus dataframe
    dfFlatGenTaus = pd.DataFrame({
        'event'        : np.repeat(dfGenTaus[b'EventNumber'].values, dfGenTaus[b'tau_eta'].str.len()), # event IDs are copied to keep proper track of what is what
        'tau_Idx'      : list(chain.from_iterable(dfGenTaus[b'tau_Idx'])),
        'tau_visEta'   : list(chain.from_iterable(dfGenTaus[b'tau_visEta'])),
        'tau_visPhi'   : list(chain.from_iterable(dfGenTaus[b'tau_visPhi'])),
        'tau_visPt'    : list(chain.from_iterable(dfGenTaus[b'tau_visPt'])),
        'tau_DM'       : list(chain.from_iterable(dfGenTaus[b'tau_DM']))
        })

    dfFlatL1Taus = pd.DataFrame({
        'event'        : np.repeat(dfL1Taus[b'EventNumber'].values, dfL1Taus[b'l1tau_eta'].str.len()), # event IDs are copied to keep proper track of what is what
        'l1tau_pt'          : list(chain.from_iterable(dfL1Taus[b'l1tau_pt'])),
        'l1tau_eta'         : list(chain.from_iterable(dfL1Taus[b'l1tau_eta'])),
        'l1tau_phi'         : list(chain.from_iterable(dfL1Taus[b'l1tau_phi'])),
        'l1tau_clusterIdx'  : list(chain.from_iterable(dfL1Taus[b'l1tau_clusterIdx'])),
        'l1tau_isBarrel'    : list(chain.from_iterable(dfL1Taus[b'l1tau_isBarrel'])),
        'l1tau_isEndcap'    : list(chain.from_iterable(dfL1Taus[b'l1tau_isEndcap'])),
        'l1tau_IDscore'     : list(chain.from_iterable(dfL1Taus[b'l1tau_IDscore'])),
        'l1tau_tauMatchIdx' : list(chain.from_iterable(dfL1Taus[b'l1tau_tauMatchIdx'])),
        })

    # flatten out the hgcal clusters dataframe
    dfFlatHGClus = pd.DataFrame({
        'event'                 : np.repeat(dfHGClus[b'EventNumber'].values, dfHGClus[b'cl3d_eta'].str.len()), # event IDs are copied to keep proper track of what is what
        'cl3d_pt'               : list(chain.from_iterable(dfHGClus[b'cl3d_pt'])),
        'cl3d_eta'              : list(chain.from_iterable(dfHGClus[b'cl3d_eta'])),
        'cl3d_showerlength'     : list(chain.from_iterable(dfHGClus[b'cl3d_showerlength'])),
        'cl3d_coreshowerlength' : list(chain.from_iterable(dfHGClus[b'cl3d_coreshowerlength'])),
        'cl3d_spptot'           : list(chain.from_iterable(dfHGClus[b'cl3d_spptot'])),
        'cl3d_srrtot'           : list(chain.from_iterable(dfHGClus[b'cl3d_srrtot'])),
        'cl3d_srrmean'          : list(chain.from_iterable(dfHGClus[b'cl3d_srrmean'])),
        'cl3d_hoe'              : list(chain.from_iterable(dfHGClus[b'cl3d_hoe'])),
        'cl3d_meanz'            : list(chain.from_iterable(dfHGClus[b'cl3d_meanz'])),
        'cl3d_tauMatchIdx'      : list(chain.from_iterable(dfHGClus[b'cl3d_tauMatchIdx'])),
        'cl3d_IDscore'          : list(chain.from_iterable(dfHGClus[b'cl3d_IDscore'])),
        'cl3d_calibPt'          : list(chain.from_iterable(dfHGClus[b'cl3d_calibPt']))
        })

    bNxM = NxM.encode('utf-8')
    # flatten out the tower clusters dataframe
    dfFlatTowClus = pd.DataFrame({
        'event'           : np.repeat(dfTowClus[b'EventNumber'].values, dfTowClus[b'cl'+bNxM+b'_seedEta'].str.len()), # event IDs are copied to keep proper track of what is what
        'cl_seedEta'      : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_seedEta'])),
        'cl_seedPhi'      : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_seedPhi'])),
        'cl_tauMatchIdx'  : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_tauMatchIdx'])),
        'cl_towerEm'      : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_towerEm'])),
        'cl_towerHad'     : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_towerHad'])),
        'cl_towerEgEt'    : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_towerEgEt'])),
        'cl_IDscore'      : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_IDscore'])),
        'cl_calibPt'      : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_calibPt']))
        })

    ##################### SAVE TO FILE ####################
    print(' ** INFO : saving DFs')

    dfFlatHGClus.to_pickle(saveDfsTo['HGClus']+'.pkl')
    dfFlatTowClus.to_pickle(saveDfsTo['TowClus']+'.pkl')
    dfFlatGenTaus.to_pickle(saveDfsTo['GenTaus']+'.pkl')
    dfFlatL1Taus.to_pickle(saveDfsTo['L1Taus']+'.pkl')

    ##################### TENSORIZE FOR NN ####################

    saveTensTo = {
        'inputsCNN'      : options.outdir+'/Tensorized_'+options.caloClNxM+outTag+'/X_CNN_'+options.caloClNxM+'.npz',
        'inputsDNN'      : options.outdir+'/Tensorized_'+options.caloClNxM+outTag+'/X_Dense_'+options.caloClNxM+'.npz',
        'targetsCNN+DNN' : options.outdir+'/Tensorized_'+options.caloClNxM+outTag+'/Y_'+options.caloClNxM+'.npz',
        'inputsBDT'      : options.outdir+'/Tensorized_'+options.caloClNxM+outTag+'/X_BDT.pkl',
    }

    print('** INFO : doing fast tensorization')
    Tensorize(dfFlatTowClus, dfFlatHGClus, options.uEtacut, options.lEtacut, options.caloClNxM, options.inputPrecision)

    print('** INFO : ALL DONE!')