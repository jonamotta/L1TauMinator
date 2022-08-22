from optparse import OptionParser
from itertools import chain
import pandas as pd
import numpy as np
import argparse
import uproot3
import glob
import sys
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
    np.savez_compressed(saveTensTo['inputsIdentifier'], X)
    np.savez_compressed(saveTensTo['targetsIdentifier'], Y)


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
    np.savez_compressed(saveTensTo['inputsCalibrator'], X)
    np.savez_compressed(saveTensTo['targetsCalibrator'], Y)


#######################################################################
######################### SCRIPT BODY #################################
#######################################################################

if __name__ == "__main__" :

    parser = OptionParser()
    # GENERAL OPTIONS
    parser.add_option("--infile",       dest="infile",                                                                               default=None)
    parser.add_option("--outdir",       dest="outdir",                                                                               default=None)
    parser.add_option('--caloClNxM',    dest='caloClNxM',    help='Which shape of CaloCluster to use?',                              default="9x9")
    # TTREE READING OPTIONS
    parser.add_option('--doHH',         dest='doHH',         help='Read the HH samples?',                       action='store_true', default=False)
    parser.add_option('--doQCD',        dest='doQCD',        help='Read the QCD samples?',                      action='store_true', default=False)
    parser.add_option('--doVBFH',       dest='doVBFH',       help='Read the VBF H samples?',                    action='store_true', default=False)
    parser.add_option('--doMinBias',    dest='doMinBias',    help='Read the Minbias samples?',                  action='store_true', default=False)
    parser.add_option('--doZp500',      dest='doZp500',      help='Read the Minbias samples?',                  action='store_true', default=False)
    parser.add_option('--doZp1500',     dest='doZp1500',     help='Read the Minbias samples?',                  action='store_true', default=False)
    parser.add_option('--doTestRun',    dest='doTestRun',    help='Do test run with reduced number of events?', action='store_true', default=False)
    # TENSORIZATION OPTIONS
    parser.add_option("--infileTag",    dest="infileTag",                         default=None)
    parser.add_option("--outTag",       dest="outTag",                            default="")
    parser.add_option("--uJetPtCut",    dest="uJetPtCut",                         default=None)
    parser.add_option("--lJetPtCut",    dest="lJetPtCut",                         default=None)
    parser.add_option("--uTauPtCut",    dest="uTauPtCut",                         default=None)
    parser.add_option("--lTauPtCut",    dest="lTauPtCut",                         default=None)
    parser.add_option("--etacut",       dest="etacut",                            default=None)
    parser.add_option('--doTens4Calib', dest='doTens4Calib', action='store_true', default=False)
    parser.add_option('--doTens4Ident', dest='doTens4Ident', action='store_true', default=False)
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
    # branches_clNxM = ['cl'+NxM+'_barrelSeeded', 'cl'+NxM+'_nHits', 'cl'+NxM+'_seedIeta', 'cl'+NxM+'_seedIphi', 'cl'+NxM+'_seedEta', 'cl'+NxM+'_seedPhi', 'cl'+NxM+'_isBarrel', 'cl'+NxM+'_isOverlap', 'cl'+NxM+'_isEndcap', 'cl'+NxM+'_tauMatchIdx', 'cl'+NxM+'_jetMatchIdx', 'cl'+NxM+'_totalEm', 'cl'+NxM+'_totalHad', 'cl'+NxM+'_totalEt', 'cl'+NxM+'_totalIem', 'cl'+NxM+'_totalIhad', 'cl'+NxM+'_totalIet', 'cl'+NxM+'_towerEta', 'cl'+NxM+'_towerPhi', 'cl'+NxM+'_towerEm', 'cl'+NxM+'_towerHad', 'cl'+NxM+'_towerEt', 'cl'+NxM+'_towerIeta', 'cl'+NxM+'_towerIphi', 'cl'+NxM+'_towerIem', 'cl'+NxM+'_towerIhad', 'cl'+NxM+'_towerIet', 'cl'+NxM+'_nEGs', 'cl'+NxM+'_towerEgEt', 'cl'+NxM+'_towerEgIet', 'cl'+NxM+'_towerNeg']
    branches_clNxM = ['cl'+NxM+'_barrelSeeded', 'cl'+NxM+'_nHits', 'cl'+NxM+'_seedIeta', 'cl'+NxM+'_seedIphi', 'cl'+NxM+'_isBarrel', 'cl'+NxM+'_isOverlap', 'cl'+NxM+'_isEndcap', 'cl'+NxM+'_tauMatchIdx', 'cl'+NxM+'_jetMatchIdx', 'cl'+NxM+'_totalIem', 'cl'+NxM+'_totalIhad', 'cl'+NxM+'_totalIet', 'cl'+NxM+'_towerIeta', 'cl'+NxM+'_towerIphi', 'cl'+NxM+'_towerIem', 'cl'+NxM+'_towerIhad', 'cl'+NxM+'_towerIet', 'cl'+NxM+'_towerEgIet']

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

    arr_event  = TTree.arrays(branches_event)
    arr_gentau = TTree.arrays(branches_gentau)
    arr_genjet = TTree.arrays(branches_genjet)
    arr_cl3d   = TTree.arrays(branches_cl3d)
    arr_clNxM  = TTree.arrays(branches_clNxM)

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

    df_event  = pd.DataFrame(arr_event)
    df_gentau = pd.DataFrame(arr_gentau)
    df_genjet = pd.DataFrame(arr_genjet)
    df_cl3d   = pd.DataFrame(arr_cl3d)
    df_clNxM  = pd.DataFrame(arr_clNxM)

    dfHGClus  = pd.concat([df_event, df_cl3d], axis=1)
    dfTowClus = pd.concat([df_event, df_clNxM], axis=1)
    dfGenTaus = pd.concat([df_event, df_gentau], axis=1)
    dfGenJets = pd.concat([df_event, df_genjet], axis=1)


    ##################### FLATTEN THE TTREES ####################
    print(' ** INFO : flattening trees')
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
        # 'cl_nEGs'         : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_nEGs'])),
        'cl_seedIeta'     : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_seedIeta'])),
        'cl_seedIphi'     : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_seedIphi'])),
        # 'cl_seedEta'      : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_seedEta'])),
        # 'cl_seedPhi'      : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_seedPhi'])),
        'cl_isBarrel'     : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_isBarrel'])),
        'cl_isOverlap'    : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_isOverlap'])),
        'cl_isEndcap'     : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_isEndcap'])),
        'cl_tauMatchIdx'  : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_tauMatchIdx'])),
        'cl_jetMatchIdx'  : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_jetMatchIdx'])),
        # 'cl_totalEm'      : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_totalEm'])),
        # 'cl_totalHad'     : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_totalHad'])),
        # 'cl_totalEt'      : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_totalEt'])),
        'cl_totalIem'     : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_totalIem'])),
        'cl_totalIhad'    : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_totalIhad'])),
        'cl_totalIet'     : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_totalIet'])),
        # 'cl_towerEta'     : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_towerEta'])),
        # 'cl_towerPhi'     : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_towerPhi'])),
        # 'cl_towerEm'      : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_towerEm'])),
        # 'cl_towerHad'     : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_towerHad'])),
        # 'cl_towerEt'      : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_towerEt'])),
        # 'cl_towerEgEt'    : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_towerEgEt'])),
        'cl_towerIeta'    : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_towerIeta'])),
        'cl_towerIphi'    : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_towerIphi'])),
        'cl_towerIem'     : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_towerIem'])),
        'cl_towerIhad'    : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_towerIhad'])),
        'cl_towerIet'     : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_towerIet'])),
        'cl_towerEgIet'   : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_towerEgIet'])),
        # 'cl_towerNeg'     : list(chain.from_iterable(dfTowClus[b'cl'+bNxM+b'_towerNeg'])),
        })


    print(dfFlatGenTaus.shape, len(dfFlatGenTaus))
    print(dfFlatGenJets.shape, len(dfFlatGenJets))
    print(dfFlatHGClus.shape, len(dfFlatHGClus))
    print(dfFlatTowClus.shape, len(dfFlatTowClus))


    ##################### SAVE TO FILE ####################
    print(' ** INFO : saving DFs')

    dfFlatHGClus.to_pickle(saveDfsTo['HGClus']+options.infileTag+'.pkl')
    dfFlatTowClus.to_pickle(saveDfsTo['TowClus']+options.infileTag+'.pkl')
    dfFlatGenTaus.to_pickle(saveDfsTo['GenTaus']+options.infileTag+'.pkl')
    dfFlatGenJets.to_pickle(saveDfsTo['GenJets']+options.infileTag+'.pkl')


    ##################### TENSORIZE FOR NN ####################

    saveTensTo = {
        'inputsCalibrator'  : options.outdir+'/TensorizedInputs_'+options.caloClNxM+options.outTag+'/X_Calibrator'+options.caloClNxM+options.infileTag+'.npz',
        'inputsIdentifier'  : options.outdir+'/TensorizedInputs_'+options.caloClNxM+options.outTag+'/X_Identifier'+options.caloClNxM+options.infileTag+'.npz',
        'targetsCalibrator' : options.outdir+'/TensorizedInputs_'+options.caloClNxM+options.outTag+'/Y_Calibrator'+options.caloClNxM+options.infileTag+'.npz',
        'targetsIdentifier' : options.outdir+'/TensorizedInputs_'+options.caloClNxM+options.outTag+'/Y_Identifier'+options.caloClNxM+options.infileTag+'.npz'
    }

    if options.doTens4Calib:
        print('** INFO : doing tensorization for calibration')
        TensorizeForCalibration(dfFlatTowClus, dfFlatGenTaus, options.uTauPtCut, options.lTauPtCut, options.etacut, options.caloClNxM)
    if options.doTens4Ident:
        print('** INFO : doing tensorization for identification')
        TensorizeForIdentification(dfFlatTowClus, dfFlatGenTaus, dfFlatGenJets, options.uJetPtCut, options.lJetPtCut, options.uTauPtCut, options.lTauPtCut, options.etacut, options.caloClNxM)


    print('** INFO : ALL DONE!')