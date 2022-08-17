from optparse import OptionParser
import pandas as pd
import numpy as np
import math
import os

def TensorizeForIdentification(dfTowClus, dfGenTaus, dfGenJets, uJetPtCut, lJetPtCut, uTauPtCut, lTauPtCut, etacut):
    # Apply cut on jet pt
    if uJetPtcut:
        dfGenJets = dfGenJets[dfGenJets['jet_pt'] <= float(uJetPtcut)]
    if lJetPtcut:
        dfGenJets = dfGenJets[dfGenJets['jet_pt'] >= float(lJetPtcut)]

    # Apply cut on tau pt
    if uJetPtcut:
        dfGenTaus = dfGenTaus[dfGenTaus['tau_pt'] <= float(uJetPtcut)]
    if lJetPtcut:
        dfGenTaus = dfGenTaus[dfGenTaus['tau_pt'] >= float(lJetPtcut)]

    # Apply cut on tau/jet eta
    if etacut:
        dfGenTaus = dfGenTaus[dfGenTaus['tau_eta'] <= float(etacut)]
        dfGenJets = dfGenJets[dfGenJets['jet_eta'] <= float(etacut)]

    # transform pt in hardware units
    dfGenJets['tau_hwpt'] = dfGenJets['tau_pt'].copy(deep=True) * 2
    dfGenTaus['jet_hwpt'] = dfGenTaus['jet_pt'].copy(deep=True) * 2

    # save unique identifier
    dfGenJets['uniqueId'] = 'jet_'+dfGenJets['event'].astype(str)+'_'+dfGenJets['jet_Idx'].astype(str)
    dfGenTaus['uniqueId'] = 'tau_'+dfGenTaus['event'].astype(str)+'_'+dfGenTaus['tau_Idx'].astype(str)


def TensorizeForCalibration(dfTowClus, dfGenTaus, uTauPtCut, lTauPtCut, etacut):
    if len(dfTowClus) == 0 or len(dfGenTaus) == 0 or len(dfGenJets) == 0:
        print(' ** WARNING: Zero data here --> EXITING!\n')
        return

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
    dfGenTaus['tau_hwpt'] = dfGenTaus['tau_pt'].copy(deep=True) * 2

    # save unique identifier
    dfGenTaus['uniqueId'] = 'tau_'+dfGenTaus['event'].astype(str)+'_'+dfGenTaus['tau_Idx'].astype(str)

    # keep only the clusters that are matched to a tau
    dfTowClus = dfTowClus[dfTowClus['cl_tauMatchIdx'] >= 0]

    # join the taus and the clusters datasets -> this creates all the possible combination of clusters and jets/taus for each event
    # important that dfFlatET is joined to dfFlatEJ and not viceversa --> this because dfFlatEJ contains the safe jets to be used and the safe event numbers
    dfGenTaus.set_index('event', inplace=True)
    dfTowClus.set_index('event', inplace=True)
    dfCluTau = dfGenTaus.join(dfTowClus, on='event', how='left', rsuffix='_joined', sort=False)

    # keep only the good matches between taus and clusters
    dfCluTau = dfCluTau[dfCluTau['tau_Idx'] == dfCluTau['cl_tauMatchIdx']]



#######################################################################
######################### SCRIPT BODY #################################
#######################################################################

if __name__ == "__main__" :

    parser = OptionParser()
    parser.add_option("--indir",      dest="indir",     default=None)
    parser.add_option("--outdir",     dest="outdir",    default=None)
    parser.add_option("--tag",        dest="tag",       default=None)
    parser.add_option("--uJetPtCut",  dest="uJetPtCut", default=None)
    parser.add_option("--lJetPtCut",  dest="lJetPtCut", default=None)
    parser.add_option("--uTauPtCut",  dest="uTauPtCut", default=None)
    parser.add_option("--lTauPtCut",  dest="lTauPtCut", default=None)
    parser.add_option("--etacut",     dest="etacut",    default=None)
    parser.add_option('--caloClNxM',  dest='caloClNxM', default=None)
    (options, args) = parser.parse_args()

    NxM = options.caloClNxM

    readFrom = {
        'TowClus' : options.indir+'/L1Clusters/TowClus'+NxM+options.tag+'.pkl',
        'GenTaus' : options.indir+'/GenObjects/GenTaus'+options.tag+'.pkl',
        'GenJets' : options.indir+'/GenObjects/GenJets'+options.tag+'.pkl'
    }

    saveTo = {
        'TowClus' : options.outdir+'/TowClus'+NxM+options.tag+'.npz',
        'GenTaus' : options.outdir+'/GenTaus'+options.tag+'.npz',
        'GenJets' : options.outdir+'/GenJets'+options.tag+'.npz'
    }

    dfTowClus = pd.read_pickle(readFrom['TowClus'])
    dfGenTaus = pd.read_pickle(readFrom['GenTaus'])
    dfGenJets = pd.read_pickle(readFrom['GenJets'])

    TensorizeForCalibration(dfTowClus, dfGenTaus, options.uTauPtCut, options.lTauPtCut, options.etacut)
    # TensorizeForIdentification(dfTowClus, dfGenTaus, dfGenJets, options.uJetPtCut, options.lJetPtCut, options.uTauPtCut, options.lTauPtCut, options.etacut)

    print('** INFO : ALL DONE!')