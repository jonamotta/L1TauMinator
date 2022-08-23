from optparse import OptionParser
import pandas as pd
import numpy as np
import math
import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import mplhep
plt.style.use(mplhep.style.CMS)


def plotResolution(df, typ, res, caloClNxMs):
    colors_dict = {
        '9x9' : 'red',
        '7x7' : 'blue',
        '5x5' : 'yellow',
        '5x9' : 'orange',
        '5x7' : 'green',
        '3x7' : 'fuchsia',
        '3x5' : 'lime'
    }

    if typ == 'tau':
        Xlables = {
            'dEta'         : r'$\eta^{L1 \tau} - \eta^{Gen \tau}$',
            'dPhi'         : r'$\phi^{L1 \tau} - \phi^{Gen \tau}$',
            'dR'           : r'$\Delta R(\tau^{L1} - \tau^{Gen})$',
            'dPtOPt'       : r'$E_{T}^{L1 \tau} - p_T^{Gen \tau} / p_T^{Gen \tau}$',
            'dPtEmOPtEm'   : r'$E_{T}^{L1 \tau}(EM) - p_T^{Gen \tau}(EM) / p_T^{Gen \tau}(EM)$',
            'dPtHadOPtHad' : r'$E_{T}^{L1 \tau}(HAD) - p_T^{Gen \tau}(HAD) / p_T^{Gen \tau}(HAD)$'
        }
    else:
        Xlables = {
            'dEta'         : r'$\eta^{L1 jet} - \eta^{Gen jet}$',
            'dPhi'         : r'$\phi^{L1 jet} - \phi^{Gen jet}$',
            'dR'           : r'$\Delta R(jet^{L1} - jet^{Gen})$',
            'dPtOPt'       : r'$E_{T}^{L1 jet} - p_T^{Gen jet} / p_T^{Gen jet}$'
        }

    bins_dict = {
        'dEta'         : np.arange(-0.2,0.2,0.02),
        'dPhi'         : np.arange(-0.3,0.3,0.02),
        'dR'           : np.arange(0.,0.3,0.01),
        'dPtOPt'       : np.arange(-1.2,2,0.2),
        'dPtEmOPtEm'   : np.arange(-1.2,5,0.2),
        'dPtHadOPtHad' : np.arange(-1.2,2,0.2)
    }

    plt.figure(figsize=(10,8))
    # for NxM in caloClNxMs:
        # plt.hist(df[typ+'_cl'+NxM+'_'+res], bins=bins_dict[res], color=colors_dict[NxM], label='TowerCluster '+NxM, histtype='step', lw=2, density=True)
    plt.hist(df[typ+'_cl_'+res], bins=bins_dict[res], color=colors_dict[caloClNxMs], label='TowerCluster '+caloClNxMs, histtype='step', lw=2, density=True)
    plt.legend(loc = 'upper right', fontsize=14)
    plt.grid(linestyle=':')
    plt.xlabel(Xlables[res])
    plt.ylabel(r'a. u.')
    mplhep.cms.label('', data=False, rlabel='')
    if typ =='tau': plt.savefig(saveTo['TauDisplay']+'/raw_resolution_'+res+'.pdf')
    else:           plt.savefig(saveTo['JetDisplay']+'/raw_resolution_'+res+'.pdf')
    plt.close()


def plot2D(df, typ, var2plot, var2label, NxM):
    plt.figure(figsize=(10,8))
    plt.scatter(df[var2plot[0]], df[var2plot[1]], color='blue', label=r'Generator $\tau$', alpha=0.1)
    plt.scatter(df[var2plot[2]], df[var2plot[3]], color='red', label=r'TowerCluster '+NxM, alpha=0.1)
    plt.legend(loc = 'upper right', fontsize=14)
    plt.grid(linestyle=':')
    plt.xlabel(var2label[0])
    plt.ylabel(var2label[1])
    plt.xlim(-5,100)
    plt.ylim(-5,100)
    mplhep.cms.label('', data=False, rlabel='')
    if typ =='tau': plt.savefig(saveTo['TauDisplay']+'/raw_'+var2label[0]+'_vs_'+var2label[1]+'.pdf')
    else:           plt.savefig(saveTo['JetDisplay']+'/raw_'+var2label[0]+'_vs_'+var2label[1]+'.pdf')
    plt.close()

#######################################################################
######################### SCRIPT BODY #################################
#######################################################################

if __name__ == "__main__" :

    parser = OptionParser()
    parser.add_option("--indir",        dest="indir",                             default=None)
    parser.add_option("--tag",          dest="tag",                               default=None)
    parser.add_option("--uJetPtCut",    dest="uJetPtCut",                         default=None)
    parser.add_option("--lJetPtCut",    dest="lJetPtCut",                         default=None)
    parser.add_option("--uTauPtCut",    dest="uTauPtCut",                         default=None)
    parser.add_option("--lTauPtCut",    dest="lTauPtCut",                         default=None)
    parser.add_option("--etacut",       dest="etacut",                            default=None)
    parser.add_option('--caloClNxM',    dest='caloClNxM',                         default=None)
    parser.add_option('--doTauDisplay', dest='doTauDisplay', action='store_true', default=False)
    parser.add_option('--doJetDisplay', dest='doJetDisplay', action='store_true', default=False)
    (options, args) = parser.parse_args()

    if not options.doTauDisplay and not options.doJetDisplay:
        print('** ERROR : no display need specified')
        print('** EXITING')
        exit()

    outdir = options.indir+'/Resolutions'

    readFrom = {
        'TowClus' : options.indir+'/L1Clusters/TowClus'+options.caloClNxM+options.tag+'.pkl',
        'GenTaus' : options.indir+'/GenObjects/GenTaus'+options.tag+'.pkl',
        'GenJets' : options.indir+'/GenObjects/GenJets'+options.tag+'.pkl'
    }

    saveTo = {
        'TauDisplay' : outdir+'TausRaw/',
        'JetDisplay' : outdir+'JetsRaw/'
    }

    os.system('mkdir -p '+saveTo['TauDisplay'])
    os.system('mkdir -p '+saveTo['JetDisplay'])

    dfGenTaus = pd.read_pickle(readFrom['GenTaus'])
    dfGenJets = pd.read_pickle(readFrom['GenJets'])
    dfTowClus = pd.read_pickle(readFrom['TowClus'])

    # save unique identifier
    dfGenTaus['uniqueId'] = 'tau_'+dfGenTaus['event'].astype(str)+'_'+dfGenTaus['tau_Idx'].astype(str)
    dfGenJets['uniqueId'] = 'jet_'+dfGenJets['event'].astype(str)+'_'+dfGenJets['jet_Idx'].astype(str)

    # keep only the clusters that are matched to a tau
    dfTowClus = dfTowClus[dfTowClus['cl_tauMatchIdx'] >= 0]

    # join the taus and the clusters datasets -> this creates all the possible combination of clusters and taus for each event
    # important that dfFlatET is joined to dfFlatEJ and not viceversa --> this because dfFlatEJ contains the safe jets to be used and the safe event numbers
    dfGenTaus.set_index('event', inplace=True)
    dfGenJets.set_index('event', inplace=True)
    dfTowClus.set_index('event', inplace=True)
    dfCluTau = dfGenTaus.join(dfTowClus, on='event', how='left', rsuffix='_joined', sort=False)
    dfCluJet = dfGenJets.join(dfTowClus, on='event', how='left', rsuffix='_joined', sort=False)

    # keep only the good matches between taus and clusters
    dfCluTau = dfCluTau[dfCluTau['tau_Idx'] == dfCluTau['cl_tauMatchIdx']]
    dfCluJet = dfCluJet[dfCluJet['jet_Idx'] == dfCluJet['cl_jetMatchIdx']]

    dfCluTau['cl_totalEmpEg'] = dfCluTau['cl_totalEm'] + dfCluTau['cl_totalEgEt']

    dfCluTau['tau_cl_dEta'] = dfCluTau['cl_seedEta'] - dfCluTau['tau_visEta']
    dfCluTau['tau_cl_dPhi'] = dfCluTau['cl_seedPhi'] - dfCluTau['tau_visPhi']
    dfCluTau['tau_cl_dPt'] = dfCluTau['cl_totalEt'] - dfCluTau['tau_visPt']
    dfCluTau['tau_cl_dPtOPt'] = (dfCluTau['cl_totalEt'] - dfCluTau['tau_visPt']) /  dfCluTau['tau_visPt']
    dfCluTau['tau_cl_dPtEm'] = dfCluTau['cl_totalEm'] + dfCluTau['cl_totalEgEt'] - dfCluTau['tau_visPtEm']
    dfCluTau['tau_cl_dPtEmOPtEm'] = (dfCluTau['cl_totalEm'] + dfCluTau['cl_totalEgEt'] - dfCluTau['tau_visPtEm']) / dfCluTau['tau_visPtEm']
    dfCluTau['tau_cl_dPtHad'] = dfCluTau['cl_totalHad'] - dfCluTau['tau_visPtHad']
    dfCluTau['tau_cl_dPtHadOPtHad'] = (dfCluTau['cl_totalHad'] - dfCluTau['tau_visPtHad']) / dfCluTau['tau_visPtHad']

    dfCluJet['jet_cl_dEta'] = dfCluJet['cl_seedEta'] - dfCluJet['jet_eta']
    dfCluJet['jet_cl_dPhi'] = dfCluJet['cl_seedPhi'] - dfCluJet['jet_phi']
    dfCluJet['jet_cl_dPt'] = dfCluJet['cl_totalEt'] - dfCluJet['jet_pt']
    dfCluJet['jet_cl_dPtOPt'] = (dfCluJet['cl_totalEt'] - dfCluJet['jet_pt']) /  dfCluJet['jet_pt']

    if options.doTauDisplay:
        print('** INFO : doing display for taus')

        res2plot = ['dEta', 'dPhi', 'dPtOPt', 'dPtEmOPtEm', 'dPtHadOPtHad']

        for res in res2plot:
            plotResolution(dfCluTau, 'tau', res, options.caloClNxM)

            plot2D(dfCluTau, 'tau', ['tau_visPtEm', 'tau_visPtHad', 'cl_totalEmpEg', 'cl_totalHad'], ['em component', 'had component'], options.caloClNxM)
    
    if options.doJetDisplay:
        print('** INFO : doing display for jets')

        res2plot = ['dEta', 'dPhi', 'dR', 'dPtOPt']

        for res in res2plot:
            plotResolution(dfGenJets, 'jet', res, caloClNxMs)


    print('** INFO : ALL DONE!')