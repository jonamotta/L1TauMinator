from optparse import OptionParser
import pandas as pd
import numpy as np
import math
import os

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import mplhep
plt.style.use(mplhep.style.CMS)


def DisplayForJets(dfFlatTowClus, dfFlatGenJets, uJetPtCut, lJetPtCut, etacut, NxM, saveTo):
    if len(dfFlatTowClus) == 0 or len(dfFlatGenJets) == 0:
        print('** ERROR : no tensorization need specified')
        print('** EXITING')
        return

    dfGenJets = dfFlatGenJets.copy(deep=True)
    dfTowClus = dfFlatTowClus.copy(deep=True)

    # get clusters' shape dimensions
    N = int(NxM.split('x')[0])
    M = int(NxM.split('x')[1])

    # Apply cut on jet pt
    if uJetPtCut:
        dfGenJets = dfGenJets[dfGenJets['jet_pt'] <= float(uJetPtCut)]
    if lJetPtCut:
        dfGenJets = dfGenJets[dfGenJets['jet_pt'] >= float(lJetPtCut)]

    # Apply cut on jet eta
    if etacut:
        dfGenJets = dfGenJets[dfGenJets['jet_eta'] <= float(etacut)]

    # save unique identifier
    dfGenJets['uniqueId'] = 'jet_'+dfGenJets['event'].astype(str)+'_'+dfGenJets['jet_Idx'].astype(str)

    # keep only the clusters that are matched to a jet
    dfTowClus = dfTowClus[dfTowClus['cl_jetMatchIdx'] >= 0]

    # join the jets and the clusters datasets -> this creates all the possible combination of clusters and jets for each event
    # important that dfFlatET is joined to dfFlatEJ and not viceversa --> this because dfFlatEJ contains the safe jets to be used and the safe event numbers
    dfGenJets.set_index('event', inplace=True)
    dfTowClus.set_index('event', inplace=True)
    dfCluJet = dfGenJets.join(dfTowClus, on='event', how='left', rsuffix='_joined', sort=False)

    # keep only the good matches between jets and clusters
    dfCluJet = dfCluJet[dfCluJet['jet_Idx'] == dfCluJet['cl_jetMatchIdx']]

    # shuffle the rows so that no possible order gets learned
    dfCluJet = dfCluJet.sample(frac=1).copy(deep=True)

    # set colormaps
    HADcmap = cm.get_cmap('Blues')
    EMcmap = cm.get_cmap('Reds')
    EGcmap = cm.get_cmap('Greens')

    # make the input tensors for the neural network
    dfCluJet.set_index('uniqueId',inplace=True)
    cnt = 0
    for i, idx in enumerate(dfCluJet.index):
        # progress
        if i%1 == 0:
            print(i/len(dfCluJet.index)*100, '%')

        # for some reason some events have some problems with some barrel towers getting ieta=-1016 and iphi=-962 --> skip out-of-shape TowerClusters
        if len(dfCluJet.cl_towerIeta.loc[idx]) != N*M: continue

        if cnt == 30: break
        cnt += 1

        # need to transpose the arrays to have eta on x-axis and phi on y-axis
        HADdeposit = np.transpose(np.array(dfCluJet.cl_towerHad.loc[idx]).reshape(N,M))
        EMdeposit  = np.transpose(np.array(dfCluJet.cl_towerEm.loc[idx]).reshape(N,M))
        EGdeposit  = np.transpose(np.array(dfCluJet.cl_towerEgEt.loc[idx]).reshape(N,M))

        if cnt == 30: break
        cnt += 1

        HADmax = np.max(dfCluJet.cl_towerHad.loc[idx])
        EMmax  = np.max(dfCluJet.cl_towerEm.loc[idx])
        EGmax  = np.max(dfCluJet.cl_towerEgEt.loc[idx])

        HADcolorbarTicks = [0, HADmax]
        EMcolorbarTicks  = [0, EMmax]
        EGcolorbarTicks  = [0, EGmax]

        etalabels = np.unique(dfCluJet.cl_towerIeta.loc[idx])
        philabels = np.unique(dfCluJet.cl_towerIphi.loc[idx])

        props = dict(boxstyle='square', facecolor='white', edgecolor='black')
        textstr1 = '\n'.join((
            r'$p_T^{Gen}=%.2f$ GeV' % (dfCluJet.jet_pt.loc[idx], ),
            r'$\eta^{Gen}=%.2f$' % (dfCluJet.jet_eta.loc[idx], ),
            r'$\phi^{Gen}=%.2f$' % (dfCluJet.jet_phi.loc[idx] )))

        # N sets the eta range - M sets the phi range
        Xticksshifter = np.linspace(0.5,N-0.5,N)
        Yticksshifter = np.linspace(0.5,M-0.5,M)

        fig, axs = plt.subplots(1,3, figsize=(40,10))
        # plt.subplots_adjust(wspace=0.2)
        
        imEG = axs[0].pcolormesh(EGdeposit, cmap=EGcmap, edgecolor='black', vmin=0)
        axs[0].text(0.2, N-0.2, textstr1, fontsize=14, verticalalignment='top',  bbox=props)
        colorbar = plt.colorbar(imEG, ax=axs[0])
        colorbar.ax.tick_params(which='both', width=0, length=0)
        cbar_yticks = plt.getp(colorbar.ax.axes, 'yticklabels')
        plt.setp(cbar_yticks, color='w')
        colorbar.set_label(label=r'EG $E_T$')
        colorbar.ax.yaxis.set_label_coords(1.2,1)
        for i in range(EGdeposit.shape[0]):
            for j in range(EGdeposit.shape[1]):
                if EGdeposit[i, j] >= 1.0: axs[0].text(j+0.5, i+0.5, format(EGdeposit[i, j], '.0f'), ha="center", va="center", fontsize=14, color='white' if EGdeposit[i, j] > EGmax*0.8 else "black")
        axs[0].set_xticks(Xticksshifter)
        axs[0].set_xticklabels(etalabels)
        axs[0].set_yticks(Yticksshifter)
        axs[0].set_yticklabels(philabels)
        axs[0].set_xlabel(r'$\eta$')
        axs[0].set_ylabel(r'$\phi$')
        axs[0].tick_params(which='both', width=0, length=0)

        imEM = axs[1].pcolormesh(EMdeposit, cmap=EMcmap, edgecolor='black', vmin=0)
        colorbar = plt.colorbar(imEM, ax=axs[1])
        colorbar.ax.tick_params(which='both', width=0, length=0)
        cbar_yticks = plt.getp(colorbar.ax.axes, 'yticklabels')
        plt.setp(cbar_yticks, color='w')
        colorbar.set_label(label=r'EM $E_T$')
        colorbar.ax.yaxis.set_label_coords(1.2,1)
        for i in range(EMdeposit.shape[0]):
            for j in range(EMdeposit.shape[1]):
                if EMdeposit[i, j] >= 1.0: axs[1].text(j+0.5, i+0.5, format(EMdeposit[i, j], '.0f'), ha="center", va="center", fontsize=14, color='white' if EMdeposit[i, j] > EMmax*0.8 else "black")
        axs[1].set_xticks(Xticksshifter)
        axs[1].set_xticklabels(etalabels)
        axs[1].set_yticks(Yticksshifter)
        axs[1].set_yticklabels(philabels)
        axs[1].set_xlabel(r'$\eta$')
        axs[1].set_ylabel(r'$\phi$')
        axs[1].tick_params(which='both', width=0, length=0)

        imHAD = axs[2].pcolormesh(HADdeposit, cmap=HADcmap, edgecolor='black', vmin=0)
        colorbar = plt.colorbar(imHAD, ax=axs[2])
        colorbar.ax.tick_params(which='both', width=0, length=0)
        cbar_yticks = plt.getp(colorbar.ax.axes, 'yticklabels')
        plt.setp(cbar_yticks, color='w')
        colorbar.set_label(label=r'HAD $E_T$')
        colorbar.ax.yaxis.set_label_coords(1.2,1)
        for i in range(HADdeposit.shape[0]):
            for j in range(HADdeposit.shape[1]):
                if HADdeposit[i, j] >= 1.0: axs[2].text(j+0.5, i+0.5, format(HADdeposit[i, j], '.0f'), ha="center", va="center", fontsize=14, color='white' if HADdeposit[i, j] > HADmax*0.8 else "black")
        axs[2].set_xticks(Xticksshifter)
        axs[2].set_xticklabels(etalabels)
        axs[2].set_yticks(Yticksshifter)
        axs[2].set_yticklabels(philabels)
        axs[2].set_xlabel(r'$\eta$')
        axs[2].set_ylabel(r'$\phi$')
        axs[2].tick_params(which='both', width=0, length=0)
        
        fig.savefig(saveTo['JetDisplay']+'/'+idx+'_footprint_clu'+str(N)+'x'+str(M)+'.pdf')


def DisplayForTaus(dfFlatTowClus, dfFlatGenTaus, uTauPtCut, lTauPtCut, etacut, NxM, saveTo):
    if len(dfFlatTowClus) == 0 or len(dfFlatGenTaus) == 0:
        print('** ERROR : no tensorization need specified')
        print('** EXITING')
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
        dfGenTaus = dfGenTaus[dfGenTaus['tau_visPt'] <= float(uTauPtCut)]
    if lTauPtCut:
        dfGenTaus = dfGenTaus[dfGenTaus['tau_visPt'] >= float(lTauPtCut)]

    # Apply cut on tau eta
    if etacut:
        dfGenTaus = dfGenTaus[dfGenTaus['tau_visEta'] <= float(etacut)]

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

    # set colormaps
    HADcmap = cm.get_cmap('Blues')
    EMcmap = cm.get_cmap('Reds')
    EGcmap = cm.get_cmap('Greens')

    # make the input tensors for the neural network
    dfCluTau.set_index('uniqueId',inplace=True)
    cnt = 0
    for i, idx in enumerate(dfCluTau.index):
        # progress
        if i%1 == 0:
            print(i/len(dfCluTau.index)*100, '%')
        
        # for some reason some events have some problems with some barrel towers getting ieta=-1016 and iphi=-962 --> skip out-of-shape TowerClusters
        if len(dfCluTau.cl_towerIeta.loc[idx]) != N*M: continue

        if cnt == 30: break
        cnt += 1

        # need to transpose the arrays to have eta on x-axis and phi on y-axis
        HADdeposit = np.transpose(np.array(dfCluTau.cl_towerHad.loc[idx]).reshape(N,M))
        EMdeposit  = np.transpose(np.array(dfCluTau.cl_towerEm.loc[idx]).reshape(N,M))
        EGdeposit  = np.transpose(np.array(dfCluTau.cl_towerEgEt.loc[idx]).reshape(N,M))

        if cnt == 30: break
        cnt += 1

        HADmax = np.max(dfCluTau.cl_towerHad.loc[idx])
        EMmax  = np.max(dfCluTau.cl_towerEm.loc[idx])
        EGmax  = np.max(dfCluTau.cl_towerEgEt.loc[idx])

        HADcolorbarTicks = [0, HADmax]
        EMcolorbarTicks  = [0, EMmax]
        EGcolorbarTicks  = [0, EGmax]

        etalabels = np.unique(dfCluTau.cl_towerIeta.loc[idx])
        philabels = np.unique(dfCluTau.cl_towerIphi.loc[idx])

        DMdict = {
            0  : r'h^{\pm}',
            1  : r'h^{\pm}\pi^{0}',
            2  : r'h^{\pm}\pi^{0}',
            10 : r'h^{\pm}h^{\mp}h^{\pm}',
            11 : r'h^{\pm}h^{\mp}h^{\pm}\pi^{0}',
            12 : r'h^{\pm}h^{\mp}h^{\pm}\pi^{0}'
        }

        props = dict(boxstyle='square', facecolor='white', edgecolor='black')
        textstr1 = '\n'.join((
            r'$p_T^{Gen}=%.2f$ GeV' % (dfCluTau.tau_visPt.loc[idx], ),
            r'$\eta^{Gen}=%.2f$' % (dfCluTau.tau_visEta.loc[idx], ),
            r'$\phi^{Gen}=%.2f$' % (dfCluTau.tau_visPhi.loc[idx], ),
            r'$DM=%s$' % (DMdict[dfCluTau.tau_DM.loc[idx]] )))

        # N sets the eta range - M sets the phi range
        Xticksshifter = np.linspace(0.5,N-0.5,N)
        Yticksshifter = np.linspace(0.5,M-0.5,M)

        fig, axs = plt.subplots(1,3, figsize=(40,10))
        # plt.subplots_adjust(wspace=0.2)
        
        imEG = axs[0].pcolormesh(EGdeposit, cmap=EGcmap, edgecolor='black', vmin=0)
        axs[0].text(0.2, N-0.2, textstr1, fontsize=14, verticalalignment='top',  bbox=props)
        colorbar = plt.colorbar(imEG, ax=axs[0])
        colorbar.ax.tick_params(which='both', width=0, length=0)
        cbar_yticks = plt.getp(colorbar.ax.axes, 'yticklabels')
        plt.setp(cbar_yticks, color='w')
        colorbar.set_label(label=r'EG $E_T$')
        colorbar.ax.yaxis.set_label_coords(1.2,1)
        for i in range(EGdeposit.shape[0]):
            for j in range(EGdeposit.shape[1]):
                if EGdeposit[i, j] >= 1.0: axs[0].text(j+0.5, i+0.5, format(EGdeposit[i, j], '.0f'), ha="center", va="center", fontsize=14, color='white' if EGdeposit[i, j] > EGmax*0.8 else "black")
        axs[0].set_xticks(Xticksshifter)
        axs[0].set_xticklabels(etalabels)
        axs[0].set_yticks(Yticksshifter)
        axs[0].set_yticklabels(philabels)
        axs[0].set_xlabel(r'$\eta$')
        axs[0].set_ylabel(r'$\phi$')
        axs[0].tick_params(which='both', width=0, length=0)

        imEM = axs[1].pcolormesh(EMdeposit, cmap=EMcmap, edgecolor='black', vmin=0)
        colorbar = plt.colorbar(imEM, ax=axs[1])
        colorbar.ax.tick_params(which='both', width=0, length=0)
        cbar_yticks = plt.getp(colorbar.ax.axes, 'yticklabels')
        plt.setp(cbar_yticks, color='w')
        colorbar.set_label(label=r'EM $E_T$')
        colorbar.ax.yaxis.set_label_coords(1.2,1)
        for i in range(EMdeposit.shape[0]):
            for j in range(EMdeposit.shape[1]):
                if EMdeposit[i, j] >= 1.0: axs[1].text(j+0.5, i+0.5, format(EMdeposit[i, j], '.0f'), ha="center", va="center", fontsize=14, color='white' if EMdeposit[i, j] > EMmax*0.8 else "black")
        axs[1].set_xticks(Xticksshifter)
        axs[1].set_xticklabels(etalabels)
        axs[1].set_yticks(Yticksshifter)
        axs[1].set_yticklabels(philabels)
        axs[1].set_xlabel(r'$\eta$')
        axs[1].set_ylabel(r'$\phi$')
        axs[1].tick_params(which='both', width=0, length=0)

        imHAD = axs[2].pcolormesh(HADdeposit, cmap=HADcmap, edgecolor='black', vmin=0)
        colorbar = plt.colorbar(imHAD, ax=axs[2])
        colorbar.ax.tick_params(which='both', width=0, length=0)
        cbar_yticks = plt.getp(colorbar.ax.axes, 'yticklabels')
        plt.setp(cbar_yticks, color='w')
        colorbar.set_label(label=r'HAD $E_T$')
        colorbar.ax.yaxis.set_label_coords(1.2,1)
        for i in range(HADdeposit.shape[0]):
            for j in range(HADdeposit.shape[1]):
                if HADdeposit[i, j] >= 1.0: axs[2].text(j+0.5, i+0.5, format(HADdeposit[i, j], '.0f'), ha="center", va="center", fontsize=14, color='white' if HADdeposit[i, j] > HADmax*0.8 else "black")
        axs[2].set_xticks(Xticksshifter)
        axs[2].set_xticklabels(etalabels)
        axs[2].set_yticks(Yticksshifter)
        axs[2].set_yticklabels(philabels)
        axs[2].set_xlabel(r'$\eta$')
        axs[2].set_ylabel(r'$\phi$')
        axs[2].tick_params(which='both', width=0, length=0)
        
        fig.savefig(saveTo['TauDisplay']+'/'+idx+'_footprint_clu'+str(N)+'x'+str(M)+'.pdf')

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

    outdir = options.indir+'/Footprints'

    readFrom = {
        'TowClus' : options.indir+'/L1Clusters/TowClus'+options.caloClNxM+options.tag+'.pkl',
        'GenTaus' : options.indir+'/GenObjects/GenTaus'+options.tag+'.pkl',
        'GenJets' : options.indir+'/GenObjects/GenJets'+options.tag+'.pkl'
    }

    saveTo = {
        'TauDisplay' : outdir+'Taus'+options.caloClNxM+'/',
        'JetDisplay' : outdir+'Jets'+options.caloClNxM+'/'
    }
    os.system('mkdir -p '+saveTo['TauDisplay'])
    os.system('mkdir -p '+saveTo['JetDisplay'])

    dfTowClus = pd.read_pickle(readFrom['TowClus'])
    dfGenTaus = pd.read_pickle(readFrom['GenTaus'])
    dfGenJets = pd.read_pickle(readFrom['GenJets'])

    if not options.doTauDisplay and not options.doJetDisplay:
        print('** ERROR : no display need specified')
        print('** EXITING')
        exit()

    if options.doTauDisplay:
        print('** INFO : doing display for taus')
        DisplayForTaus(dfTowClus, dfGenTaus, options.uTauPtCut, options.lTauPtCut, options.etacut, options.caloClNxM, saveTo)
    
    if options.doJetDisplay:
        print('** INFO : doing display for jets')
        DisplayForJets(dfTowClus, dfGenJets, options.uJetPtCut, options.lJetPtCut, options.etacut, options.caloClNxM, saveTo)

    print('** INFO : ALL DONE!')