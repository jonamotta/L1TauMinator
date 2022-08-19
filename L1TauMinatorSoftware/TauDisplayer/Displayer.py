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

def colorbar_index(ncolors, cmap, label):
    cmap = cmap_discretize(cmap, ncolors)
    mappable = cm.ScalarMappable(cmap=cmap)
    mappable.set_array([])
    mappable.set_clim(-0.5, ncolors+0.5)
    colorbar = plt.colorbar(mappable, label=label)
    
    if ncolors > 10:
        nticks = 10
        colorbar.set_ticks(np.linspace(0, ncolors, nticks))
        colorbar.set_ticklabels(np.linspace(0, ncolors-1, nticks).astype('int64'))
        colorbar.ax.tick_params(which='minor', width=0, length=0)
    else:
        colorbar.set_ticks(np.linspace(0, ncolors, ncolors))
        colorbar.set_ticklabels(range(ncolors))
        colorbar.ax.tick_params(which='minor', width=0, length=0)

def cmap_discretize(cmap, N):
    """Return a discrete colormap from the continuous colormap cmap.

        cmap: colormap instance, eg. cm.jet. 
        N: number of colors.

    Example
        x = resize(arange(100), (5,100))
        djet = cmap_discretize(cm.jet, 5)
        imshow(x, cmap=djet)
    """

    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)
    colors_i = np.concatenate((np.linspace(0, 1., N), (0.,0.,0.,0.)))
    colors_rgba = cmap(colors_i)
    indices = np.linspace(0, 1., N+1)
    cdict = {}
    for ki,key in enumerate(('red','green','blue')):
        cdict[key] = [ (indices[i], colors_rgba[i-1,ki], colors_rgba[i,ki])
                       for i in range(N+1) ]
    # Return colormap object.
    return mcolors.LinearSegmentedColormap(cmap.name + "_%d"%N, cdict, 1024)


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

    # transform pt in hardware units
    dfGenJets['jet_hwPt'] = dfGenJets['jet_pt'].copy(deep=True) * 2

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

    # make the input tensors for the neural network
    dfCluJet.set_index('uniqueId',inplace=True)
    for i, idx in enumerate(dfCluJet.index):
        # progress
        if i%1 == 0:
            print(i/len(dfCluJet.index)*100, '%')

        if i == 25: break
        
        # need to transpose the arrays to have eta on x-axis and phi on y-axis
        HADdeposit = (np.array(dfCluJet.cl_towerIhad.loc[idx]).reshape(N,M))
        ECALdeposit  = (np.array(dfCluJet.cl_towerIem.loc[idx]).reshape(N,M))
        EGdeposit  = (np.array(dfCluJet.cl_towerEgIet.loc[idx]).reshape(N,M))
        EMdeposit = ECALdeposit + EGdeposit

        HADmax = np.max(dfCluJet.cl_towerIhad.loc[idx])
        EMmax  = np.max(dfCluJet.cl_towerIem.loc[idx])

        HADcolorbarTicks = [0, HADmax]
        EMcolorbarTicks  = [0, EMmax]

        etalabels = np.unique(dfCluJet.cl_towerIeta.loc[idx])
        philabels = np.unique(dfCluJet.cl_towerIphi.loc[idx])

        props = dict(boxstyle='square', facecolor='white')
        textstr1 = '\n'.join((
            r'$E_T^{%i\times%i}=%.0f$' % (N, M, dfCluJet.cl_totalIet.loc[idx], ),
            r'$p_T^{Gen}=%.2f$ GeV' % (dfCluJet.jet_hwPt.loc[idx], ),
            r'$\eta^{Gen}=%.2f$' % (dfCluJet.jet_eta.loc[idx], ),
            r'$\phi^{Gen}=%.2f$' % (dfCluJet.jet_phi.loc[idx], )))

        # N sets the eta range - M sets the phi range
        Xticksshifter = np.linspace(0.5,N-0.5,N)
        Yticksshifter = np.linspace(0.5,M-0.5,M)

        plt.figure(figsize=(10*(N+1)/9,8))
        im = plt.pcolormesh(EMdeposit, cmap=EMcmap, edgecolor='black', vmin=0)
        ncolors = max(2,EMmax+1)
        colorbar = plt.colorbar(im)
        colorbar.set_label(label='iem', fontsize=20)
        plt.clim(0., ncolors+0.5)
        if ncolors > 10:
            nticks = 10
            colorbar.set_ticks(np.linspace(0, ncolors, nticks))
            colorbar.set_ticklabels(np.linspace(0, ncolors, nticks).astype('int64'))
            colorbar.ax.tick_params(which='minor', width=0, length=0)
        else:
            colorbar.set_ticks(np.linspace(0, ncolors, ncolors))
            colorbar.set_ticklabels(range(ncolors))
            colorbar.ax.tick_params(which='minor', width=0, length=0)

        # colorbar_index(ncolors=max(2,EMmax+1), cmap=EMcmap, label='iem') # for discretized colorbar
        for i in range(EMdeposit.shape[0]):
            for j in range(EMdeposit.shape[1]):
                if EMdeposit[i, j] > 0: plt.text(j+0.5, i+0.5, format(EMdeposit[i, j], '.0f'), ha="center", va="center", fontsize=14, color='white' if EMdeposit[i, j] > EMmax*0.8 else "black")
        plt.xticks(Xticksshifter, etalabels)
        plt.yticks(Yticksshifter, philabels)
        plt.tick_params(which='both', width=0, length=0)
        plt.xlabel(f'$i\eta$', fontsize=20)
        plt.ylabel(f'$i\phi$', fontsize=20)
        plt.text(0.2, 8.8, textstr1, fontsize=14, verticalalignment='top',  bbox=props)
        mplhep.cms.label('', data=False, rlabel='', fontsize=20)
        plt.savefig(saveTo['JetDisplay']+'/'+idx+'_EMdeposit_clu'+str(N)+'x'+str(M)+'.pdf')
        plt.close()
        
        plt.figure(figsize=(10*(N+1)/9,8))
        im = plt.pcolormesh(HADdeposit, cmap=HADcmap, edgecolor='black', vmin=0)
        ncolors = max(2,HADmax+1)
        colorbar = plt.colorbar(im)
        colorbar.set_label(label='ihad', fontsize=20)
        plt.clim(-0.5, ncolors+0.5)
        if ncolors > 10:
            nticks = 10
            colorbar.set_ticks(np.linspace(0, ncolors, nticks))
            colorbar.set_ticklabels(np.linspace(0, ncolors-1, nticks).astype('int64'))
            colorbar.ax.tick_params(which='minor', width=0, length=0)
        else:
            colorbar.set_ticks(np.linspace(0, ncolors, ncolors))
            colorbar.set_ticklabels(range(ncolors))
            colorbar.ax.tick_params(which='minor', width=0, length=0)

        # colorbar_index(ncolors=max(2,HADmax+1), cmap=HADcmap, label='ihad') # for discretized colorbar
        for i in range(HADdeposit.shape[0]):
            for j in range(HADdeposit.shape[1]):
                if HADdeposit[i, j] > 0: plt.text(j+0.5, i+0.5, format(HADdeposit[i, j], '.0f'), ha="center", va="center", fontsize=14, color='white' if HADdeposit[i, j] > HADmax*0.8 else "black")
        plt.xticks(Xticksshifter, etalabels)
        plt.yticks(Yticksshifter, philabels)
        plt.tick_params(which='both', width=0, length=0)
        plt.xlabel(f'$i\eta$', fontsize=20)
        plt.ylabel(f'$i\phi$', fontsize=20)
        plt.text(0.2, 8.8, textstr1, fontsize=14, verticalalignment='top',  bbox=props)
        mplhep.cms.label('', data=False, rlabel='', fontsize=20)
        plt.savefig(saveTo['JetDisplay']+'/'+idx+'_HADdeposit_clu'+str(N)+'x'+str(M)+'.pdf')
        plt.close()


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

    # set colormaps
    HADcmap = cm.get_cmap('Blues')
    EMcmap = cm.get_cmap('Reds')

    # make the input tensors for the neural network
    dfCluTau.set_index('uniqueId',inplace=True)
    for i, idx in enumerate(dfCluTau.index):
        # progress
        if i%1 == 0:
            print(i/len(dfCluTau.index)*100, '%')
        
        # need to transpose the arrays to have eta on x-axis and phi on y-axis
        HADdeposit = (np.array(dfCluTau.cl_towerIhad.loc[idx]).reshape(N,M))
        ECALdeposit  = (np.array(dfCluTau.cl_towerIem.loc[idx]).reshape(N,M))
        EGdeposit  = (np.array(dfCluTau.cl_towerEgIet.loc[idx]).reshape(N,M))
        EMdeposit = ECALdeposit + EGdeposit

        HADmax = np.max(dfCluTau.cl_towerIhad.loc[idx])
        EMmax  = np.max(dfCluTau.cl_towerIem.loc[idx])

        HADcolorbarTicks = [0, HADmax]
        EMcolorbarTicks  = [0, EMmax]

        etalabels = np.unique(dfCluTau.cl_towerIeta.loc[idx])
        philabels = np.unique(dfCluTau.cl_towerIphi.loc[idx])

        props = dict(boxstyle='square', facecolor='white')
        textstr1 = '\n'.join((
            r'$E_T^{%i\times%i}=%.0f$' % (N, M, dfCluTau.cl_totalIet.loc[idx], ),
            r'$p_T^{Gen}=%.2f$ GeV' % (dfCluTau.tau_hwVisPt.loc[idx], ),
            r'$\eta^{Gen}=%.2f$' % (dfCluTau.tau_visEta.loc[idx], ),
            r'$\phi^{Gen}=%.2f$' % (dfCluTau.tau_visPhi.loc[idx], ),
            r'$DM=%i$' % (dfCluTau.tau_DM.loc[idx], )))

        # N sets the eta range - M sets the phi range
        Xticksshifter = np.linspace(0.5,N-0.5,N)
        Yticksshifter = np.linspace(0.5,M-0.5,M)

        plt.figure(figsize=(10*(N+1)/9,8*M/9))
        im = plt.pcolormesh(EMdeposit, cmap=EMcmap, edgecolor='black', vmin=0)
        ncolors = max(2,EMmax+1)
        colorbar = plt.colorbar(im)
        colorbar.set_label(label='iem', fontsize=20)
        plt.clim(0., ncolors+0.5)
        if ncolors > 10:
            nticks = 10
            colorbar.set_ticks(np.linspace(0, ncolors, nticks))
            colorbar.set_ticklabels(np.linspace(0, ncolors, nticks).astype('int64'))
            colorbar.ax.tick_params(which='minor', width=0, length=0)
        else:
            colorbar.set_ticks(np.linspace(0, ncolors, ncolors))
            colorbar.set_ticklabels(range(ncolors))
            colorbar.ax.tick_params(which='minor', width=0, length=0)

        # colorbar_index(ncolors=max(2,EMmax+1), cmap=EMcmap, label='iem') # for discretized colorbar
        for i in range(EMdeposit.shape[0]):
            for j in range(EMdeposit.shape[1]):
                if EMdeposit[i, j] > 0: plt.text(j+0.5, i+0.5, format(EMdeposit[i, j], '.0f'), ha="center", va="center", fontsize=14, color='white' if EMdeposit[i, j] > EMmax*0.8 else "black")
        plt.xticks(Xticksshifter, etalabels)
        plt.yticks(Yticksshifter, philabels)
        plt.tick_params(which='both', width=0, length=0)
        plt.xlabel(f'$i\eta$', fontsize=20)
        plt.ylabel(f'$i\phi$', fontsize=20)
        plt.text(0.2, 8.8, textstr1, fontsize=14, verticalalignment='top',  bbox=props)
        mplhep.cms.label('', data=False, rlabel='', fontsize=20)
        plt.savefig(saveTo['TauDisplay']+'/'+idx+'_EMdeposit_clu'+str(N)+'x'+str(M)+'.pdf')
        plt.close()
        
        plt.figure(figsize=(10*(N+1)/9,8*M/9))
        im = plt.pcolormesh(HADdeposit, cmap=HADcmap, edgecolor='black', vmin=0)
        ncolors = max(2,HADmax+1)
        colorbar = plt.colorbar(im)
        colorbar.set_label(label='ihad', fontsize=20)
        plt.clim(-0.5, ncolors+0.5)
        if ncolors > 10:
            nticks = 10
            colorbar.set_ticks(np.linspace(0, ncolors, nticks))
            colorbar.set_ticklabels(np.linspace(0, ncolors-1, nticks).astype('int64'))
            colorbar.ax.tick_params(which='minor', width=0, length=0)
        else:
            colorbar.set_ticks(np.linspace(0, ncolors, ncolors))
            colorbar.set_ticklabels(range(ncolors))
            colorbar.ax.tick_params(which='minor', width=0, length=0)

        # colorbar_index(ncolors=max(2,HADmax+1), cmap=HADcmap, label='ihad') # for discretized colorbar
        for i in range(HADdeposit.shape[0]):
            for j in range(HADdeposit.shape[1]):
                if HADdeposit[i, j] > 0: plt.text(j+0.5, i+0.5, format(HADdeposit[i, j], '.0f'), ha="center", va="center", fontsize=14, color='white' if HADdeposit[i, j] > HADmax*0.8 else "black")
        plt.xticks(Xticksshifter, etalabels)
        plt.yticks(Yticksshifter, philabels)
        plt.tick_params(which='both', width=0, length=0)
        plt.xlabel(f'$i\eta$', fontsize=20)
        plt.ylabel(f'$i\phi$', fontsize=20)
        plt.text(0.2, 8.8, textstr1, fontsize=14, verticalalignment='top',  bbox=props)
        mplhep.cms.label('', data=False, rlabel='', fontsize=20)
        plt.savefig(saveTo['TauDisplay']+'/'+idx+'_HADdeposit_clu'+str(N)+'x'+str(M)+'.pdf')
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