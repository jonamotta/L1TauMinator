from optparse import OptionParser
import pandas as pd
import numpy as np
import sys
import os

np.random.seed(7)

from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import mplhep
plt.style.use(mplhep.style.CMS)

class Logger(object):
    def __init__(self,file):
        self.terminal = sys.stdout
        self.log = open(file, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

def inspectWeights(model, which):
    if which=='kernel': idx=0
    if which=='bias':   idx=1

    allWeightsByLayer = {}
    for layer in model.layers:
        if (layer._name).find("batch")!=-1 or len(layer.get_weights())<1:
            continue 
        weights=layer.weights[idx].numpy().flatten()
        allWeightsByLayer[layer._name] = weights
        print('Layer {}: % of zeros = {}'.format(layer._name,np.sum(weights==0)/np.size(weights)))

    labelsW = []
    histosW = []

    for key in reversed(sorted(allWeightsByLayer.keys())):
        labelsW.append(key)
        histosW.append(allWeightsByLayer[key])

    fig = plt.figure(figsize=(10,10))
    bins = np.linspace(-0.4, 0.4, 50)
    plt.hist(histosW,bins,histtype='step',stacked=True,label=labelsW)
    plt.legend(frameon=False,loc='upper left', fontsize=16)
    plt.ylabel('Recurrence')
    plt.xlabel('Weight value')
    plt.xlim(-0.7,0.5)
    plt.yscale('log')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauMinator_CB_calib_plots/modelSparsity'+which+'.pdf')
    plt.close()

def save_obj(obj,dest):
    with open(dest,'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def deltaPhi(phi1, phi2):
    dPhi = abs(phi1 - phi2)
    if dPhi  < np.pi: return dPhi
    else:             return dPhi - 2*np.pi
vect_deltaPhi = np.vectorize(deltaPhi)

#######################################################################
######################### SCRIPT BODY #################################
#######################################################################

if __name__ == "__main__" :
    
    parser = OptionParser()
    parser.add_option("--v",            dest="v",                                default=None)
    parser.add_option("--date",         dest="date",                             default=None)
    parser.add_option("--inTag",        dest="inTag",                            default="")
    parser.add_option('--caloClNxM',    dest='caloClNxM',                        default="5x9")
    (options, args) = parser.parse_args()
    print(options)

    # get clusters' shape dimensions
    N = int(options.caloClNxM.split('x')[0])
    M = int(options.caloClNxM.split('x')[1])

    ############################## Get model inputs ##############################

    user = os.getcwd().split('/')[5]
    indir = '/data_CMS/cms/'+user+'/Phase2L1T/'+options.date+'_v'+options.v
    outdir = '/data_CMS/cms/'+user+'/Phase2L1T/'+options.date+'_v'+options.v+'/TauMinator_CB_cltw'+options.caloClNxM+'_Training'+options.inTag
    os.system('mkdir -p '+outdir+'/TauMinator_CB_calib_plots')

    # X1 is (None, N, M, 3)
    #       N runs over eta, M runs over phi
    #       3 features are: EgIet, Iem, Ihad
    # 
    # X2 is (None, 2)
    #       2 are eta and phi values
    #
    # Y is (None, 1)
    #       target: particel ID (tau = 1, non-tau = 0)

    X1 = np.load(outdir+'/tensors/images_train.npz')['arr_0']
    X2 = np.load(outdir+'/tensors/posits_train.npz')['arr_0']
    Y  = np.load(outdir+'/tensors/target_train.npz')['arr_0']
    Yid  = Y[:,1].reshape(-1,1)
    Ycal = Y[:,0].reshape(-1,1)

    ################################################################################
    # TEST ETA-PHI POSITION
    dEta = X2[:,0] - Y[:,2]
    dPhi = vect_deltaPhi(X2[:,1], Y[:,3])
    dR2 = dEta*dEta + dPhi*dPhi

    tau_sel = Yid.reshape(1,-1)[0] > 0
    dR2_id1 = dR2[tau_sel]
    pu_sel = Yid.reshape(1,-1)[0] < 1
    dR2_id0 = dR2[pu_sel]

    plt.figure(figsize=(10,10))
    # plt.hist(dR2,     bins=np.arange(0,20000,0.01), label="All", color='blue',  lw=2, histtype='step', alpha=0.7)
    plt.hist(dR2_id1, bins=np.arange(0,0.3,0.01), label="Id1", color='green', lw=2, histtype='step', alpha=0.7)
    plt.xlabel(r'$p_{T}^{Gen \tau}$')
    plt.ylabel(r'a.u.')
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle='dotted')
    plt.yscale('log')
    # plt.xlim(0,175)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig('./plotter_test/dR_id1.pdf')
    plt.close()

    plt.figure(figsize=(10,10))
    # plt.hist(dR2,     bins=np.arange(0,20000,0.01), label="All", color='blue',  lw=2, histtype='step', alpha=0.7)
    plt.hist(dR2_id0, bins=np.arange(15000,20000,0.01), label="Id0", color='red',   lw=2, histtype='step', alpha=0.7)
    plt.xlabel(r'$p_{T}^{Gen \tau}$')
    plt.ylabel(r'a.u.')
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle='dotted')
    plt.yscale('log')
    # plt.xlim(0,175)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig('./plotter_test/dR_id0.pdf')
    plt.close()

    ################################################################################
    # PREPROCESSING OF THE INPUTS

    # merge EG and EM in one single filter
    X1_new = np.zeros((len(X1), 5, 9, 2))
    X1_new[:,:,:,0] = (X1[:,:,:,0] + X1[:,:,:,1])
    X1_new[:,:,:,1] = X1[:,:,:,2]
    X1 = X1_new

    # select only objects below 150GeV
    pt_sel = np.sum(X1, (1,2,3)).reshape(1,-1)[0] > 60

    # pt_sel = Ycal.reshape(1,-1)[0] > 100

    # print(pt_sel)
    # exit()

    X1 = X1[pt_sel]
    X2 = X2[pt_sel]
    Y = Y[pt_sel]
    Ycal = Ycal[pt_sel]
    Yid = Yid[pt_sel]

    # select not too weird reconstruuctions
    # X1_totE = np.sum(X1, (1,2,3)).reshape(-1,1)
    # Yfactor = Ycal / X1_totE
    # farctor_sel = (Yfactor.reshape(1,-1)[0] < 2.4) & (Yfactor.reshape(1,-1)[0] > 0.4)
    # X1 = X1[farctor_sel]
    # X2 = X2[farctor_sel]
    # Y = Y[farctor_sel]
    # Ycal = Ycal[farctor_sel]
    # Yid = Yid[farctor_sel]
    # Yfactor = Yfactor[farctor_sel]

    # normalize everything
    X1 = X1 / 256.
    X2 = X2 / np.pi
    Ycal = Ycal / 256.
    rawPt = np.sum(X1, (1,2,3))

    ################################################################################
    # PLOT AVERAGE IMAGES

    tau_sel = Yid.reshape(1,-1)[0] > 0
    X1_id1 = X1[tau_sel]
    pu_sel = Yid.reshape(1,-1)[0] < 1
    X1_id0 = X1[pu_sel]

    averageImg_id1 = np.sum(X1_id1, (0))
    EMdeposit_id1 = averageImg_id1[:,:,0]
    HADdeposit_id1 = averageImg_id1[:,:,1]

    averageImg_id0 = np.sum(X1_id0, (0))
    EMdeposit_id0 = averageImg_id0[:,:,0]
    HADdeposit_id0 = averageImg_id0[:,:,1]

    HADcmap = cm.get_cmap('viridis')
    EMcmap = cm.get_cmap('viridis')

    # etalabels = np.unique(dfCluJet.cl_towerIeta.loc[idx])
    # philabels = np.unique(dfCluJet.cl_towerIphi.loc[idx])

    fig, axs = plt.subplots(1,2, figsize=(30,10))
    # plt.subplots_adjust(wspace=0.2)

    imEM = axs[0].pcolormesh(EMdeposit_id1, cmap=EMcmap, edgecolor='black', vmin=0)
    colorbar = plt.colorbar(imEM, ax=axs[0])
    colorbar.ax.tick_params(which='both', width=0, length=0)
    cbar_yticks = plt.getp(colorbar.ax.axes, 'yticklabels')
    plt.setp(cbar_yticks, color='w')
    colorbar.set_label(label=r'EM $E_T$')
    colorbar.ax.yaxis.set_label_coords(1.2,1)
    # for i in range(EMdeposit.shape[0]):
    #     for j in range(EMdeposit.shape[1]):
    #         if EMdeposit[i, j] >= 1.0: axs[1].text(j+0.5, i+0.5, format(EMdeposit[i, j], '.0f'), ha="center", va="center", fontsize=14, color='white' if EMdeposit[i, j] > EMmax*0.8 else "black")
    # axs[0].set_xticks(Xticksshifter)
    # axs[0].set_xticklabels(etalabels)
    # axs[0].set_yticks(Yticksshifter)
    # axs[0].set_yticklabels(philabels)
    axs[0].set_xlabel(r'$\eta$')
    axs[0].set_ylabel(r'$\phi$')
    axs[0].tick_params(which='both', width=0, length=0)

    imHAD = axs[1].pcolormesh(HADdeposit_id1, cmap=HADcmap, edgecolor='black', vmin=0)
    colorbar = plt.colorbar(imHAD, ax=axs[1])
    colorbar.ax.tick_params(which='both', width=0, length=0)
    cbar_yticks = plt.getp(colorbar.ax.axes, 'yticklabels')
    plt.setp(cbar_yticks, color='w')
    colorbar.set_label(label=r'HAD $E_T$')
    colorbar.ax.yaxis.set_label_coords(1.2,1)
    # for i in range(HADdeposit.shape[0]):
    #     for j in range(HADdeposit.shape[1]):
    #         if HADdeposit[i, j] >= 1.0: axs[2].text(j+0.5, i+0.5, format(HADdeposit[i, j], '.0f'), ha="center", va="center", fontsize=14, color='white' if HADdeposit[i, j] > HADmax*0.8 else "black")
    # axs[1].set_xticks(Xticksshifter)
    # axs[1].set_xticklabels(etalabels)
    # axs[1].set_yticks(Yticksshifter)
    # axs[1].set_yticklabels(philabels)
    axs[1].set_xlabel(r'$\eta$')
    axs[1].set_ylabel(r'$\phi$')
    axs[1].tick_params(which='both', width=0, length=0)
    
    fig.savefig('./plotter_test/average_sgn.pdf')


    fig, axs = plt.subplots(1,2, figsize=(30,10))
    # plt.subplots_adjust(wspace=0.2)

    imEM = axs[0].pcolormesh(EMdeposit_id0, cmap=EMcmap, edgecolor='black', vmin=0)
    colorbar = plt.colorbar(imEM, ax=axs[0])
    colorbar.ax.tick_params(which='both', width=0, length=0)
    cbar_yticks = plt.getp(colorbar.ax.axes, 'yticklabels')
    plt.setp(cbar_yticks, color='w')
    colorbar.set_label(label=r'EM $E_T$')
    colorbar.ax.yaxis.set_label_coords(1.2,1)
    # for i in range(EMdeposit.shape[0]):
    #     for j in range(EMdeposit.shape[1]):
    #         if EMdeposit[i, j] >= 1.0: axs[1].text(j+0.5, i+0.5, format(EMdeposit[i, j], '.0f'), ha="center", va="center", fontsize=14, color='white' if EMdeposit[i, j] > EMmax*0.8 else "black")
    # axs[0].set_xticks(Xticksshifter)
    # axs[0].set_xticklabels(etalabels)
    # axs[0].set_yticks(Yticksshifter)
    # axs[0].set_yticklabels(philabels)
    axs[0].set_xlabel(r'$\eta$')
    axs[0].set_ylabel(r'$\phi$')
    axs[0].tick_params(which='both', width=0, length=0)

    imHAD = axs[1].pcolormesh(HADdeposit_id0, cmap=HADcmap, edgecolor='black', vmin=0)
    colorbar = plt.colorbar(imHAD, ax=axs[1])
    colorbar.ax.tick_params(which='both', width=0, length=0)
    cbar_yticks = plt.getp(colorbar.ax.axes, 'yticklabels')
    plt.setp(cbar_yticks, color='w')
    colorbar.set_label(label=r'HAD $E_T$')
    colorbar.ax.yaxis.set_label_coords(1.2,1)
    # for i in range(HADdeposit.shape[0]):
    #     for j in range(HADdeposit.shape[1]):
    #         if HADdeposit[i, j] >= 1.0: axs[2].text(j+0.5, i+0.5, format(HADdeposit[i, j], '.0f'), ha="center", va="center", fontsize=14, color='white' if HADdeposit[i, j] > HADmax*0.8 else "black")
    # axs[1].set_xticks(Xticksshifter)
    # axs[1].set_xticklabels(etalabels)
    # axs[1].set_yticks(Yticksshifter)
    # axs[1].set_yticklabels(philabels)
    axs[1].set_xlabel(r'$\eta$')
    axs[1].set_ylabel(r'$\phi$')
    axs[1].tick_params(which='both', width=0, length=0)
    
    fig.savefig('./plotter_test/average_bkg.pdf')





































