from tensorflow.keras.initializers import RandomNormal as RN
from sklearn.linear_model import LinearRegression
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score
from optparse import OptionParser
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn import metrics
import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import cmsml
import sys
import os

np.random.seed(7)
tf.random.set_seed(7)

from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
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


#######################################################################
######################### SCRIPT BODY #################################
#######################################################################

if __name__ == "__main__" :
    
    parser = OptionParser()
    parser.add_option("--v",            dest="v",                                default=None)
    parser.add_option("--date",         dest="date",                             default=None)
    parser.add_option("--inTag",        dest="inTag",                            default="")
    parser.add_option('--caloClNxM',    dest='caloClNxM',                        default="5x9")
    parser.add_option('--train',        dest='train',       action='store_true', default=False)
    parser.add_option('--dm_weighted',  dest='dm_weighted', action='store_true', default=False)
    parser.add_option('--pt_weighted',  dest='pt_weighted', action='store_true', default=False)
    (options, args) = parser.parse_args()
    print(options)

    # get clusters' shape dimensions
    N = int(options.caloClNxM.split('x')[0])
    M = int(options.caloClNxM.split('x')[1])

    ############################## Get model inputs ##############################

    user = os.getcwd().split('/')[5]
    indir = '/data_CMS/cms/'+user+'/Phase2L1T/'+options.date+'_v'+options.v
    outdir = '/data_CMS/cms/'+user+'/Phase2L1T/'+options.date+'_v'+options.v+'/TauMinator_CB_cltw'+options.caloClNxM+'_Training'+options.inTag
    tag = ''
    if options.dm_weighted: tag = '_dmWeighted'
    if options.pt_weighted: tag = '_ptWeighted'
    outdir += tag
    os.system('mkdir -p '+outdir+'/TauMinator_CB_calib_plots')
    os.system('mkdir -p '+indir+'/CMSSWmodels'+tag)

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

    # select only taus
    tau_sel = Yid.reshape(1,-1)[0] > 0
    X1 = X1[tau_sel]
    X2 = X2[tau_sel]
    Y = Y[tau_sel]
    Ycal = Ycal[tau_sel]

    # select only below 150 GeV
    # pt_sel = Ycal.reshape(1,-1)[0] < 150
    # X1 = X1[pt_sel]
    # X2 = X2[pt_sel]
    # Y = Y[pt_sel]
    # Ycal = Ycal[pt_sel]

    # select not too weird reconstruuctions
    X1_totE = np.sum(X1, (1,2,3)).reshape(-1,1)
    Yfactor = Ycal / X1_totE
    farctor_sel = (Yfactor.reshape(1,-1)[0] < 2.4) & (Yfactor.reshape(1,-1)[0] > 0.4)
    X1 = X1[farctor_sel]
    X2 = X2[farctor_sel]
    Y = Y[farctor_sel]
    Ycal = Ycal[farctor_sel]
    Yfactor = Yfactor[farctor_sel]

    if options.dm_weighted:
        # create dm weights
        Ydm = Y[:,4].reshape(-1,1)
        dm0 = Ydm.reshape(1,-1)[0] == 0
        dm1 = (Ydm.reshape(1,-1)[0] == 1) | (Ydm.reshape(1,-1)[0] == 2)
        dm10 = Ydm.reshape(1,-1)[0] == 10
        dm11 = (Ydm.reshape(1,-1)[0] == 11) | (Ydm.reshape(1,-1)[0] == 12)

        dm0_w = np.sum(tau_sel) / np.sum(dm0)
        dm1_w = np.sum(tau_sel) / np.sum(dm1)
        dm10_w = np.sum(tau_sel) / np.sum(dm10)
        dm11_w = np.sum(tau_sel) / np.sum(dm11)

        dm_weights = dm0_w*dm0 + dm1_w*dm1 + dm10_w*dm10 + dm11_w*dm11

    if options.pt_weighted:
        def customPtDownWeight(row):
            if (row['gen_pt']<20):
                return row['weight']*0.5

            elif ((row['gen_pt']>100)&(row['gen_pt']<150)):
                return row['weight']*0.5
            
            elif (row['gen_pt']>150):
                return 1.0

            else:
                return row['weight']

        dfweights = pd.DataFrame(Y[:,0], columns=['gen_pt'])
        weight_Ebins = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90, 100, 110, 130, 150, 2000]
        dfweights['gen_pt_bin'] = pd.cut(dfweights['gen_pt'], bins=weight_Ebins, labels=False, include_lowest=True)
        dfweights['weight'] = dfweights.shape[0] / dfweights.groupby(['gen_pt_bin'])['gen_pt_bin'].transform('count')
        dfweights['weight'] = dfweights.apply(lambda row: customPtDownWeight(row) , axis=1)
        pt_weights = dfweights['weight'].to_numpy()

        plt.figure(figsize=(10,10))
        plt.hist(dfweights['gen_pt'], bins=weight_Ebins,                               label="Un-weighted", color='red',   lw=2, histtype='step', alpha=0.7)
        plt.hist(dfweights['gen_pt'], bins=weight_Ebins, weights=dfweights['weight'],  label="Weighted",    color='green', lw=2, histtype='step', alpha=0.7)
        plt.xlabel(r'$p_{T}^{Gen \tau}$')
        plt.ylabel(r'a.u.')
        plt.legend(loc = 'upper right', fontsize=16)
        plt.grid(linestyle='dotted')
        plt.yscale('log')
        plt.xlim(0,175)
        mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
        plt.savefig(outdir+'/TauMinator_CB_calib_plots/pt_gentau.pdf')
        plt.close()

        del dfweights

    # print(dfweights.groupby('gen_pt_bin')['weight'].mean())
    # exit()


    ############################## Models definition ##############################

    # This model calibrates the tau object:
    #    - one CNN that takes eg, em, had deposit images
    #    - one DNN that takes the flat output of the the CNN and the cluster position 

    def scale_precorrector(x, A,B,C):
        return  A * x**2 + B * x + C

    ## select not too weird reconstruuctions
    X1_totE = np.sum(X1, (1,2,3)).reshape(-1,1)
    Yfactor = Ycal / X1_totE
    farctor_sel = Yfactor.reshape(1,-1)[0] > 0.4
    X1_precorr = X1[farctor_sel]
    Ycal_precorr = Ycal[farctor_sel]

    dfscale = pd.DataFrame()
    dfscale['uncalib_pt'] = np.sum(X1_precorr, (1,2,3)).ravel()
    dfscale['gen_pt']     = Ycal_precorr.ravel()
    dfscale['uncalib_scale'] = dfscale['uncalib_pt'] / dfscale['gen_pt']
    dfscale['l1pt_bin'] = pd.cut(dfscale['uncalib_pt'], bins=np.arange(18,151,3), labels=False, include_lowest=True)
    uncalib = np.array(dfscale.groupby('l1pt_bin')['uncalib_scale'].mean())
    pt_bins_centers = np.array(np.arange(19.5,150,3))
    p0 = [1,1,1]
    popt, pcov = curve_fit(scale_precorrector, pt_bins_centers, uncalib, p0, maxfev=5000)
    dfscale['corrected_pt'] = dfscale['uncalib_pt'] / scale_precorrector(dfscale['uncalib_pt'], *popt)
    dfscale['corrected_scale'] = dfscale['corrected_pt'] / dfscale['gen_pt']
    corrected = np.array(dfscale.groupby('l1pt_bin')['corrected_scale'].mean())

    plt.figure(figsize=(10,10))
    plt.errorbar(pt_bins_centers, uncalib, xerr=1.5, label='Uncalibrated dataset', color='red', ls='None', lw=2, marker='o')
    plt.hlines(1, 0, 200, label='Ideal calibration', color='black', ls='--', lw=2)
    plt.plot(np.linspace(1,150,150), scale_precorrector(np.linspace(1,150,150), *popt), '-', label='_', lw=1.5, color='green')
    plt.errorbar(pt_bins_centers, corrected, xerr=1.5, label='Corrected dataset', color='blue', ls='None', lw=2, marker='o')
    plt.legend(loc = 'lower right', fontsize=16)
    plt.ylabel(r'L1 $p_{T}$ / Gen $p_{T}$')
    plt.xlabel(r'Gen $p_{T}$ [GeV]')
    plt.xlim(17, 151)
    plt.ylim(0.5, 1.5)
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauMinator_CB_calib_plots/scale_correction.pdf')
    plt.close()

    print('*****************************************')
    print('Scale correction function parameters')
    print('    function = A * x**2 + B * x + C')
    print('        A =', popt[0])
    print('        B =', popt[1])
    print('        C =', popt[2])
    print('*****************************************')

    dfscale = pd.DataFrame()
    dfscale['uncalib_pt'] = np.sum(X1_precorr, (1,2,3)).ravel()
    dfscale['gen_pt']     = Ycal_precorr.ravel()
    dfscale['uncalib_scale'] = dfscale['uncalib_pt'] / dfscale['gen_pt']
    dfscale['genpt_bin'] = pd.cut(dfscale['gen_pt'], bins=np.arange(18,151,3), labels=False, include_lowest=True)
    uncalib = np.array(dfscale.groupby('genpt_bin')['uncalib_scale'].mean())
    dfscale['corrected_pt'] = dfscale['uncalib_pt'] / scale_precorrector(dfscale['uncalib_pt'], *popt)
    dfscale['corrected_scale'] = dfscale['corrected_pt'] / dfscale['gen_pt']
    corrected = np.array(dfscale.groupby('genpt_bin')['corrected_scale'].mean())

    plt.figure(figsize=(10,10))
    plt.errorbar(pt_bins_centers, uncalib, xerr=1.5, label='Uncalibrated dataset', color='red', ls='None', lw=2, marker='o')
    plt.hlines(1, 0, 200, label='Ideal calibration', color='black', ls='--', lw=2)
    plt.errorbar(pt_bins_centers, corrected, xerr=1.5, label='Corrected dataset', color='blue', ls='None', lw=2, marker='o')
    plt.legend(loc = 'lower right', fontsize=16)
    plt.ylabel(r'L1 $p_{T}$ / Gen $p_{T}$')
    plt.xlabel(r'Gen $p_{T}$ [GeV]')
    plt.xlim(17, 151)
    plt.ylim(0.5, 1.5)
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauMinator_CB_calib_plots/scale_correction_vs_genpt.pdf')
    plt.close()


    ############################## Model validation and pots ##############################

    X1_valid = np.load(outdir+'/tensors/images_valid.npz')['arr_0']
    X2_valid = np.load(outdir+'/tensors/posits_valid.npz')['arr_0']
    Y_valid  = np.load(outdir+'/tensors/target_valid.npz')['arr_0']
    Yid_valid  = Y_valid[:,1].reshape(-1,1)
    Ycal_valid = Y_valid[:,0].reshape(-1,1)

    # select only taus
    tau_sel_valid = Yid_valid.reshape(1,-1)[0] > 0
    X1_valid = X1_valid[tau_sel_valid]
    X2_valid = X2_valid[tau_sel_valid]
    Y_valid = Y_valid[tau_sel_valid]
    Ycal_valid = Ycal_valid[tau_sel_valid]

    # select only below 150 GeV
    pt_sel = Ycal_valid.reshape(1,-1)[0] < 150
    X1_valid = X1_valid[pt_sel]
    X2_valid = X2_valid[pt_sel]
    Y_valid = Y_valid[pt_sel]
    Ycal_valid = Ycal_valid[pt_sel]

    # select not too weird reconstruuctions
    X1_totE_valid = np.sum(X1_valid, (1,2,3)).reshape(-1,1)
    Yfactor_valid = Ycal_valid / X1_totE_valid
    farctor_sel_valid = (Yfactor_valid.reshape(1,-1)[0] < 2.4) & (Yfactor_valid.reshape(1,-1)[0] > 0.4)
    X1_valid = X1_valid[farctor_sel_valid]
    X2_valid = X2_valid[farctor_sel_valid]
    Y_valid = Y_valid[farctor_sel_valid]
    Ycal_valid = Ycal_valid[farctor_sel_valid]
    Yfactor_valid = Yfactor_valid[farctor_sel_valid]

    dfTrain = pd.DataFrame()
    dfTrain['uncalib_pt'] = np.sum(X1, (1,2,3)).ravel()
    dfTrain['calib_pt']   = dfTrain['uncalib_pt'] / scale_precorrector(dfTrain['uncalib_pt'], *popt)
    dfTrain['gen_pt']     = Y[:,0].ravel()
    dfTrain['gen_eta']    = Y[:,2].ravel()
    dfTrain['gen_phi']    = Y[:,3].ravel()
    # dfTrain['gen_dm']     = Y[:,4].ravel()

    dfValid = pd.DataFrame()
    dfValid['uncalib_pt'] = np.sum(X1_valid, (1,2,3)).ravel()
    dfValid['calib_pt']   = dfValid['uncalib_pt'] / scale_precorrector(dfValid['uncalib_pt'], *popt)
    dfValid['gen_pt']     = Y_valid[:,0].ravel()
    dfValid['gen_eta']    = Y_valid[:,2].ravel()
    dfValid['gen_phi']    = Y_valid[:,3].ravel()
    # dfValid['gen_dm']     = Y_valid[:,4].ravel()

    # PLOTS INCLUSIVE
    plt.figure(figsize=(10,10))
    plt.hist(dfValid['uncalib_pt']/dfValid['gen_pt'], bins=np.arange(0.05,5,0.1), label=r'Uncalibrated response : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(dfValid['uncalib_pt']/dfValid['gen_pt']), np.std(dfValid['uncalib_pt']/dfValid['gen_pt'])),  color='red',  lw=2, density=True, histtype='step', alpha=0.7)
    plt.hist(dfTrain['calib_pt']/dfTrain['gen_pt'],   bins=np.arange(0.05,5,0.1), label=r'Train. Calibrated response : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(dfTrain['calib_pt']/dfTrain['gen_pt']), np.std(dfTrain['calib_pt']/dfTrain['gen_pt'])), color='blue', lw=2, density=True, histtype='step', alpha=0.7)
    plt.hist(dfValid['calib_pt']/dfValid['gen_pt'],   bins=np.arange(0.05,5,0.1), label=r'Valid. Calibrated response : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(dfValid['calib_pt']/dfValid['gen_pt']), np.std(dfValid['calib_pt']/dfValid['gen_pt'])), color='green',lw=2, density=True, histtype='step', alpha=0.7)
    plt.xlabel(r'$p_{T}^{L1 \tau} / p_{T}^{Gen \tau}$')
    plt.ylabel(r'a.u.')
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle='dotted')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauMinator_CB_calib_plots/responses_comparison.pdf')
    plt.close()

    # 2D REPOSNSE VS ETA
    plt.figure(figsize=(10,10))
    plt.scatter(dfValid['uncalib_pt'].head(1000)/dfValid['gen_pt'].head(1000), dfValid['gen_eta'].head(1000), label=r'Uncalibrated', alpha=0.2, color='red')
    plt.scatter(dfValid['calib_pt'].head(1000)/dfValid['gen_pt'].head(1000), dfValid['gen_eta'].head(1000), label=r'Calibrated', alpha=0.2, color='green')
    plt.xlabel(r'$p_{T}^{L1 \tau} / p_{T}^{Gen \tau}$')
    plt.ylabel(r'$|\eta^{Gen \tau}|$')
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle='dotted')
    plt.xlim(-0.1,5)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauMinator_CB_calib_plots/response_vs_eta_comparison.pdf')
    plt.close()

    # 2D REPOSNSE VS PHI
    plt.figure(figsize=(10,10))
    plt.scatter(dfValid['uncalib_pt'].head(1000)/dfValid['gen_pt'].head(1000), dfValid['gen_phi'].head(1000), label=r'Uncalibrated', alpha=0.2, color='red')
    plt.scatter(dfValid['calib_pt'].head(1000)/dfValid['gen_pt'].head(1000), dfValid['gen_phi'].head(1000), label=r'Calibrated', alpha=0.2, color='green')
    plt.xlabel(r'$p_{T}^{L1 \tau} / p_{T}^{Gen \tau}$')
    plt.ylabel(r'$\phi^{Gen \tau}$')
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle='dotted')
    plt.xlim(-0.1,5)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauMinator_CB_calib_plots/response_vs_phi_comparison.pdf')
    plt.close()

    # 2D REPOSNSE VS PT
    plt.figure(figsize=(10,10))
    plt.scatter(dfValid['uncalib_pt'].head(1000)/dfValid['gen_pt'].head(1000), dfValid['gen_pt'].head(1000), label=r'Uncalibrated', alpha=0.2, color='red')
    plt.scatter(dfValid['calib_pt'].head(1000)/dfValid['gen_pt'].head(1000), dfValid['gen_pt'].head(1000), label=r'Calibrated', alpha=0.2, color='green')
    plt.xlabel(r'$p_{T}^{L1 \tau} / p_{T}^{Gen \tau}$')
    plt.ylabel(r'$p_{T}^{Gen \tau}$')
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle='dotted')
    # plt.xlim(-0.1,5)
    plt.xlim(0.0,2.0)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauMinator_CB_calib_plots/response_vs_pt_comparison.pdf')
    plt.close()


    # L1 TO GEN MAPPING
    bins = np.arange(18,151,3)
    pt_bins_centers = np.array(np.arange(19.5,150,3))

    dfTrain['gen_pt_bin'] = pd.cut(dfTrain['gen_pt'], bins=bins, labels=False, include_lowest=True)
    dfValid['gen_pt_bin'] = pd.cut(dfValid['gen_pt'], bins=bins, labels=False, include_lowest=True)

    trainL1 = np.array(dfTrain.groupby('gen_pt_bin')['calib_pt'].mean())
    validL1 = np.array(dfValid.groupby('gen_pt_bin')['calib_pt'].mean())
    trainL1std = np.array(dfTrain.groupby('gen_pt_bin')['calib_pt'].std())
    validL1std = np.array(dfValid.groupby('gen_pt_bin')['calib_pt'].std())

    plt.figure(figsize=(10,10))
    plt.errorbar(pt_bins_centers, trainL1, yerr=trainL1std, label='Train. dataset', color='blue', ls='None', lw=2, marker='o')
    plt.errorbar(pt_bins_centers, validL1, yerr=validL1std, label='Valid. dataset', color='green', ls='None', lw=2, marker='o')
    plt.plot(np.array(np.arange(0,200,1)), np.array(np.arange(0,200,1)), label='Ideal calibration', color='black', ls='--', lw=2)
    plt.legend(loc = 'lower right', fontsize=16)
    plt.ylabel(r'L1 calibrated $p_{T}$ [GeV]')
    plt.xlabel(r'Gen $p_{T}$ [GeV]')
    plt.xlim(17, 151)
    plt.ylim(17, 151)
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauMinator_CB_calib_plots/GenToCalibL1_pt.pdf')
    plt.close()

    trainL1 = np.array(dfTrain.groupby('gen_pt_bin')['uncalib_pt'].mean())
    validL1 = np.array(dfValid.groupby('gen_pt_bin')['uncalib_pt'].mean())
    trainL1std = np.array(dfTrain.groupby('gen_pt_bin')['uncalib_pt'].std())
    validL1std = np.array(dfValid.groupby('gen_pt_bin')['uncalib_pt'].std())

    plt.figure(figsize=(10,10))
    plt.errorbar(pt_bins_centers, trainL1, yerr=trainL1std, label='Train. dataset', color='blue', ls='None', lw=2, marker='o')
    plt.errorbar(pt_bins_centers, validL1, yerr=validL1std, label='Valid. dataset', color='green', ls='None', lw=2, marker='o')
    plt.plot(np.array(np.arange(0,200,1)), np.array(np.arange(0,200,1)), label='Ideal calibration', color='black', ls='--', lw=2)
    plt.legend(loc = 'lower right', fontsize=16)
    plt.ylabel(r'L1 uncalibrated $p_{T}$ [GeV]')
    plt.xlabel(r'Gen $p_{T}$ [GeV]')
    plt.xlim(17, 151)
    plt.ylim(17, 151)
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauMinator_CB_calib_plots/GenToUncalinL1_pt.pdf')
    plt.close()

    scale_vs_pt_uncalib = []
    resol_vs_pt_uncalib = []
    scale_vs_pt_calib = []
    resol_vs_pt_calib = []

    for ibin in np.sort(dfValid['gen_pt_bin'].unique()):
        ledge = bins[ibin]
        uedge = bins[ibin+1]

        tmpTrain = dfTrain[dfTrain['gen_pt_bin']==ibin]
        tmpValid = dfValid[dfValid['gen_pt_bin']==ibin]

        plt.figure(figsize=(10,10))
        plt.hist(tmpValid['uncalib_pt']/tmpValid['gen_pt'], bins=np.arange(0.05,5,0.1), label=r'Uncalibrated response : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(tmpValid['uncalib_pt']/tmpValid['gen_pt']), np.std(tmpValid['uncalib_pt']/tmpValid['gen_pt'])),  color='red',  lw=2, density=True, histtype='step', alpha=0.7)
        plt.hist(tmpTrain['calib_pt']/tmpTrain['gen_pt'],   bins=np.arange(0.05,5,0.1), label=r'Train. Calibrated response : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(tmpTrain['calib_pt']/tmpTrain['gen_pt']), np.std(tmpTrain['calib_pt']/tmpTrain['gen_pt'])), color='blue', lw=2, density=True, histtype='step', alpha=0.7)
        plt.hist(tmpValid['calib_pt']/tmpValid['gen_pt'],   bins=np.arange(0.05,5,0.1), label=r'Valid. Calibrated response : $\mu$ = %.2f, $\sigma$ =  %.2f' % (np.mean(tmpValid['calib_pt']/tmpValid['gen_pt']), np.std(tmpValid['calib_pt']/tmpValid['gen_pt'])), color='green',lw=2, density=True, histtype='step', alpha=0.7)
        plt.xlabel(r'$p_{T}^{L1 \tau} / p_{T}^{Gen \tau}$')
        plt.ylabel(r'a.u.')
        plt.legend(loc = 'upper right', fontsize=16)
        plt.grid(linestyle='dotted')
        mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
        plt.savefig(outdir+'/TauMinator_CB_calib_plots/responses_comparison_'+str(ledge)+'pt'+str(uedge)+'.pdf')
        plt.close()

        scale_vs_pt_uncalib.append(np.mean(tmpValid['uncalib_pt']/tmpValid['gen_pt']))
        resol_vs_pt_uncalib.append(np.std(tmpValid['uncalib_pt']/tmpValid['gen_pt'])/np.mean(tmpValid['uncalib_pt']/tmpValid['gen_pt']))

        scale_vs_pt_calib.append(np.mean(tmpValid['calib_pt']/tmpValid['gen_pt']))
        resol_vs_pt_calib.append(np.std(tmpValid['calib_pt']/tmpValid['gen_pt'])/np.mean(tmpValid['uncalib_pt']/tmpValid['gen_pt']))


    dfTrain['uncalib_scale'] = dfTrain['uncalib_pt'] / dfTrain['gen_pt']
    dfTrain['calib_scale'] = dfTrain['calib_pt'] / dfTrain['gen_pt']
    dfTrain['l1pt_bin'] = pd.cut(dfTrain['calib_pt'], bins=bins, labels=False, include_lowest=True)
    uncalib = np.array(dfTrain.groupby('l1pt_bin')['uncalib_scale'].mean())
    calib = np.array(dfTrain.groupby('l1pt_bin')['calib_scale'].mean())

    # scale vs pt
    plt.figure(figsize=(10,10))
    plt.errorbar(pt_bins_centers, uncalib, xerr=1.5, label=r'Uncalibrated',  color='red',  lw=2, alpha=0.7, marker='o', ls='None')
    plt.errorbar(pt_bins_centers, calib, xerr=1.5, label=r'Calibrated',  color='green',  lw=2, alpha=0.7, marker='o', ls='None')
    plt.hlines(1, 0, 200, label='Ideal calibration', color='black', ls='--', lw=2)
    plt.xlabel(r'$p_{T}^{L1 \tau}$')
    plt.ylabel(r'Scale')
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle='dotted')
    plt.xlim(17,151)
    plt.ylim(0.5,1.5)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauMinator_CB_calib_plots/scale_vs_l1pt.pdf')
    plt.close()

    # scale vs pt
    plt.figure(figsize=(10,10))
    plt.errorbar(pt_bins_centers, scale_vs_pt_uncalib, xerr=1.5, label=r'Uncalibrated',  color='red',  lw=2, alpha=0.7, marker='o', ls='None')
    plt.errorbar(pt_bins_centers, scale_vs_pt_calib, xerr=1.5,   label=r'Calibrated',    color='green', lw=2, alpha=0.7, marker='o', ls='None')
    plt.hlines(1, 0, 200, label='Ideal calibration', color='black', ls='--', lw=2)
    plt.xlabel(r'$p_{T}^{Gen \tau}$')
    plt.ylabel(r'Scale')
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle='dotted')
    plt.xlim(17,151)
    plt.ylim(0.5,1.5)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauMinator_CB_calib_plots/scale_vs_pt.pdf')
    plt.close()

    # resolution vs pt
    plt.figure(figsize=(10,10))
    plt.errorbar(pt_bins_centers, resol_vs_pt_uncalib, xerr=1.5, label=r'Uncalibrated',  color='red',  lw=2, alpha=0.7, marker='o', ls='None')
    plt.errorbar(pt_bins_centers, resol_vs_pt_calib, xerr=1.5,   label=r'Calibrated',    color='green', lw=2, alpha=0.7, marker='o', ls='None')
    plt.hlines(1, 0, 200, label='Ideal calibration', color='black', ls='--', lw=2)
    plt.xlabel(r'$p_{T}^{Gen \tau}$')
    plt.ylabel(r'Resolution')
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle='dotted')
    plt.xlim(17,151)
    plt.ylim(0.1,0.4)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauMinator_CB_calib_plots/resolution_vs_pt.pdf')
    plt.close()
