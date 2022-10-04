from scipy.optimize import curve_fit
from optparse import OptionParser
from scipy.special import btdtri # beta quantile function
from tensorflow import keras
from sklearn import metrics
import tensorflow as tf
import xgboost as xgb
import pandas as pd
import numpy as np
import pickle
import os

import matplotlib.pyplot as plt
import matplotlib
import mplhep
plt.style.use(mplhep.style.CMS)

np.random.seed(7)
tf.random.set_seed(7)


def load_obj(source):
    with open(source,'rb') as f:
        return pickle.load(f)


#######################################################################
######################### SCRIPT BODY #################################
#######################################################################

if __name__ == "__main__" :
    
    parser = OptionParser()
    parser.add_option("--v",            dest="v",         default=None)
    parser.add_option("--date",         dest="date",      default=None)
    parser.add_option('--caloClNxM',    dest='caloClNxM', default="5x9")
    (options, args) = parser.parse_args()
    print(options)

    # get clusters' shape dimensions
    N = int(options.caloClNxM.split('x')[0])
    M = int(options.caloClNxM.split('x')[1])

    ############################## Get models and inputs ##############################

    indir = '/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v
    perfdir = indir+'/CMSSWNtuplized/CMSSWCrossChecked_'+options.caloClNxM
    os.system('mkdir -p '+perfdir)

    CNNmodel = keras.models.load_model(indir+'/TauCNNIdentifier5x9Training_lTauPtCut18/CNNmodel', compile=False)
    DNNident = keras.models.load_model(indir+'/TauCNNIdentifier5x9Training_lTauPtCut18/DNNmodel', compile=False)
    DNNcalib = keras.models.load_model(indir+'/TauCNNCalibrator5x9Training_lTauPtCut18_uEtacut1.5/TauCNNCalibrator', compile=False)

    PUmodel = load_obj(indir+'/TauBDTIdentifierTraining_lTauPtCut18_lEtacut1.5/TauBDTIdentifier/PUmodel.pkl')
    C1model = load_obj(indir+'/TauBDTCalibratorTraining_lTauPtCut18_lEtacut1.5/TauBDTCalibrator/C1model.pkl')
    C2model = load_obj(indir+'/TauBDTCalibratorTraining_lTauPtCut18_lEtacut1.5/TauBDTCalibrator/C2model.pkl')
    C3model = load_obj(indir+'/TauBDTCalibratorTraining_lTauPtCut18_lEtacut1.5/TauBDTCalibrator/C3model.pkl')

    C1coefs = []
    with open(indir+'/TauBDTCalibratorTraining_lTauPtCut18_lEtacut1.5/TauBDTCalibrator/C1model.txt') as file:
        for line in file.readlines():
            C1coefs.append( float(line.split("= ")[1]) )

    C3coefs = []
    with open(indir+'/TauBDTCalibratorTraining_lTauPtCut18_lEtacut1.5/TauBDTCalibrator/C3model.txt') as file:
        for line in file.readlines():
            C3coefs.append( float(line.split("= ")[1]) )

    X1CNN = np.load(indir+'/CMSSWNtuplized/Tensorized_'+options.caloClNxM+'/X_CNN_'+options.caloClNxM+'.npz')['arr_0']
    X2CNN = np.load(indir+'/CMSSWNtuplized/Tensorized_'+options.caloClNxM+'/X_Dense_'+options.caloClNxM+'.npz')['arr_0']
    YCNN  = np.load(indir+'/CMSSWNtuplized/Tensorized_'+options.caloClNxM+'/Y_'+options.caloClNxM+'.npz')['arr_0']
    XBDT = pd.read_pickle(indir+'/CMSSWNtuplized/Tensorized_'+options.caloClNxM+'/X_BDT.pkl')

    # print(X1CNN[0])
    # exit()

    ############################## Apply CNN models to inputs ##############################

    dfCNN = pd.DataFrame()
    dfCNN['CMSSW_IDscore'] = YCNN[:,0].ravel()
    dfCNN['CMSSW_calibPt'] = YCNN[:,1].ravel()
    dfCNN['Keras_IDscore'] = DNNident.predict(CNNmodel.predict([X1CNN, X2CNN])).ravel()
    dfCNN['Keras_calibPt'] = DNNcalib.predict(CNNmodel.predict([X1CNN, X2CNN])).ravel()

    ############################## Apply BDT models to inputs ##############################

    featuresCalib = ['cl3d_showerlength', 'cl3d_coreshowerlength', 'cl3d_abseta', 'cl3d_spptot', 'cl3d_srrmean', 'cl3d_meanz']
    featuresCalibN = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5']
    
    featuresIdent = ['cl3d_pt', 'cl3d_coreshowerlength', 'cl3d_srrtot', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz']
    featuresIdentN = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5']

    # rename features with numbers (needed by PUmodel) and apply it
    for i in range(len(featuresIdent)): XBDT[featuresIdentN[i]] = XBDT[featuresIdent[i]].copy(deep=True)
    XBDT['XGB_IDscore'] = PUmodel.predict_proba(XBDT[featuresIdentN])[:,1]

    # rename features with numbers (needed by C2model) and apply calibrations
    XBDT.drop(featuresIdentN, axis=1, inplace=True)
    for i in range(len(featuresCalib)): XBDT[featuresCalibN[i]] = XBDT[featuresCalib[i]].copy(deep=True)
    XBDT['XGB_pt_c1'] = C1model.predict(XBDT[['cl3d_abseta']]) + XBDT['cl3d_pt']
    XBDT['XGB_pt_c2'] = C2model.predict(XBDT[featuresCalibN]) * XBDT['XGB_pt_c1']
    logpt1 = np.log(abs(XBDT['XGB_pt_c2']))
    logpt2 = logpt1**2
    logpt3 = logpt1**3
    logpt4 = logpt1**4
    XBDT['XGB_pt_c2_log'] = logpt1
    XBDT['XGB_calibPt'] = XBDT['XGB_pt_c2'] / C3model.predict(np.vstack([logpt1, logpt2, logpt3, logpt4]).T)

    XBDT['MAN_pt_c1'] = XBDT['cl3d_pt'] + C1coefs[0]*XBDT['cl3d_abseta'] + C1coefs[1]
    XBDT['MAN_pt_c2'] = C2model.predict(XBDT[featuresCalibN]) * XBDT['MAN_pt_c1']
    XBDT['MAN_calibPt'] = XBDT['MAN_pt_c2'] / (C3coefs[0] + C3coefs[1]*logpt1 + C3coefs[2]*(logpt1**2) + C3coefs[3]*(logpt1**3) + C3coefs[4]*(logpt1**4))

    XBDT['CMSSW_IDscore'] = XBDT['cl3d_IDscore']
    XBDT['CMSSW_calibPt'] = XBDT['cl3d_calibPt']



    plt.figure(figsize=(10,10))
    plt.hist(dfCNN['Keras_IDscore'], bins=np.arange(0,1,0.05), label='Keras', color='red', density=True, histtype='step', lw=2, zorder=-1)
    n,bins,patches = plt.hist(dfCNN['CMSSW_IDscore'], bins=np.arange(0,1,0.05), density=True, histtype='step', lw=0, zorder=-1)
    plt.scatter(bins[:-1]+ 0.5*(bins[1:] - bins[:-1]), n, marker='o', c='black', s=40, label='CMSSW', zorder=1)
    plt.grid(linestyle=':')
    plt.legend(loc = 'upper right', fontsize=16)
    # plt.xlim(0.85,1.001)
    plt.yscale('log')
    #plt.ylim(0.01,1)
    plt.xlabel(r'CNN score')
    plt.ylabel(r'a.u.')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(perfdir+'/DNNident_score.pdf')
    plt.close()

    plt.figure(figsize=(10,10))
    plt.hist(dfCNN['Keras_calibPt'], bins=np.arange(0,150,5), label='Keras', color='red', density=True, histtype='step', lw=2, zorder=-1)
    n,bins,patches = plt.hist(dfCNN['CMSSW_calibPt'], bins=np.arange(0,150,5), density=True, histtype='step', lw=0, zorder=-1)
    plt.scatter(bins[:-1]+ 0.5*(bins[1:] - bins[:-1]), n, marker='o', c='black', s=40, label='CMSSW', zorder=1)
    plt.grid(linestyle=':')
    plt.legend(loc = 'upper right', fontsize=16)
    # plt.xlim(0.85,1.001)
    plt.yscale('log')
    #plt.ylim(0.01,1)
    plt.xlabel(r'CNN score')
    plt.ylabel(r'a.u.')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(perfdir+'/DNNcalib_pt.pdf')
    plt.close()



    plt.figure(figsize=(10,10))
    plt.hist(XBDT['XGB_IDscore'], bins=np.arange(0,1,0.05), label='XGBoost', color='red', density=True, histtype='step', lw=2, zorder=-1)
    n,bins,patches = plt.hist(XBDT['CMSSW_IDscore'], bins=np.arange(0,1,0.05), density=True, histtype='step', lw=0, zorder=-1)
    plt.scatter(bins[:-1]+ 0.5*(bins[1:] - bins[:-1]), n, marker='o', c='black', s=40, label='CMSSW', zorder=1)
    plt.grid(linestyle=':')
    plt.legend(loc = 'upper right', fontsize=16)
    # plt.xlim(0.85,1.001)
    plt.yscale('log')
    #plt.ylim(0.01,1)
    plt.xlabel(r'CNN score')
    plt.ylabel(r'a.u.')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(perfdir+'/XGBident_score.pdf')
    plt.close()

    plt.figure(figsize=(10,10))
    plt.hist(XBDT['XGB_calibPt'], bins=np.arange(0,100,5), label='XGBoost', color='red', density=True, histtype='step', lw=2, zorder=-1)
    n,bins,patches = plt.hist(XBDT['CMSSW_calibPt'], bins=np.arange(0,100,5), density=True, histtype='step', lw=0, zorder=-1)
    plt.scatter(bins[:-1]+ 0.5*(bins[1:] - bins[:-1]), n, marker='o', c='black', s=40, label='CMSSW', zorder=1)
    plt.grid(linestyle=':')
    plt.legend(loc = 'upper right', fontsize=16)
    # plt.xlim(0.85,1.001)
    plt.yscale('log')
    #plt.ylim(0.01,1)
    plt.xlabel(r'CNN score')
    plt.ylabel(r'a.u.')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(perfdir+'/XGBcalib_pt.pdf')
    plt.close()































