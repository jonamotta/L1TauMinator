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

    CLTW_CNNmodel = keras.models.load_model(indir+'/TauCNNIdentifier5x9Training_lTauPtCut18/CNNmodel', compile=False)
    CLTW_DNNident = keras.models.load_model(indir+'/TauCNNIdentifier5x9Training_lTauPtCut18/DNNmodel', compile=False)
    CLTW_DNNcalib = keras.models.load_model(indir+'/TauCNNCalibrator5x9Training_lTauPtCut18_uEtacut1.5/TauCNNCalibrator', compile=False)

    # PUmodel = load_obj(indir+'/TauBDTIdentifierTraining_lTauPtCut18_lEtacut1.5/TauBDTIdentifier/PUmodel.pkl')
    # C1model = load_obj(indir+'/TauBDTCalibratorTraining_lTauPtCut18_lEtacut1.5/TauBDTCalibrator/C1model.pkl')
    # C2model = load_obj(indir+'/TauBDTCalibratorTraining_lTauPtCut18_lEtacut1.5/TauBDTCalibrator/C2model.pkl')
    # C3model = load_obj(indir+'/TauBDTCalibratorTraining_lTauPtCut18_lEtacut1.5/TauBDTCalibrator/C3model.pkl')

    # C1coefs = []
    # with open(indir+'/TauBDTCalibratorTraining_lTauPtCut18_lEtacut1.5/TauBDTCalibrator/C1model.txt') as file:
    #     for line in file.readlines():
    #         C1coefs.append( float(line.split("= ")[1]) )

    # C3coefs = []
    # with open(indir+'/TauBDTCalibratorTraining_lTauPtCut18_lEtacut1.5/TauBDTCalibrator/C3model.txt') as file:
    #     for line in file.readlines():
    #         C3coefs.append( float(line.split("= ")[1]) )

    CL3D_DNNident = keras.models.load_model(indir+'/TauDNNIdentifierTraining_lTauPtCut18_lEtacut1.5/TauDNNIdentifier', compile=False)
    CL3D_DNNcalib = keras.models.load_model(indir+'/TauDNNCalibratorTraining_lTauPtCut18_lEtacut1.5/TauDNNCalibrator', compile=False)

    X1CNN = np.load(indir+'/CMSSWNtuplized/Tensorized_'+options.caloClNxM+'/X_CNN_'+options.caloClNxM+'.npz')['arr_0']
    X2CNN = np.load(indir+'/CMSSWNtuplized/Tensorized_'+options.caloClNxM+'/X_Dense_'+options.caloClNxM+'.npz')['arr_0']
    YCNN  = np.load(indir+'/CMSSWNtuplized/Tensorized_'+options.caloClNxM+'/Y_'+options.caloClNxM+'.npz')['arr_0']
    XBDT = pd.read_pickle(indir+'/CMSSWNtuplized/Tensorized_'+options.caloClNxM+'/X_BDT.pkl')

    pt = ['cl3d_pt']
    feats = ['cl3d_localAbsEta', 'cl3d_showerlength', 'cl3d_coreshowerlength', 'cl3d_firstlayer', 'cl3d_seetot', 'cl3d_szz', 'cl3d_srrtot', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_localAbsMeanZ']
    scaler = load_obj(indir+'/TauDNNOptimization/dnn_features_scaler.pkl')
    scaled = pd.DataFrame(scaler.transform(XBDT[pt+feats]), columns=pt+feats)


    manually_scaled = pd.DataFrame(columns=pt+feats)
    manually_scaled['cl3d_pt']               = (XBDT['cl3d_pt'] - 20.241386991916283)/23.01837359563478
    manually_scaled['cl3d_localAbsEta']      = (XBDT['cl3d_localAbsEta'] - 1.0074996364465862)/0.49138567390141064
    manually_scaled['cl3d_showerlength']     = (XBDT['cl3d_showerlength'] - 35.80472904577922)/7.440518440086
    manually_scaled['cl3d_coreshowerlength'] = (XBDT['cl3d_coreshowerlength'] - 11.933355669294706)/4.791972733791
    manually_scaled['cl3d_firstlayer']       = (XBDT['cl3d_firstlayer'] - 1.3720078128341484)/1.6893861207515
    manually_scaled['cl3d_seetot']           = (XBDT['cl3d_seetot'] - 0.03648582652086512)/0.020407089049271552
    manually_scaled['cl3d_szz']              = (XBDT['cl3d_szz'] - 20.51621627863874)/11.633317917896875
    manually_scaled['cl3d_srrtot']           = (XBDT['cl3d_srrtot'] - 0.00534390307737272)/0.001325129860675611
    manually_scaled['cl3d_srrmean']          = (XBDT['cl3d_srrmean'] - 0.00365570411813347367)/0.0009327963551387752
    manually_scaled['cl3d_hoe']              = (XBDT['cl3d_hoe'] - 1.3676566630073708)/7.978238945457623
    manually_scaled['cl3d_localAbsMeanZ']    = (XBDT['cl3d_localAbsMeanZ'] - 291.6762877632198)/178.8235004591792

    TensorizedInputIdent  = scaled[feats].to_numpy()
    TensorizedInputCalib  = scaled[pt+feats].to_numpy()

    TensorizedInputIdentMan  = manually_scaled[feats].to_numpy()
    TensorizedInputCalibMan  = manually_scaled[pt+feats].to_numpy()

    ############################## Apply CNN models to inputs ##############################

    dfCNN = pd.DataFrame()
    dfCNN['CMSSW_IDscore'] = YCNN[:,0].ravel()
    dfCNN['CMSSW_calibPt'] = YCNN[:,1].ravel()
    dfCNN['Keras_IDscore'] = CLTW_DNNident.predict(CLTW_CNNmodel.predict([X1CNN, X2CNN])).ravel()
    dfCNN['Keras_calibPt'] = CLTW_DNNcalib.predict(CLTW_CNNmodel.predict([X1CNN, X2CNN])).ravel()


    ############################## Apply BDT models to inputs ##############################

    # featuresCalib = ['cl3d_showerlength', 'cl3d_coreshowerlength', 'cl3d_abseta', 'cl3d_spptot', 'cl3d_srrmean', 'cl3d_meanz']
    # featuresCalibN = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5']
    
    # featuresIdent = ['cl3d_pt', 'cl3d_coreshowerlength', 'cl3d_srrtot', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz']
    # featuresIdentN = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5']

    # # rename features with numbers (needed by PUmodel) and apply it
    # for i in range(len(featuresIdent)): XBDT[featuresIdentN[i]] = XBDT[featuresIdent[i]].copy(deep=True)
    # XBDT['XGB_IDscore'] = PUmodel.predict_proba(XBDT[featuresIdentN])[:,1]

    # # rename features with numbers (needed by C2model) and apply calibrations
    # XBDT.drop(featuresIdentN, axis=1, inplace=True)
    # for i in range(len(featuresCalib)): XBDT[featuresCalibN[i]] = XBDT[featuresCalib[i]].copy(deep=True)
    # XBDT['XGB_pt_c1'] = C1model.predict(XBDT[['cl3d_abseta']]) + XBDT['cl3d_pt']
    # XBDT['XGB_pt_c2'] = C2model.predict(XBDT[featuresCalibN]) * XBDT['XGB_pt_c1']
    # logpt1 = np.log(abs(XBDT['XGB_pt_c2']))
    # logpt2 = logpt1**2
    # logpt3 = logpt1**3
    # logpt4 = logpt1**4
    # XBDT['XGB_pt_c2_log'] = logpt1
    # XBDT['XGB_calibPt'] = XBDT['XGB_pt_c2'] / C3model.predict(np.vstack([logpt1, logpt2, logpt3, logpt4]).T)

    # XBDT['MAN_pt_c1'] = XBDT['cl3d_pt'] + C1coefs[0]*XBDT['cl3d_abseta'] + C1coefs[1]
    # XBDT['MAN_pt_c2'] = C2model.predict(XBDT[featuresCalibN]) * XBDT['MAN_pt_c1']
    # XBDT['MAN_calibPt'] = XBDT['MAN_pt_c2'] / (C3coefs[0] + C3coefs[1]*logpt1 + C3coefs[2]*(logpt1**2) + C3coefs[3]*(logpt1**3) + C3coefs[4]*(logpt1**4))

    # XBDT['CMSSW_IDscore'] = XBDT['cl3d_IDscore']
    # XBDT['CMSSW_calibPt'] = XBDT['cl3d_calibPt']

    ############################## Apply DNN models to inputs ##############################

    dfDNN = pd.DataFrame()
    dfDNN['CMSSW_IDscore'] = XBDT['cl3d_IDscore']
    dfDNN['CMSSW_calibPt'] = XBDT['cl3d_calibPt']
    dfDNN['Keras_IDscore'] = CL3D_DNNident.predict(TensorizedInputIdent).ravel()
    dfDNN['Keras_calibPt'] = CL3D_DNNcalib.predict(TensorizedInputCalib).ravel()
    dfDNN['Manual_Keras_IDscore'] = CL3D_DNNident.predict(TensorizedInputIdentMan).ravel()
    dfDNN['Manual_Keras_calibPt'] = CL3D_DNNcalib.predict(TensorizedInputCalibMan).ravel()


    ############################## Make validation plots ##############################

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
    plt.savefig(perfdir+'/CLTW_DNNident_score.pdf')
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
    plt.savefig(perfdir+'/CLTW_DNNcalib_pt.pdf')
    plt.close()



    # plt.figure(figsize=(10,10))
    # plt.hist(XBDT['XGB_IDscore'], bins=np.arange(0,1,0.05), label='XGBoost', color='red', density=True, histtype='step', lw=2, zorder=-1)
    # n,bins,patches = plt.hist(XBDT['CMSSW_IDscore'], bins=np.arange(0,1,0.05), density=True, histtype='step', lw=0, zorder=-1)
    # plt.scatter(bins[:-1]+ 0.5*(bins[1:] - bins[:-1]), n, marker='o', c='black', s=40, label='CMSSW', zorder=1)
    # plt.grid(linestyle=':')
    # plt.legend(loc = 'upper right', fontsize=16)
    # # plt.xlim(0.85,1.001)
    # plt.yscale('log')
    # #plt.ylim(0.01,1)
    # plt.xlabel(r'CNN score')
    # plt.ylabel(r'a.u.')
    # mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    # plt.savefig(perfdir+'/XGBident_score.pdf')
    # plt.close()

    # plt.figure(figsize=(10,10))
    # plt.hist(XBDT['XGB_calibPt'], bins=np.arange(0,100,5), label='XGBoost', color='red', density=True, histtype='step', lw=2, zorder=-1)
    # n,bins,patches = plt.hist(XBDT['CMSSW_calibPt'], bins=np.arange(0,100,5), density=True, histtype='step', lw=0, zorder=-1)
    # plt.scatter(bins[:-1]+ 0.5*(bins[1:] - bins[:-1]), n, marker='o', c='black', s=40, label='CMSSW', zorder=1)
    # plt.grid(linestyle=':')
    # plt.legend(loc = 'upper right', fontsize=16)
    # # plt.xlim(0.85,1.001)
    # plt.yscale('log')
    # #plt.ylim(0.01,1)
    # plt.xlabel(r'CNN score')
    # plt.ylabel(r'a.u.')
    # mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    # plt.savefig(perfdir+'/XGBcalib_pt.pdf')
    # plt.close()



    plt.figure(figsize=(10,10))
    plt.hist(dfDNN['Keras_IDscore'], bins=np.arange(0,1,0.05), label='Keras', color='red', density=True, histtype='step', lw=2, zorder=-1)
    n,bins,patches = plt.hist(dfDNN['CMSSW_IDscore'], bins=np.arange(0,1,0.05), density=True, histtype='step', lw=0, zorder=-1)
    plt.scatter(bins[:-1]+ 0.5*(bins[1:] - bins[:-1]), n, marker='o', c='black', s=40, label='CMSSW', zorder=1)
    plt.grid(linestyle=':')
    plt.legend(loc = 'upper right', fontsize=16)
    # plt.xlim(0.85,1.001)
    plt.yscale('log')
    #plt.ylim(0.01,1)
    plt.xlabel(r'DNN score')
    plt.ylabel(r'a.u.')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(perfdir+'/CL3D_DNNident_score.pdf')
    plt.close()

    plt.figure(figsize=(10,10))
    plt.hist(dfDNN['Keras_calibPt'], bins=np.arange(0,150,5), label='Keras', color='red', density=True, histtype='step', lw=2, zorder=-1)
    n,bins,patches = plt.hist(dfDNN['CMSSW_calibPt'], bins=np.arange(0,150,5), density=True, histtype='step', lw=0, zorder=-1)
    plt.scatter(bins[:-1]+ 0.5*(bins[1:] - bins[:-1]), n, marker='o', c='black', s=40, label='CMSSW', zorder=1)
    plt.grid(linestyle=':')
    plt.legend(loc = 'upper right', fontsize=16)
    # plt.xlim(0.85,1.001)
    plt.yscale('log')
    #plt.ylim(0.01,1)
    plt.xlabel(r'DNN score')
    plt.ylabel(r'a.u.')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(perfdir+'/CL3D_DNNcalib_pt.pdf')
    plt.close()



    plt.figure(figsize=(10,10))
    plt.hist(dfDNN['Manual_Keras_IDscore'], bins=np.arange(0,1,0.05), label='Keras', color='red', density=True, histtype='step', lw=2, zorder=-1)
    n,bins,patches = plt.hist(dfDNN['CMSSW_IDscore'], bins=np.arange(0,1,0.05), density=True, histtype='step', lw=0, zorder=-1)
    plt.scatter(bins[:-1]+ 0.5*(bins[1:] - bins[:-1]), n, marker='o', c='black', s=40, label='CMSSW', zorder=1)
    plt.grid(linestyle=':')
    plt.legend(loc = 'upper right', fontsize=16)
    # plt.xlim(0.85,1.001)
    plt.yscale('log')
    #plt.ylim(0.01,1)
    plt.xlabel(r'DNN score')
    plt.ylabel(r'a.u.')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(perfdir+'/CL3D_manual_DNNident_score.pdf')
    plt.close()

    plt.figure(figsize=(10,10))
    plt.hist(dfDNN['Manual_Keras_calibPt'], bins=np.arange(0,150,5), label='Keras', color='red', density=True, histtype='step', lw=2, zorder=-1)
    n,bins,patches = plt.hist(dfDNN['CMSSW_calibPt'], bins=np.arange(0,150,5), density=True, histtype='step', lw=0, zorder=-1)
    plt.scatter(bins[:-1]+ 0.5*(bins[1:] - bins[:-1]), n, marker='o', c='black', s=40, label='CMSSW', zorder=1)
    plt.grid(linestyle=':')
    plt.legend(loc = 'upper right', fontsize=16)
    # plt.xlim(0.85,1.001)
    plt.yscale('log')
    #plt.ylim(0.01,1)
    plt.xlabel(r'DNN score')
    plt.ylabel(r'a.u.')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(perfdir+'/CL3D_manual_DNNcalib_pt.pdf')
    plt.close()



