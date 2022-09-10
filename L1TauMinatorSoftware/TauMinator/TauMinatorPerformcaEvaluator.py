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


def save_obj(obj,dest):
    with open(dest,'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(source):
    with open(source,'rb') as f:
        return pickle.load(f)

def efficiency(g, WP, thr, upper=False):
    sel = g[(g['pass'+WP]==1) & (g['L1_pt_c3']>thr)].shape[0]
    # sel = g[g['L1_pt']>thr].shape[0]
    tot = g.shape[0]

    efficiency = float(sel) / float(tot)

    # clopper pearson errors --> ppf gives the boundary of the cinfidence interval, therefore for plotting we have to subtract the value of the central value of the efficiency!!
    alpha = (1 - 0.9) / 2

    if sel == tot:
        uError = 0.
    else:
        uError = abs(btdtri(sel+1, tot-sel, 1-alpha) - efficiency)

    if sel == 0:
        lError = 0.
    else:
        lError = abs(efficiency - btdtri(sel, tot-sel+1, alpha))

    return efficiency, lError, uError

def sigmoid(x , a, x0, k):
    return a / ( 1 + np.exp(-k*(x-x0)) )


#######################################################################
######################### SCRIPT BODY #################################
#######################################################################

if __name__ == "__main__" :
    parser = OptionParser()
    parser.add_option("--v",                dest="v",                default=None)
    parser.add_option("--date",             dest="date",             default=None)
    parser.add_option("--inTag",            dest="inTag",            default="")
    parser.add_option("--inTagCNNCalib",    dest="inTagCNNCalib",    default="")
    parser.add_option("--CNNCalibSparsity", dest="CNNCalibSparsity", default=None)
    parser.add_option("--inTagCNNIdent",    dest="inTagCNNIdent",    default="")
    parser.add_option("--CNNIdentSparsity", dest="CNNIdentSparsity", default=None)
    parser.add_option("--inTagBDTCalib",    dest="inTagBDTCalib",    default="")
    parser.add_option("--inTagBDTIdent",    dest="inTagBDTIdent",    default="")
    parser.add_option('--caloClNxM',        dest='caloClNxM',        default="5x9")
    (options, args) = parser.parse_args()
    print(options)

    # get clusters' shape dimensions
    N = int(options.caloClNxM.split('x')[0])
    M = int(options.caloClNxM.split('x')[1])

    ############################## Get models and inputs ##############################

    indir = '/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v
    perfdir = indir+'/TauMinatorPerformanceEvaluator_'+options.caloClNxM
    CNNWPdir = indir+'/TauCNNEvaluator'+options.caloClNxM
    BDTWPdir = indir+'/TauBDTEvaluator'

    if options.CNNCalibSparsity:
        TauCalibratorModel = keras.models.load_model(indir+'/TauCNNCalibrator'+options.caloClNxM+'Training'+options.inTagCNNCalib+'/TauCNNCalibrator'+options.CNNCalibSparsity+'Pruned', compile=False)
        perfdir += '_CNNCalib'+options.inTagCNNCalib+'_'+options.CNNCalibSparsity+'Pruned'
        CNNWPdir += '_Calib'+options.inTagCNNCalib+'_'+options.CNNCalibSparsity+'Pruned'
    else:
        TauCalibratorModel = keras.models.load_model(indir+'/TauCNNCalibrator'+options.caloClNxM+'Training'+options.inTagCNNCalib+'/TauCNNCalibrator', compile=False)
        perfdir += '_CNNCalib'+options.inTagCNNCalib
        CNNWPdir += '_Calib'+options.inTagCNNCalib
    
    if options.CNNIdentSparsity:
        TauIdentifierModel = keras.models.load_model(indir+'/TauCNNIdentifier'+options.caloClNxM+'Training'+options.inTagCNNIdent+'/TauCNNIdentifier'+options.CNNIdentSparsity+'Pruned', compile=False)
        perfdir += '_CNNIdent'+options.inTagCNNIdent+'_'+options.CNNIdentSparsity+'Pruned'
        CNNWPdir += '_Ident'+options.inTagCNNIdent+'_'+options.CNNIdentSparsity+'Pruned'
    else:
        TauIdentifierModel = keras.models.load_model(indir+'/TauCNNIdentifier'+options.caloClNxM+'Training'+options.inTagCNNIdent+'/TauCNNIdentifier', compile=False)
        perfdir += '_CNNIdent'+options.inTagCNNIdent
        CNNWPdir += '_Ident'+options.inTagCNNIdent

    PUmodel = load_obj(indir+'/TauBDTIdentifierTraining'+options.inTagBDTIdent+'/TauBDTIdentifier/PUmodel.pkl')
    C1model = load_obj(indir+'/TauBDTCalibratorTraining'+options.inTagBDTCalib+'/TauBDTCalibrator/C1model.pkl')
    C2model = load_obj(indir+'/TauBDTCalibratorTraining'+options.inTagBDTCalib+'/TauBDTCalibrator/C2model.pkl')
    C3model = load_obj(indir+'/TauBDTCalibratorTraining'+options.inTagBDTCalib+'/TauBDTCalibrator/C3model.pkl')
    perfdir += '_BDTCalib'+options.inTagBDTCalib+'_BDTIdent'+options.inTagBDTIdent
    BDTWPdir += '_Calib'+options.inTagBDTCalib+'_Ident'+options.inTagBDTCalib

    CNN_WP_dict = load_obj(CNNWPdir+'/TauMinatorCNN_WPs.pkl')
    BDT_WP_dict = load_obj(BDTWPdir+'/TauMinatorBDT_WPs.pkl')

    os.system('mkdir -p '+perfdir)

    X1CNN = np.load(indir+'/TauMinatorInputs_'+options.caloClNxM+options.inTag+'/X_CNN_'+options.caloClNxM+'.npz')['arr_0']
    X2CNN = np.load(indir+'/TauMinatorInputs_'+options.caloClNxM+options.inTag+'/X_Dense_'+options.caloClNxM+'.npz')['arr_0']
    YCNN  = np.load(indir+'/TauMinatorInputs_'+options.caloClNxM+options.inTag+'/Y_'+options.caloClNxM+'.npz')['arr_0']

    XBDT = pd.read_pickle(indir+'/TauMinatorInputs_'+options.caloClNxM+options.inTag+'/X_BDT.pkl')
    XBDT['cl3d_abseta'] = abs(XBDT['cl3d_eta']).copy(deep=True)

    ############################## Apply CNN models to inputs ##############################

    dfCNN = pd.DataFrame()
    dfCNN['event']       = YCNN[:,4].ravel()
    dfCNN['tau_visPt' ]  = YCNN[:,0].ravel()
    dfCNN['tau_visEta']  = YCNN[:,1].ravel()
    dfCNN['tau_visPhi']  = YCNN[:,2].ravel()
    dfCNN['tau_DM']      = YCNN[:,3].ravel()
    dfCNN['L1_pt_CLNxM'] = TauCalibratorModel.predict([X1CNN, X2CNN]).ravel()
    dfCNN['CNNscore']    = TauIdentifierModel.predict([X1CNN, X2CNN]).ravel()
    dfCNN['CNNpass99']   = dfCNN['CNNscore'] > CNN_WP_dict['wp99']
    dfCNN['CNNpass95']   = dfCNN['CNNscore'] > CNN_WP_dict['wp95']
    dfCNN['CNNpass90']   = dfCNN['CNNscore'] > CNN_WP_dict['wp90']

    ############################## Apply BDT models to inputs ##############################

    featuresCalib = ['cl3d_showerlength', 'cl3d_coreshowerlength', 'cl3d_abseta', 'cl3d_spptot', 'cl3d_srrmean', 'cl3d_meanz']
    featuresCalibN = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5']
    
    featuresIdent = ['cl3d_pt', 'cl3d_coreshowerlength', 'cl3d_srrtot', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz']
    featuresIdentN = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5']

    # rename features with numbers (needed by PUmodel) and apply it
    for i in range(len(featuresIdent)): XBDT[featuresIdentN[i]] = XBDT[featuresIdent[i]].copy(deep=True)
    XBDT['bdt_output'] = PUmodel.predict_proba(XBDT[featuresIdentN])[:,1]

    # rename features with numbers (needed by C2model) and apply calibrations
    XBDT.drop(featuresIdentN, axis=1, inplace=True)
    for i in range(len(featuresCalib)): XBDT[featuresCalibN[i]] = XBDT[featuresCalib[i]].copy(deep=True)
    XBDT['cl3d_pt_c1'] = C1model.predict(XBDT[['cl3d_abseta']]) + XBDT['cl3d_pt']
    XBDT['cl3d_pt_c2'] = C2model.predict(XBDT[featuresCalibN]) * XBDT['cl3d_pt_c1']
    logpt1 = np.log(abs(XBDT['cl3d_pt_c2']))
    logpt2 = logpt1**2
    logpt3 = logpt1**3
    logpt4 = logpt1**4
    XBDT['cl3d_pt_c3'] = XBDT['cl3d_pt_c2'] / C3model.predict(np.vstack([logpt1, logpt2, logpt3, logpt4]).T)

    dfBDT = XBDT[['event', 'tau_visPt', 'tau_visEta', 'tau_visPhi', 'tau_DM', 'cl3d_pt_c1', 'cl3d_pt_c2', 'cl3d_pt_c3', 'bdt_output']].copy(deep=True)
    dfBDT.rename(columns={'cl3d_pt_c3':'L1_pt_CL3D', 'bdt_output':'BDTscore'}, inplace=True)
    dfBDT['BDTpass99']  = dfBDT['BDTscore'] > BDT_WP_dict['wp99']
    dfBDT['BDTpass95']  = dfBDT['BDTscore'] > BDT_WP_dict['wp95']
    dfBDT['BDTpass90']  = dfBDT['BDTscore'] > BDT_WP_dict['wp90']

    ############################## Match CL3D to CLNxM in the endcap ##############################

    dfCNN_CE = dfCNN[abs(dfCNN['tau_visEta'])>1.4]

    dfCNN_CE.sort_values('event', inplace=True)
    dfBDT.sort_values('event', inplace=True)
    # dfCNN_CE = dfCNN_CE.head(10).copy(deep=True)
    # dfBDT = dfBDT.head(10).copy(deep=True)

    dfCNN_CE.set_index(['event', 'tau_visPt', 'tau_visEta', 'tau_visPhi', 'tau_DM'], inplace=True)
    dfBDT.set_index(['event', 'tau_visPt', 'tau_visEta', 'tau_visPhi', 'tau_DM'], inplace=True)

    print(dfCNN_CE)
    print("\n--------------------------")
    print("--------------------------\n")
    print(dfBDT)

    dfTauMinated = dfCNN_CE.join(dfBDT, on=['event', 'tau_visPt', 'tau_visEta', 'tau_visPhi', 'tau_DM'], how='left', rsuffix='_joined', sort=False)

    dfTauMinated.dropna(axis=0, how='any', inplace=True)

    print("\n--------------------------")
    print("--------------------------\n")
    print(dfTauMinated)















