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
    parser.add_option("--v",             dest="v",                                 default=None)
    parser.add_option("--date",          dest="date",                              default=None)
    parser.add_option("--inTagCNNCalib", dest="inTagCNNCalib",                     default="")
    parser.add_option("--inTagCNNIdent", dest="inTagCNNIdent",                     default="")
    parser.add_option("--inTagBDTCalib", dest="inTagBDTCalib",                     default="")
    parser.add_option("--inTagBDTIdent", dest="inTagBDTIdent",                     default="")
    parser.add_option('--caloClNxM',     dest='caloClNxM',                         default="9x9")
    (options, args) = parser.parse_args()
    print(options)

    indir = '/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v

    # get clusters' shape dimensions
    N = int(options.caloClNxM.split('x')[0])
    M = int(options.caloClNxM.split('x')[1])

    TauCalibratorModel = keras.models.load_model(indir+'/TauCNNCalibrator'+options.caloClNxM+'Training'+options.inTagCNNCalib+'/TauCNNCalibrator', compile=False)
    TauIdentifierModel = keras.models.load_model(indir+'/TauCNNIdentifier'+options.caloClNxM+'Training'+options.inTagCNNIdent+'/TauCNNIdentifier', compile=False)
    PUmodel = load_obj(indir+'/TauBDTIdentifierTraining'+options.inTagBDTIdent+'/TauBDTIdentifier/PUmodel.pkl')
    C1model = load_obj(indir+'/TauBDTCalibratorTraining'+options.inTagBDTCalib+'/TauBDTCalibrator/C1model.pkl')
    C2model = load_obj(indir+'/TauBDTCalibratorTraining'+options.inTagBDTCalib+'/TauBDTCalibrator/C2model.pkl')
    C3model = load_obj(indir+'/TauBDTCalibratorTraining'+options.inTagBDTCalib+'/TauBDTCalibrator/C3model.pkl')

    CNN_WP_dict = load_obj(indir+'/TauCNNEvaluator'+options.caloClNxM+options.inTagCNNCalib+'/TauMinatorCNN_WPs.pkl')
    BDT_WP_dict = load_obj(indir+'/TauBDTEvaluator'+options.inTagBDTCalib+'/TauMinatorBDT_WPs.pkl')

    perfdir = indir+'/TauMinatorPerformanceEvaluator_CNNIdent'+options.inTagCNNIdent+'_BDTIdent'+options.inTagBDTIdent+'_CNNCalib'+options.inTagCNNCalib+'_BDTCalib'+options.inTagBDTCalib
    os.system('mkdir -p '+perfdir)

    











