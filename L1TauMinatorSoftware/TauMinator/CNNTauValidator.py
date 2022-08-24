from optparse import OptionParser
from tensorflow import keras
from sklearn import metrics
import tensorflow as tf
import numpy as np
import os

import matplotlib.pyplot as plt
# import mplhep
# plt.style.use(mplhep.style.CMS)

np.random.seed(7)

#######################################################################
######################### SCRIPT BODY #################################
#######################################################################

if __name__ == "__main__" :
    parser = OptionParser()
    parser.add_option("--v",            dest="v",                                 default=None)
    parser.add_option("--date",         dest="date",                              default=None)
    parser.add_option("--inTag",        dest="inTag",                             default="")
    parser.add_option('--caloClNxM',    dest='caloClNxM',                         default="9x9")
    (options, args) = parser.parse_args()
    print(options)

    # get clusters' shape dimensions
    N = int(options.caloClNxM.split('x')[0])
    M = int(options.caloClNxM.split('x')[1])

    indir = '/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v

    TauCalibratorModel = keras.models.load_model(indir+'/TauCNNCalibrator'+options.caloClNxM+'Training'+options.inTag+'/TauCNNCalibrator', compile=False)
    TauIdentifierModel = keras.models.load_model(indir+'/TauCNNIdentifier'+options.caloClNxM+'Training'+options.inTag+'/TauCNNIdentifier', compile=False)

    X1 = np.load(indir+'/TauCNNValidator'+options.caloClNxM+'/X_Calib_CNN_'+options.caloClNxM+'_forValidator.npz')['arr_0']
    X2 = np.load(indir+'/TauCNNValidator'+options.caloClNxM+'/X_Calib_Dense_'+options.caloClNxM+'_forValidator.npz')['arr_0']
    Y = np.load(indir+'/TauCNNValidator'+options.caloClNxM+'/Y_Calib_'+options.caloClNxM+'_forValidator.npz')['arr_0']

    Xcalibrated = TauCalibratorModel.predict([X1, X2])
    Xuncalibrated = np.sum(np.sum(np.sum(X1, axis=3), axis=2), axis=1)
    plt.figure(figsize=(10,10))
    plt.hist(Xuncalibrated.ravel()/Y.ravel(), bins=np.arange(0,5,0.1), label='Uncalibrated response, mean: {0}, rms : {1}'.format(np.mean(Xuncalibrated.ravel()/Y.ravel()), np.std(Xuncalibrated.ravel()/Y.ravel())), color='red',lw=2, density=True)
    plt.hist(Xcalibrated.ravel()/Y.ravel(),   bins=np.arange(0,5,0.1), label='Calibrated response, mean: {0}, rms : {1}'.format(np.mean(Xcalibrated.ravel()/Y.ravel()), np.std(Xcalibrated.ravel()/Y.ravel())), color='blue',lw=2, density=True)
    plt.grid(linestyle=':')
    plt.legend(loc = 'upper left')
    # plt.xlim(0.85,1.001) 
    #plt.yscale('log')
    #plt.ylim(0.01,1)
    plt.xlabel(r'$p_{T}}^{L1\tau}-p_{T}^{Gen \tau}$')
    plt.ylabel(r'a.u.')
    plt.savefig(indir+'/TauCNNValidator'+options.caloClNxM+'/validation_calibration.pdf')
    plt.close()



    X1 = np.load(indir+'/TauCNNValidator'+options.caloClNxM+'/X_Ident_CNN_'+options.caloClNxM+'_forValidator.npz')['arr_0']
    X2 = np.load(indir+'/TauCNNValidator'+options.caloClNxM+'/X_Ident_Dense_'+options.caloClNxM+'_forValidator.npz')['arr_0']
    Y = np.load(indir+'/TauCNNValidator'+options.caloClNxM+'/Y_Ident_'+options.caloClNxM+'_forValidator.npz')['arr_0']

    Xidentified = TauIdentifierModel.predict([X1, X2])
    FPR, TPR, THR = metrics.roc_curve(Y, Xidentified)
    plt.figure(figsize=(10,10))
    plt.plot(TPR, FPR, label='Validation ROC', color='blue',lw=2)
    plt.grid(linestyle=':')
    plt.legend(loc = 'upper left')
    # plt.xlim(0.85,1.001)
    #plt.yscale('log')
    #plt.ylim(0.01,1)
    plt.xlabel('Signal efficiency')
    plt.ylabel('Background efficiency')
    plt.savefig(indir+'/TauCNNValidator'+options.caloClNxM+'/validation_roc.pdf')
    plt.close()


