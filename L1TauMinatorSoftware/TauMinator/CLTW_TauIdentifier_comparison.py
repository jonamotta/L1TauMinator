from optparse import OptionParser
from tensorflow import keras
from sklearn import metrics
import tensorflow as tf
import pandas as pd
import numpy as np
import sys
import os

np.random.seed(77)

import matplotlib.pyplot as plt
import mplhep
plt.style.use(mplhep.style.CMS)


#######################################################################
######################### SCRIPT BODY #################################
#######################################################################

if __name__ == "__main__" :

    parser = OptionParser()
    parser.add_option("--v",            dest="v",                              default=None)
    parser.add_option("--date",         dest="date",                           default=None)
    parser.add_option("--inTag",        dest="inTag",                          default="")
    parser.add_option('--caloClNxM',    dest='caloClNxM',                      default="5x9")
    parser.add_option('--sparsity',     dest='sparsity',  type=float,          default=0.5)
    (options, args) = parser.parse_args()
    print(options)

    indir = '/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNNIdentifier'+options.caloClNxM+'Training'+options.inTag

    # get clusters' shape dimensions
    N = int(options.caloClNxM.split('x')[0])
    M = int(options.caloClNxM.split('x')[1])

    sparsityTag = str(options.sparsity).split('.')[0]+'p'+str(options.sparsity).split('.')[1]

    # load non-quantized models
    TauIdentifierModel = keras.models.load_model('/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNNIdentifier'+options.caloClNxM+'Training'+options.inTag+'/TauCNNIdentifier', compile=False)
    TauIdentifierModelPruned = keras.models.load_model('/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNNIdentifier'+options.caloClNxM+'Training'+options.inTag+'/TauCNNIdentifier'+sparsityTag+'Pruned', compile=False)

    # load quantized models
    TauQIdentifierModel = keras.models.load_model('/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNNIdentifier'+options.caloClNxM+'Training'+options.inTag+'/TauQCNNIdentifier', compile=False)
    TauQIdentifierModelPruned = keras.models.load_model('/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNNIdentifier'+options.caloClNxM+'Training'+options.inTag+'/TauQCNNIdentifier'+sparsityTag+'Pruned', compile=False)

    # load datasets
    X1_valid = np.load(indir+'/X_CNN_'+options.caloClNxM+'_forEvaluator.npz')['arr_0']
    X2_valid = np.load(indir+'/X_Dense_'+options.caloClNxM+'_forEvaluator.npz')['arr_0']
    Y_valid  = np.load(indir+'/Y_'+options.caloClNxM+'_forEvaluator.npz')['arr_0']


    # apply baseline model
    valid_ident = TauIdentifierModel.predict([X1_valid, X2_valid])
    FPRvalid, TPRvalid, THRvalid = metrics.roc_curve(Y_valid, valid_ident)
    AUCvalid = metrics.roc_auc_score(Y_valid, valid_ident)

    # apply pruned model
    valid_ident_pruned = TauIdentifierModelPruned.predict([X1_valid, X2_valid])
    FPRvalid_pruned, TPRvalid_pruned, THRvalid_pruned = metrics.roc_curve(Y_valid, valid_ident_pruned)
    AUCvalid_pruned = metrics.roc_auc_score(Y_valid, valid_ident_pruned)

    # apply quantized model
    valid_Qident = TauQIdentifierModel.predict([X1_valid, X2_valid])
    QFPRvalid, QTPRvalid, QTHRvalid = metrics.roc_curve(Y_valid, valid_Qident)
    QAUCvalid = metrics.roc_auc_score(Y_valid, valid_Qident)

    # apply quantized and pruned model
    valid_Qident_pruned = TauQIdentifierModelPruned.predict([X1_valid, X2_valid])
    QFPRvalid_pruned, QTPRvalid_pruned, QTHRvalid_pruned = metrics.roc_curve(Y_valid, valid_Qident_pruned)
    QAUCvalid_pruned = metrics.roc_auc_score(Y_valid, valid_Qident_pruned)\

    plt.figure(figsize=(10,10))
    plt.plot(TPRvalid, FPRvalid, label='ROC - Baseline, AUC = %.3f' % (AUCvalid),   color='blue',lw=2)
    plt.plot(TPRvalid_pruned, FPRvalid_pruned, label='ROC - Pruned ('+str(options.sparsity)+'), AUC = %.3f' % (AUCvalid_pruned),   color='blue',lw=2, ls='--')
    plt.plot(QTPRvalid, QFPRvalid, label='ROC - Quantized, AUC = %.3f' % (QAUCvalid), color='green',lw=2)
    plt.plot(QTPRvalid_pruned, QFPRvalid_pruned, label='ROC - Quantized & Pruned ('+str(options.sparsity)+'), AUC = %.3f' % (QAUCvalid_pruned), color='green',lw=2, ls='--')
    plt.grid(linestyle=':')
    plt.legend(loc = 'upper left', fontsize=16)
    # plt.xlim(0.8,1.01)
    plt.yscale('log')
    #plt.ylim(0.01,1)
    plt.xlabel('Signal Efficiency')
    plt.ylabel('Background Efficiency')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(indir+'/validation_roc_fullComparisons'+sparsityTag+'Pruned.pdf')
    plt.close()

    plt.figure(figsize=(10,10))
    plt.plot(TPRvalid, FPRvalid, label='ROC - Baseline, AUC = %.3f' % (AUCvalid),   color='blue',lw=2)
    plt.plot(TPRvalid_pruned, FPRvalid_pruned, label='ROC - Pruned ('+str(options.sparsity)+'), AUC = %.3f' % (AUCvalid_pruned),   color='blue',lw=2, ls='--')
    plt.plot(QTPRvalid, QFPRvalid, label='ROC - Quantized, AUC = %.3f' % (QAUCvalid), color='green',lw=2)
    plt.plot(QTPRvalid_pruned, QFPRvalid_pruned, label='ROC - Quantized & Pruned ('+str(options.sparsity)+'), AUC = %.3f' % (QAUCvalid_pruned), color='green',lw=2, ls='--')
    plt.grid(linestyle=':')
    plt.legend(loc = 'upper left', fontsize=16)
    plt.xlim(0.8,1.01)
    plt.ylim(0.03,1)
    plt.yscale('log')
    plt.xlabel('Signal Efficiency')
    plt.ylabel('Background Efficiency')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(indir+'/validation_roc_fullComparisons'+sparsityTag+'Pruned_zoomed.pdf')
    plt.close()

