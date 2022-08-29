from optparse import OptionParser
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import shap
import os

np.random.seed(7)

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
    parser.add_option('--caloClNxM',    dest='caloClNxM',                      default="9x9")
    (options, args) = parser.parse_args()
    print(options)

    # get clusters' shape dimensions
    N = int(options.caloClNxM.split('x')[0])
    M = int(options.caloClNxM.split('x')[1])

    ############################## Get model inputs ##############################

    indir = '/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNNCalibrator'+options.caloClNxM+'Training'+options.inTag
    outdir = '/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNNCalibrator'+options.caloClNxM+'Training'+options.inTag
    os.system('mkdir -p '+outdir+'/TauCNNInspector_plots')

    # X1 is (None, N, M, 3)
    #       N runs over eta, M runs over phi
    #       3 features are: EgIet, Iem, Ihad
    # 
    # X2 is (None, 2)
    #       2 are eta and phi values
    #
    # Y is (None, 4)
    #       target: visPt, visEta, visPhi, DM

    X1 = np.load(indir+'/X_CNN_'+options.caloClNxM+'_forCalibrator.npz')['arr_0']
    X2 = np.load(indir+'/X_Dense_'+options.caloClNxM+'_forCalibrator.npz')['arr_0']
    Y = np.load(indir+'/Y'+options.caloClNxM+'_forCalibrator.npz')['arr_0']


    ############################## Model validation ##############################

    X1_valid = np.load('/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNNEvaluator'+options.caloClNxM+options.inTag+'/X_Calib_CNN_'+options.caloClNxM+'_forEvaluator.npz')['arr_0']
    X2_valid = np.load('/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNNEvaluator'+options.caloClNxM+options.inTag+'/X_Calib_Dense_'+options.caloClNxM+'_forEvaluator.npz')['arr_0']
    Y_valid  = np.load('/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauCNNEvaluator'+options.caloClNxM+options.inTag+'/Y_Calib_'+options.caloClNxM+'_forEvaluator.npz')['arr_0']
    
    dfTrain = pd.DataFrame()
    dfTrain['uncalib_pt'] = np.sum(np.sum(np.sum(X1, axis=3), axis=2), axis=1).ravel()
    dfTrain['gen_pt']     = Y[:,0].ravel()
    dfTrain['gen_eta']    = Y[:,1].ravel()
    dfTrain['gen_phi']    = Y[:,2].ravel()
    dfTrain['gen_dm']     = Y[:,3].ravel()

    dfValid = pd.DataFrame()
    dfValid['uncalib_pt'] = np.sum(np.sum(np.sum(X1_valid, axis=3), axis=2), axis=1).ravel()
    dfValid['gen_pt']     = Y_valid[:,0].ravel()
    dfValid['gen_eta']    = Y_valid[:,1].ravel()
    dfValid['gen_phi']    = Y_valid[:,2].ravel()
    dfValid['gen_dm']     = Y_valid[:,3].ravel()


    # PLOTS INCLUSIVE
    plt.figure(figsize=(10,10))
    plt.hist(dfValid['uncalib_pt']/dfValid['gen_pt'], bins=np.arange(0,5,0.1), label=r'Valid. Uncalibrated response, $\mu$: %.2f, $\sigma$ : %.2f' % (np.mean(dfValid['uncalib_pt']/dfValid['gen_pt']), np.std(dfValid['uncalib_pt']/dfValid['gen_pt'])), color='red',  lw=2, density=True, histtype='step')
    plt.hist(dfTrain['uncalib_pt']/dfTrain['gen_pt'], bins=np.arange(0,5,0.1), label=r'Train. Uncalibrated response, $\mu$: %.2f, $\sigma$ : %.2f' % (np.mean(dfTrain['uncalib_pt']/dfTrain['gen_pt']), np.std(dfTrain['uncalib_pt']/dfTrain['gen_pt'])), color='blue', lw=2, density=True, histtype='step')
    plt.xlabel(r'$p_{T}^{L1 \tau} / p_{T}^{Gen \tau}$')
    plt.ylabel(r'a.u.')
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle='dotted')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/TauCNNInspector_plots/input_responses_comparison.pdf')
    plt.close()

    dfValid = dfValid[dfValid['gen_pt'] <= 500].copy(deep=True)
    dfTrain = dfTrain[dfTrain['gen_pt'] <= 500].copy(deep=True)

    print('MAX uncalib_pt')
    print('valid -',dfValid['uncalib_pt'].max())
    print('train -',dfTrain['uncalib_pt'].max())
    print('\nMIN uncalib_pt')
    print('valid -',dfValid['uncalib_pt'].min())
    print('train -',dfTrain['uncalib_pt'].min())
    print('\n')
    print('MAX gen_pt')
    print('valid -',dfValid['gen_pt'].max())
    print('train -',dfTrain['gen_pt'].max())
    print('\nMIN gen_pt')
    print('valid -',dfValid['gen_pt'].min())
    print('train -',dfTrain['gen_pt'].min())
    print('\n')
    print('MAX uncalib_pt/gen_pt')
    print('valid -',(dfValid['uncalib_pt']/dfValid['gen_pt']).max())
    print('train -',(dfTrain['uncalib_pt']/dfTrain['gen_pt']).max())
    print('\nMIN uncalib_pt/gen_pt')
    print('valid -',(dfValid['uncalib_pt']/dfValid['gen_pt']).min())
    print('train -',(dfTrain['uncalib_pt']/dfTrain['gen_pt']).min())
    print('\nMAX uncalib_pt-gen_pt')
    print('valid -',(dfValid['uncalib_pt']-dfValid['gen_pt']).max())
    print('train -',(dfTrain['uncalib_pt']-dfTrain['gen_pt']).max())
    print('\nMIN uncalib_pt-gen_pt')
    print('valid -',(dfValid['uncalib_pt']-dfValid['gen_pt']).min())
    print('train -',(dfTrain['uncalib_pt']-dfTrain['gen_pt']).min())


