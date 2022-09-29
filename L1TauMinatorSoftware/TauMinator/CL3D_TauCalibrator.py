from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from optparse import OptionParser
import xgboost as xgb
import pandas as pd
import numpy as np
import pickle
import sys
import os

np.random.seed(7)

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import mplhep
plt.style.use(mplhep.style.CMS)


def save_obj(obj,dest):
    with open(dest,'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(source):
    with open(source,'rb') as f:
        return pickle.load(f)

#######################################################################
######################### SCRIPT BODY #################################
#######################################################################
if __name__ == "__main__" :

    parser = OptionParser()
    parser.add_option("--v",            dest="v",                              default=None)
    parser.add_option("--date",         dest="date",                           default=None)
    parser.add_option("--inTag",        dest="inTag",                          default="")
    parser.add_option('--train',        dest='train',     action='store_true', default=False)
    (options, args) = parser.parse_args()
    print(options)

    ############################## Get model inputs ##############################

    indir = '/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauBDTCalibratorTraining'+options.inTag
    os.system('mkdir -p '+indir+'/TauBDTCalibrator_plots')
    os.system('mkdir -p '+indir+'/TauBDTCalibrator')

    dfTr = pd.read_pickle(indir+'/X_Calib_BDT_forCalibrator.pkl')
    dfTr['cl3d_abseta'] = abs(dfTr['cl3d_eta']).copy(deep=True)


    ############################## Define model features and hyperparameters ##############################

    # features used for the C2 calibration step - FULL AVAILABLE
    # features = ['cl3d_showerlength', 'cl3d_coreshowerlength', 'cl3d_firstlayer', 'cl3d_seetot', 'cl3d_seemax', 'cl3d_spptot', 'cl3d_sppmax', 'cl3d_szz', 'cl3d_srrtot', 'cl3d_srrmax', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz']
    # boostRounds = 1000
    # max_depth = 2
    
    # features used for the C2 calibration step - OPTIMIZED
    features = ['cl3d_showerlength', 'cl3d_coreshowerlength', 'cl3d_abseta', 'cl3d_spptot', 'cl3d_srrmean', 'cl3d_meanz']
    featuresN = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5']
    boostRounds = 60
    max_depth = 5

    # rename features with numbers (needed by ONNX)
    for i in range(len(features)): dfTr[featuresN[i]] = dfTr[features[i]].copy(deep=True)

    # features to plot for the C3 calibration step
    vars = ['tau_visPt', 'tau_bin_visPt', 'cl3d_pt_c2', 'cl3d_response_c2']

    # variables to plot
    plot_var = ['tau_visPt', 'tau_absVisEta', 'tau_bin_visEta', 'tau_bin_visPt', 'cl3d_pt', 'cl3d_response', 'cl3d_abseta', 'cl3d_pt_c1', 'cl3d_response_c1', 'cl3d_pt_c2', 'cl3d_response_c2', 'cl3d_pt_c3', 'cl3d_response_c3']

    etamin = 1.5
    ptmin = 15

    if options.train:
        dfTr['cl3d_response'] = dfTr['cl3d_pt'] / dfTr['tau_visPt']

        ######################### C1 CALIBRATION TRAINING (PU eta dependent calibration) #########################

        print('\n** INFO: training calibration C1')

        input_c1 = dfTr[['cl3d_abseta']]
        target_c1 = dfTr['tau_visPt'] - dfTr['cl3d_pt']
        C1model = LinearRegression().fit(input_c1, target_c1)

        save_obj(C1model, indir+'/TauBDTCalibrator/C1model.pkl')
        with open(indir+'/TauBDTCalibrator/C1model.txt', 'w') as f:
            f.write('m  = '+str(C1model.coef_[0])+'\n')
            f.write('z0 = '+str(C1model.intercept_)+'\n')

        dfTr['cl3d_c1'] = C1model.predict(dfTr[['cl3d_abseta']])
        dfTr['cl3d_pt_c1'] = dfTr['cl3d_c1'] + dfTr['cl3d_pt']
        dfTr['cl3d_response_c1'] = dfTr['cl3d_pt_c1'] / dfTr['tau_visPt']

        ######################### C2 CALIBRATION TRAINING (DM dependent calibration) #########################

        print('\n** INFO: training calibration C2')

        input_c2 = dfTr[featuresN]
        target_c2 = dfTr['tau_visPt'] / dfTr['cl3d_pt_c1']
        C2model = xgb.XGBRegressor(booster='gbtree', n_estimators=boostRounds, learning_rate=0.1, max_depth=max_depth).fit(input_c2, target_c2) # eval_metric=mean_absolute_error

        save_obj(C2model, indir+'/TauBDTCalibrator/C2model.pkl')
        C2model.save_model(indir+'/TauBDTCalibrator/C2model.model')

        dfTr['cl3d_c2'] = C2model.predict(dfTr[featuresN])
        dfTr['cl3d_pt_c2'] = dfTr['cl3d_c2'] * dfTr['cl3d_pt_c1']
        dfTr['cl3d_response_c2'] = dfTr['cl3d_pt_c2'] / dfTr['tau_visPt']

        # print importance of the features used for training
        feature_importance = C2model.feature_importances_
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + 0.5
        fig = plt.figure(figsize=(15, 10))
        plt.gcf().subplots_adjust(left=0.25)
        plt.barh(pos, feature_importance[sorted_idx], align='center')
        plt.yticks(pos, np.array(features)[sorted_idx])
        plt.xlabel(r'Importance score')
        mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
        plt.savefig(indir+'/TauBDTCalibrator_plots/featureImportance_modelC2.pdf')

        ######################### C3 CALIBRATION TRAINING (E dependent calibration) #########################

        print('\n** INFO: training calibration C3')

        dfTr['tau_absVisEta'] = np.abs(dfTr['tau_visEta'])
        dfTr['tau_bin_visEta'] = ((dfTr['tau_absVisEta'] - etamin)/0.1).astype('int32')
        dfTr['tau_bin_visPt']  = ((dfTr['tau_visPt'] - ptmin)/5).astype('int32')

        meansTrainPt = dfTr[vars].groupby('tau_bin_visPt').mean() 
        rmssTrainPt = dfTr[vars].groupby('tau_bin_visPt').std() 

        meansTrainPt['logpt1'] = np.log(meansTrainPt['cl3d_pt_c2'])
        meansTrainPt['logpt2'] = meansTrainPt.logpt1**2
        meansTrainPt['logpt3'] = meansTrainPt.logpt1**3
        meansTrainPt['logpt4'] = meansTrainPt.logpt1**4

        input_c3 = meansTrainPt[['logpt1', 'logpt2', 'logpt3', 'logpt4']]
        target_c3 = meansTrainPt['cl3d_response_c2']
        C3model = LinearRegression().fit(input_c3, target_c3)

        save_obj(C3model, indir+'/TauBDTCalibrator/C3model.pkl')
        with open(indir+'/TauBDTCalibrator/C3model.txt', 'w') as f:
            f.write('k0 = '+str(C3model.intercept_)+'\n')
            f.write('k1 = '+str(C3model.coef_[0])+'\n')
            f.write('k2 = '+str(C3model.coef_[1])+'\n')
            f.write('k3 = '+str(C3model.coef_[2])+'\n')
            f.write('k4 = '+str(C3model.coef_[3])+'\n')

        logpt1 = np.log(abs(dfTr['cl3d_pt_c2']))
        logpt2 = logpt1**2
        logpt3 = logpt1**3
        logpt4 = logpt1**4
        dfTr['cl3d_c3'] = C3model.predict(np.vstack([logpt1, logpt2, logpt3, logpt4]).T)
        dfTr['cl3d_pt_c3'] = dfTr['cl3d_pt_c2'] / dfTr['cl3d_c3']
        dfTr['cl3d_response_c3'] = dfTr['cl3d_pt_c3'] / dfTr['tau_visPt']

    else:
        C1model = load_obj(indir+'/TauBDTCalibrator/C1model.pkl')
        C2model = load_obj(indir+'/TauBDTCalibrator/C2model.pkl')
        C3model = load_obj(indir+'/TauBDTCalibrator/C3model.pkl')

        dfTr['cl3d_response'] = dfTr['cl3d_pt'] / dfTr['tau_visPt']
        # application calibration 1
        dfTr['cl3d_c1'] = C1model.predict(dfTr[['cl3d_abseta']])
        dfTr['cl3d_pt_c1'] = dfTr['cl3d_c1'] + dfTr['cl3d_pt']
        dfTr['cl3d_response_c1'] = dfTr['cl3d_pt_c1'] / dfTr['tau_visPt']
        # application calibration 2
        dfTr['cl3d_c2'] = C2model.predict(dfTr[featuresN])
        dfTr['cl3d_pt_c2'] = dfTr['cl3d_c2'] * dfTr['cl3d_pt_c1']
        dfTr['cl3d_response_c2'] = dfTr['cl3d_pt_c2'] / dfTr['tau_visPt']
        # application calibration 3
        logpt1 = np.log(abs(dfTr['cl3d_pt_c2']))
        logpt2 = logpt1**2
        logpt3 = logpt1**3
        logpt4 = logpt1**4
        dfTr['cl3d_c3'] = C3model.predict(np.vstack([logpt1, logpt2, logpt3, logpt4]).T)
        dfTr['cl3d_pt_c3'] = dfTr['cl3d_pt_c2'] / dfTr['cl3d_c3']
        dfTr['cl3d_response_c3'] = dfTr['cl3d_pt_c3'] / dfTr['tau_visPt']

    ############################## Model validation ##############################

    dfVal = pd.read_pickle(indir+'/X_Calib_BDT_forEvaluator.pkl')
    dfVal['cl3d_abseta'] = abs(dfVal['cl3d_eta']).copy(deep=True)
    for i in range(len(features)): dfVal[featuresN[i]] = dfVal[features[i]].copy(deep=True)

    # VALIDATION DATASET
    dfVal['cl3d_response'] = dfVal['cl3d_pt'] / dfVal['tau_visPt']
    # application calibration 1
    dfVal['cl3d_c1'] = C1model.predict(dfVal[['cl3d_abseta']])
    dfVal['cl3d_pt_c1'] = dfVal['cl3d_c1'] + dfVal['cl3d_pt']
    dfVal['cl3d_response_c1'] = dfVal['cl3d_pt_c1'] / dfVal['tau_visPt']
    # application calibration 2
    dfVal['cl3d_c2'] = C2model.predict(dfVal[featuresN])
    dfVal['cl3d_pt_c2'] = dfVal['cl3d_c2'] * dfVal['cl3d_pt_c1']
    dfVal['cl3d_response_c2'] = dfVal['cl3d_pt_c2'] / dfVal['tau_visPt']
    # application calibration 3
    logpt1 = np.log(abs(dfVal['cl3d_pt_c2']))
    logpt2 = logpt1**2
    logpt3 = logpt1**3
    logpt4 = logpt1**4
    dfVal['cl3d_c3'] = C3model.predict(np.vstack([logpt1, logpt2, logpt3, logpt4]).T)
    dfVal['cl3d_pt_c3'] = dfVal['cl3d_pt_c2'] / dfVal['cl3d_c3']
    dfVal['cl3d_response_c3'] = dfVal['cl3d_pt_c3'] / dfVal['tau_visPt']

    # Fill per DM datasets
    dfTrainingDM0 = dfTr[(dfTr['tau_DM']==0)]
    dfTrainingDM1 = dfTr[(dfTr['tau_DM']==1) | (dfTr['tau_DM']==2)]
    dfTrainingDM10 = dfTr[(dfTr['tau_DM']==10)]
    dfTrainingDM11 = dfTr[(dfTr['tau_DM']==11) | (dfTr['tau_DM']==12)]

    dfValidationDM0 = dfVal[(dfVal['tau_DM']==0)]
    dfValidationDM1 = dfVal[(dfVal['tau_DM']==1) | (dfVal['tau_DM']==2)]
    dfValidationDM10 = dfVal[(dfVal['tau_DM']==10)]
    dfValidationDM11 = dfVal[(dfVal['tau_DM']==11) | (dfVal['tau_DM']==12)]

    # TRAINING RESPONSE
    plt.figure(figsize=(10,10))
    plt.hist(dfTr['cl3d_response'], bins=np.arange(0., 5., 0.1),  label=r'Uncalibrated :  $\mu$ = %.2f, $\sigma$ =  %.2f' % (dfTr['cl3d_response'].mean(), dfTr['cl3d_response'].std()), color='red',    histtype='step', lw=2, density=True)
    plt.hist(dfTr['cl3d_response_c1'], bins=np.arange(0., 5., 0.1),  label=r'Calibration 1 :  $\mu$ = %.2f, $\sigma$ =  %.2f' % (dfTr['cl3d_response_c1'].mean(), dfTr['cl3d_response_c1'].std()), color='blue',    histtype='step', lw=2, density=True)
    plt.hist(dfTr['cl3d_response_c2'], bins=np.arange(0., 5., 0.1),  label=r'Calibration 2 :  $\mu$ = %.2f, $\sigma$ =  %.2f' % (dfTr['cl3d_response_c2'].mean(), dfTr['cl3d_response_c2'].std()), color='green',  histtype='step', lw=2, density=True)
    plt.hist(dfTr['cl3d_response_c3'], bins=np.arange(0., 5., 0.1),  label=r'Calibration 3 :  $\mu$ = %.2f, $\sigma$ =  %.2f' % (dfTr['cl3d_response_c3'].mean(), dfTr['cl3d_response_c3'].std()), color='black',   histtype='step', lw=2, density=True)
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle=':')
    plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
    plt.ylabel(r'a. u.')
    plt.xlim(0, 4)
    # plt.ylim(0,1750)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(indir+'/TauBDTCalibrator_plots/calibResponse_training.pdf')
    plt.close()

    # 2D TRAINING RESPONSE
    plt.figure(figsize=(10,10))
    plt.scatter(dfTr['cl3d_response'], np.abs(dfTr['tau_visEta']), label='Uncalibrated', color='red', marker='.', alpha=0.3)
    plt.scatter(dfTr['cl3d_response_c3'], np.abs(dfTr['tau_visEta']), label='Calib. 3', color='black', marker='.', alpha=0.3)
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle=':')
    plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
    plt.ylabel(r'$|\eta^{gen,\tau}|$')
    plt.xlim(0, 2)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(indir+'/TauBDTCalibrator_plots/response_training_vs_eta.pdf')
    plt.close()

    plt.figure(figsize=(10,10))
    plt.scatter(dfTr['cl3d_response'], dfTr['tau_visPt'], label='Uncalibrated', color='red', marker='.', alpha=0.3)
    plt.scatter(dfTr['cl3d_response_c3'], dfTr['tau_visPt'], label='Calib. 3', color='black', marker='.', alpha=0.3)
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle=':')
    plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
    plt.ylabel(r'$p_{T}^{gen,\tau}$')
    plt.xlim(0, 2)
    # plt.ylim(0, 200)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(indir+'/TauBDTCalibrator_plots/response_training_vs_pt.pdf')
    plt.close()

    # VALIDATION RESPONSE
    plt.figure(figsize=(10,10))
    plt.hist(dfVal['cl3d_response'], bins=np.arange(0., 5., 0.1),  label=r'Uncalibrated : $\mu$ = %.2f, $\sigma$ =  %.2f' % (dfVal['cl3d_response'].mean(), dfVal['cl3d_response'].std(),), color='red',    histtype='step', lw=2, density=True)
    plt.hist(dfVal['cl3d_response_c1'], bins=np.arange(0., 5., 0.1),  label=r'Calib. 1 : $\mu$ = %.2f, $\sigma$ =  %.2f' % (dfVal['cl3d_response_c1'].mean(), dfVal['cl3d_response_c1'].std(),), color='blue',    histtype='step', lw=2, density=True)
    plt.hist(dfVal['cl3d_response_c2'], bins=np.arange(0., 5., 0.1),  label=r'Calib. 2 : $\mu$ = %.2f, $\sigma$ =  %.2f' % (dfVal['cl3d_response_c2'].mean(), dfVal['cl3d_response_c2'].std(),), color='limegreen',  histtype='step', lw=2, density=True)
    plt.hist(dfVal['cl3d_response_c3'], bins=np.arange(0., 5., 0.1),  label=r'Calib. 3 : $\mu$ = %.2f, $\sigma$ =  %.2f' % (dfVal['cl3d_response_c3'].mean(), dfVal['cl3d_response_c3'].std(),), color='black',   histtype='step', lw=2, density=True)
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle=':')
    plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
    plt.ylabel(r'a. u.')
    plt.xlim(0, 4)
    # plt.ylim(0, 800)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(indir+'/TauBDTCalibrator_plots/calibResponse_validation.pdf')
    plt.close()

    # 2D VALIDATION RESPONSE
    plt.figure(figsize=(10,10))
    plt.scatter(dfVal['cl3d_response'], np.abs(dfVal['tau_visEta']), label='Uncalibrated', color='red', marker='.', alpha=0.3)
    plt.scatter(dfVal['cl3d_response_c3'], np.abs(dfVal['tau_visEta']), label='Calib. 3', color='black', marker='.', alpha=0.3)
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle=':')
    plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
    plt.ylabel(r'$|\eta^{gen,\tau}|$')
    plt.xlim(0, 2)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(indir+'/TauBDTCalibrator_plots/response_validation_vs_eta.pdf')
    plt.close()

    plt.figure(figsize=(10,10))
    plt.scatter(dfVal['cl3d_response'], dfVal['tau_visPt'], label='Uncalibrated', color='red', marker='.', alpha=0.3)
    plt.scatter(dfVal['cl3d_response_c3'], dfVal['tau_visPt'], label='Calib. 3', color='black', marker='.', alpha=0.3)
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle=':')
    plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
    plt.ylabel(r'$p_{T}^{gen,\tau}$')
    plt.xlim(0, 2)
    # plt.ylim(0, 200)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(indir+'/TauBDTCalibrator_plots/response_validation_vs_pt.pdf')
    plt.close()

    # 2D VALIDATION C1 RESPONSE
    plt.figure(figsize=(10,10))
    plt.scatter(dfVal['cl3d_response'], np.abs(dfVal['tau_visEta']), label='Uncalibrated', color='red', marker='.', alpha=0.3)
    plt.scatter(dfVal['cl3d_response_c1'], np.abs(dfVal['tau_visEta']), label='Calib. 1', color='blue', marker='.', alpha=0.3)
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle=':')
    plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
    plt.ylabel(r'$|\eta^{gen,\tau}|$')
    plt.xlim(0, 2)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(indir+'/TauBDTCalibrator_plots/responseC1_validation_vs_eta.pdf')
    plt.close()

    plt.figure(figsize=(10,10))
    plt.scatter(dfVal['cl3d_response'], dfVal['tau_visPt'], label='Uncalibrated', color='red', marker='.', alpha=0.3)
    plt.scatter(dfVal['cl3d_response_c1'], dfVal['tau_visPt'], label='Calib. 1', color='blue', marker='.', alpha=0.3)
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle=':')
    plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
    plt.ylabel(r'$p_{T}^{gen,\tau}$')
    plt.xlim(0, 2)
    # plt.ylim(0, 200)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(indir+'/TauBDTCalibrator_plots/responseC1_validation_vs_pt.pdf')
    plt.close()

    # 2D VALIDATION C2 RESPONSE
    plt.figure(figsize=(10,10))
    plt.scatter(dfVal['cl3d_response'], np.abs(dfVal['tau_visEta']), label='Uncalibrated', color='red', marker='.', alpha=0.3)
    plt.scatter(dfVal['cl3d_response_c2'], np.abs(dfVal['tau_visEta']), label='Calib. 2', color='limegreen', marker='.', alpha=0.3)
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle=':')
    plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
    plt.ylabel(r'$|\eta^{gen,\tau}|$')
    plt.xlim(0, 2)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(indir+'/TauBDTCalibrator_plots/responseC2_validation_vs_eta.pdf')
    plt.close()

    plt.figure(figsize=(10,10))
    plt.scatter(dfVal['cl3d_response'], dfVal['tau_visPt'], label='Uncalibrated', color='red', marker='.', alpha=0.3)
    plt.scatter(dfVal['cl3d_response_c2'], dfVal['tau_visPt'], label='Calib. 2', color='limegreen', marker='.', alpha=0.3)
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle=':')
    plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
    plt.ylabel(r'$p_{T}^{gen,\tau}$')
    plt.xlim(0, 2)
    # plt.ylim(0, 200)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(indir+'/TauBDTCalibrator_plots/responseC2_validation_vs_pt.pdf')
    plt.close()

    # SEPARATE DMs RESPONSE
    DMdict = {
            0  : r'$h^{\pm}$',
            1  : r'$h^{\pm}\pi^{0}$',
            10 : r'$h^{\pm}h^{\mp}h^{\pm}$',
            11 : r'$h^{\pm}h^{\mp}h^{\pm}\pi^{0}$',
        }

    plt.figure(figsize=(10,10))
    plt.hist(dfValidationDM0['cl3d_response'], bins=np.arange(0., 5., 0.1),  label=DMdict[0]+r' : $\mu$ = %.2f, $\sigma$ =  %.2f' % (dfValidationDM0['cl3d_response'].mean(), dfValidationDM0['cl3d_response'].std()), color='limegreen',    histtype='step', lw=2, density=True)
    plt.hist(dfValidationDM1['cl3d_response'], bins=np.arange(0., 5., 0.1),  label=DMdict[1]+r' : $\mu$ = %.2f, $\sigma$ =  %.2f' % (dfValidationDM1['cl3d_response'].mean(), dfValidationDM1['cl3d_response'].std()), color='blue',    histtype='step', lw=2, density=True)
    plt.hist(dfValidationDM10['cl3d_response'], bins=np.arange(0., 5., 0.1),  label=DMdict[10]+r' : $\mu$ = %.2f, $\sigma$ =  %.2f' % (dfValidationDM10['cl3d_response'].mean(), dfValidationDM10['cl3d_response'].std()), color='orange',  histtype='step', lw=2, density=True)
    plt.hist(dfValidationDM11['cl3d_response'], bins=np.arange(0., 5., 0.1),  label=DMdict[11]+r' : $\mu$ = %.2f, $\sigma$ =  %.2f' % (dfValidationDM11['cl3d_response'].mean(), dfValidationDM11['cl3d_response'].std()), color='fuchsia',   histtype='step', lw=2, density=True)
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle=':')
    plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
    plt.ylabel(r'a. u.')
    plt.xlim(0, 4)
    # plt.ylim(0, 450)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(indir+'/TauBDTCalibrator_plots/DM_c0Response_validation.pdf')
    plt.close()

    plt.figure(figsize=(10,10))
    plt.hist(dfValidationDM0['cl3d_response_c1'], bins=np.arange(0., 5., 0.1),  label=DMdict[0]+r' : $\mu$ = %.2f, $\sigma$ =  %.2f' % (dfValidationDM0['cl3d_response_c1'].mean(), dfValidationDM0['cl3d_response_c1'].std()), color='limegreen',    histtype='step', lw=2, density=True)
    plt.hist(dfValidationDM1['cl3d_response_c1'], bins=np.arange(0., 5., 0.1),  label=DMdict[1]+r' : $\mu$ = %.2f, $\sigma$ =  %.2f' % (dfValidationDM1['cl3d_response_c1'].mean(), dfValidationDM1['cl3d_response_c1'].std()), color='blue',    histtype='step', lw=2, density=True)
    plt.hist(dfValidationDM10['cl3d_response_c1'], bins=np.arange(0., 5., 0.1),  label=DMdict[10]+r' : $\mu$ = %.2f, $\sigma$ =  %.2f' % (dfValidationDM10['cl3d_response_c1'].mean(), dfValidationDM10['cl3d_response_c1'].std()), color='orange',  histtype='step', lw=2, density=True)
    plt.hist(dfValidationDM11['cl3d_response_c1'], bins=np.arange(0., 5., 0.1),  label=DMdict[11]+r' : $\mu$ = %.2f, $\sigma$ =  %.2f' % (dfValidationDM11['cl3d_response_c1'].mean(), dfValidationDM11['cl3d_response_c1'].std()), color='fuchsia',   histtype='step', lw=2, density=True)
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle=':')
    plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
    plt.ylabel(r'a. u.')
    plt.xlim(0, 4)
    # plt.ylim(0, 450)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(indir+'/TauBDTCalibrator_plots/DM_c1Response_validation.pdf')
    plt.close()

    plt.figure(figsize=(10,10))
    plt.hist(dfValidationDM0['cl3d_response_c2'], bins=np.arange(0., 5., 0.1),  label=DMdict[0]+r' : $\mu$ = %.2f, $\sigma$ =  %.2f' % (dfValidationDM0['cl3d_response_c2'].mean(), dfValidationDM0['cl3d_response_c2'].std()), color='limegreen',    histtype='step', lw=2, density=True)
    plt.hist(dfValidationDM1['cl3d_response_c2'], bins=np.arange(0., 5., 0.1),  label=DMdict[1]+r' : $\mu$ = %.2f, $\sigma$ =  %.2f' % (dfValidationDM1['cl3d_response_c2'].mean(), dfValidationDM1['cl3d_response_c2'].std()), color='blue',    histtype='step', lw=2, density=True)
    plt.hist(dfValidationDM10['cl3d_response_c2'], bins=np.arange(0., 5., 0.1),  label=DMdict[10]+r' : $\mu$ = %.2f, $\sigma$ =  %.2f' % (dfValidationDM0['cl3d_response_c2'].mean(), dfValidationDM10['cl3d_response_c2'].std()), color='orange',  histtype='step', lw=2, density=True)
    plt.hist(dfValidationDM11['cl3d_response_c2'], bins=np.arange(0., 5., 0.1),  label=DMdict[11]+r' : $\mu$ = %.2f, $\sigma$ =  %.2f' % (dfValidationDM11['cl3d_response_c2'].mean(), dfValidationDM11['cl3d_response_c2'].std()), color='fuchsia',   histtype='step', lw=2, density=True)
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle=':')
    plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
    plt.ylabel(r'a. u.')
    plt.xlim(0, 4)
    # plt.ylim(0, 450)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(indir+'/TauBDTCalibrator_plots/DM_c2Response_validation.pdf')
    plt.close()

    plt.figure(figsize=(10,10))
    plt.hist(dfValidationDM0['cl3d_response_c3'], bins=np.arange(0., 5., 0.1),  label=DMdict[0]+r' : $\mu$ = %.2f, $\sigma$ =  %.2f' % (dfValidationDM0['cl3d_response_c3'].mean(), dfValidationDM0['cl3d_response_c3'].std()), color='limegreen',    histtype='step', lw=2, density=True)
    plt.hist(dfValidationDM1['cl3d_response_c3'], bins=np.arange(0., 5., 0.1),  label=DMdict[1]+r' : $\mu$ = %.2f, $\sigma$ =  %.2f' % (dfValidationDM1['cl3d_response_c3'].mean(), dfValidationDM1['cl3d_response_c3'].std()), color='blue',    histtype='step', lw=2, density=True)
    plt.hist(dfValidationDM10['cl3d_response_c3'], bins=np.arange(0., 5., 0.1),  label=DMdict[10]+r' : $\mu$ = %.2f, $\sigma$ =  %.2f' % (dfValidationDM10['cl3d_response_c3'].mean(), dfValidationDM10['cl3d_response_c3'].std()), color='orange',  histtype='step', lw=2, density=True)
    plt.hist(dfValidationDM11['cl3d_response_c3'], bins=np.arange(0., 5., 0.1),  label=DMdict[11]+r' : $\mu$ = %.2f, $\sigma$ =  %.2f' % (dfValidationDM11['cl3d_response_c3'].mean(), dfValidationDM11['cl3d_response_c3'].std()), color='fuchsia',   histtype='step', lw=2, density=True)
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle=':')
    plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
    plt.ylabel(r'a. u.')
    plt.xlim(0, 4)
    # plt.ylim(0, 450)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(indir+'/TauBDTCalibrator_plots/DM_c3Response_validation.pdf')
    plt.close()

    # TRAINING VALIDATION OVERLAY
    plt.figure(figsize=(10,10))
    plt.hist(dfTr['cl3d_response'], bins=np.arange(0., 5., 0.1),  label=r'Uncalibrated response :  $\mu$ = %.2f, $\sigma$ =  %.2f' % (dfTr['cl3d_response'].mean(), dfTr['cl3d_response'].std()), color='red',    histtype='step', lw=2, density=True)
    plt.hist(dfTr['cl3d_response_c3'], bins=np.arange(0., 5., 0.1),  label=r'Train. Calibrated response :  $\mu$ = %.2f, $\sigma$ =  %.2f' % (dfTr['cl3d_response_c3'].mean(), dfTr['cl3d_response_c3'].std()), color='blue',   histtype='step', lw=2, density=True)
    plt.hist(dfVal['cl3d_response_c3'], bins=np.arange(0., 5., 0.1),  label=r'Valid. Calibrated response :  $\mu$ = %.2f, $\sigma$ =  %.2f' % (dfVal['cl3d_response_c3'].mean(), dfVal['cl3d_response_c3'].std()), color='green',   histtype='step', lw=2, density=True)
    plt.legend(loc = 'upper right', fontsize=16)
    plt.grid(linestyle=':')
    plt.xlabel(r'$E_{T}^{L1,\tau}\ /\ p_{T}^{gen,\tau}$')
    plt.ylabel(r'a. u.')
    plt.xlim(0,5)
    # plt.ylim(0,1750)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(indir+'/TauBDTCalibrator_plots/responses_comparison.pdf')
    plt.close()
