from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from optparse import OptionParser
from sklearn import metrics
import xgboost as xgb
import pandas as pd
import numpy as np
import pickle
import shap
import sys
import os

np.random.seed(7)

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import mplhep
plt.style.use(mplhep.style.CMS)


def train_xgb(dfTr, features, target, hyperparams):
    

    return classifier, FPRtrain, TPRtrain, threshold_train, auroc_train, FPRtest, TPRtest, threshold_test, auroc_test

def save_obj(obj,dest):
    with open(dest,'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(source):
    with open(source,'rb') as f:
        return pickle.load(f)

def global_shap_importance(shap_values):
    cohorts = {"": shap_values}
    cohort_labels = list(cohorts.keys())
    cohort_exps = list(cohorts.values())
    for i in range(len(cohort_exps)):
        if len(cohort_exps[i].shape) == 2:
            cohort_exps[i] = cohort_exps[i].abs.mean(0)
    features = cohort_exps[0].data
    feature_names = cohort_exps[0].feature_names
    values = np.array([cohort_exps[i].values for i in range(len(cohort_exps))])
    feature_importance = pd.DataFrame(
        list(zip(feature_names, sum(values))), columns=['features', 'importance'])
    feature_importance.sort_values(
        by=['importance'], ascending=False, inplace=True)
    return feature_importance


#######################################################################
######################### SCRIPT BODY #################################
#######################################################################


if __name__ == "__main__" :

    parser = OptionParser()
    parser.add_option("--v",            dest="v",                              default=None)
    parser.add_option("--date",         dest="date",                           default=None)
    parser.add_option("--inTag",        dest="inTag",                          default="")
    parser.add_option('--train',        dest='train',     action='store_true', default=False)
    parser.add_option('--doRescale',    dest='doRescale', action='store_true', default=False)
    (options, args) = parser.parse_args()
    print(options)


    ############################## Get model inputs ##############################

    indir = '/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauBDTIdentifierTraining'+options.inTag
    os.system('mkdir -p '+indir+'/TauBDTIdentifier_plots')
    os.system('mkdir -p '+indir+'/TauBDTIdentifier')

    dfTr = pd.read_pickle(indir+'/X_Ident_BDT_forIdentifier.pkl')
    dfTr['cl3d_abseta'] = abs(dfTr['cl3d_eta']).copy(deep=True)


    ############################## Define model features and hyperparameters ##############################

    features_dict = {'cl3d_pt'               : [r'3D cluster $p_{T}$',[0,100,33]],
                     'cl3d_c1'               : [r'3D cluster C1',[0,30,30]],
                     'cl3d_c2'               : [r'3D cluster C2',[0,4,10]],
                     'cl3d_c3'               : [r'3D cluster C3',[0,10,20]],
                     'cl3d_abseta'           : [r'3D cluster |$\eta$|',[1.5,3.,10]], 
                     'cl3d_showerlength'     : [r'3D cluster shower length',[0.,35.,15]], 
                     'cl3d_coreshowerlength' : [r'Core shower length ',[0.,35.,15]], 
                     'cl3d_firstlayer'       : [r'3D cluster first layer',[0.,20.,20]], 
                     'cl3d_seetot'           : [r'3D cluster total $\sigma_{ee}$',[0.,0.15,10]],
                     'cl3d_seemax'           : [r'3D cluster max $\sigma_{ee}$',[0.,0.15,10]],
                     'cl3d_spptot'           : [r'3D cluster total $\sigma_{\phi\phi}$',[0.,0.1,10]],
                     'cl3d_sppmax'           : [r'3D cluster max $\sigma_{\phi\phi}$',[0.,0.1,10]],
                     'cl3d_szz'              : [r'3D cluster $\sigma_{zz}$',[0.,60.,20]], 
                     'cl3d_srrtot'           : [r'3D cluster total $\sigma_{rr}$',[0.,0.01,10]],
                     'cl3d_srrmax'           : [r'3D cluster max $\sigma_{rr}$',[0.,0.01,10]],
                     'cl3d_srrmean'          : [r'3D cluster mean $\sigma_{rr}$',[0.,0.01,10]], 
                     'cl3d_hoe'              : [r'Energy in CE-H / Energy in CE-E',[0.,4.,20]], 
                     'cl3d_meanz'            : [r'3D cluster meanz',[325.,375.,30]], 
    }
    if options.doRescale:
        features_dict = {'cl3d_c1'               : [r'3D cluster C1',[-33.,33.,66]],
                         'cl3d_c2'               : [r'3D cluster C2',[-33.,33.,66]],
                         'cl3d_c3'               : [r'3D cluster C3',[-33.,33.,66]],
                         'c1oc3'                 : [r'3D cluster c1oc3',[-33.,33.,66]],
                         'c1oc2'                 : [r'3D cluster c1oc2',[-33.,33.,66]],
                         'c2oc3'                 : [r'3D cluster c2oc3',[-33.,33.,66]],
                         'c1oPt'                 : [r'3D cluster c1oPt',[-33.,33.,66]],
                         'c2oPt'                 : [r'3D cluster c2oPt',[-33.,33.,66]],
                         'c3oPt'                 : [r'3D cluster c3oPt',[-33.,33.,66]],
                         'c1oPtc1'               : [r'3D cluster c1oPtc1',[-33.,33.,66]],
                         'c2oPtc2'               : [r'3D cluster c2oPtc2',[-33.,33.,66]],
                         'c3oPtc3'               : [r'3D cluster c3oPtc3',[-33.,33.,66]],
                         'fullScale'             : [r'3D cluster fullScale',[-33.,33.,66]],
                         'cl3d_abseta'           : [r'3D cluster |$\eta$|',[-33.,33.,66]], 
                         'cl3d_showerlength'     : [r'3D cluster shower length',[-33.,33.,66]], 
                         'cl3d_coreshowerlength' : [r'Core shower length ',[-33.,33.,66]], 
                         'cl3d_firstlayer'       : [r'3D cluster first layer',[-33.,33.,66]], 
                         'cl3d_seetot'           : [r'3D cluster total $\sigma_{ee}$',[-33.,33.,66]],
                         'cl3d_seemax'           : [r'3D cluster max $\sigma_{ee}$',[-33.,33.,66]],
                         'cl3d_spptot'           : [r'3D cluster total $\sigma_{\phi\phi}$',[-33.,33.,66]],
                         'cl3d_sppmax'           : [r'3D cluster max $\sigma_{\phi\phi}$',[-33.,33.,66]],
                         'cl3d_szz'              : [r'3D cluster $\sigma_{zz}$',[-33.,33.,66]], 
                         'cl3d_srrtot'           : [r'3D cluster total $\sigma_{rr}$',[-33.,33.,66]],
                         'cl3d_srrmax'           : [r'3D cluster max $\sigma_{rr}$',[-33.,33.,66]],
                         'cl3d_srrmean'          : [r'3D cluster mean $\sigma_{rr}$',[-33.,33.,66]], 
                         'cl3d_hoe'              : [r'Energy in CE-H / Energy in CE-E',[-33.,33.,66]], 
                         'cl3d_meanz'            : [r'3D cluster meanz',[-33.,33.,66]], 
        }

    # selected features from FS
    features = ['cl3d_pt', 'cl3d_coreshowerlength', 'cl3d_srrtot', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz']
    featuresN = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5']
    features2shift = ['cl3d_coreshowerlength']
    features2saturate = ['cl3d_c3', 'cl3d_srrtot', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz']
    saturation_dict = {'cl3d_c3': [0,30],
                       'cl3d_srrtot': [0, 0.02],
                       'cl3d_srrmean': [0, 0.01],
                       'cl3d_hoe': [0, 63],
                       'cl3d_meanz': [305, 535]
                    }
    # BDT hyperparameters
    hyperparams = {}
    hyperparams['objective']          = 'binary:logistic'
    hyperparams['booster']            = 'gbtree'
    hyperparams['eval_metric']        = 'logloss'
    hyperparams['reg_alpha']          = 9
    hyperparams['reg_lambda']         = 5
    hyperparams['max_depth']          = 4 # from HPO
    hyperparams['learning_rate']      = 0.35 # from HPO
    hyperparams['subsample']          = 0.22 # from HPO
    hyperparams['colsample_bytree']   = 0.7 # from HPO
    hyperparams['num_trees']          = 24 # from HPO

    # rescale the features if required
    if options.doRescale:
        print('\n** INFO: rescaling features to bound their values')

        #define a DF with the bound values of the features to use for the MinMaxScaler fit
        bounds4features = pd.DataFrame(columns=features2saturate)

        # shift features to be shifted
        for feat in features2shift:
            dfTr[feat] = dfTr[feat] - 25

        # saturate features
        for feat in features2saturate:
            dfTr[feat].clip(saturation_dict[feat][0],saturation_dict[feat][1], inplace=True)
            
            # fill the bounds DF
            bounds4features[feat] = np.linspace(saturation_dict[feat][0],saturation_dict[feat][1],100)

        scale_range = [-32,32]
        MMS = MinMaxScaler(scale_range)

        for feat in features2saturate:
            MMS.fit( np.array(bounds4features[feat]).reshape(-1,1) ) # we fit every feature separately otherwise MMS compleins for dimensionality
            dfTr[feat] = MMS.transform( np.array(dfTr[feat]).reshape(-1,1) )

    # rename features with numbers (needed by ONNX)
    for i in range(len(features)): dfTr[featuresN[i]] = dfTr[features[i]].copy(deep=True)

    if options.train:

        X_train, X_test, y_train, y_test = train_test_split(dfTr[featuresN], dfTr['targetId'], stratify=dfTr['targetId'], test_size=0.2)

        PUmodel = xgb.XGBClassifier(objective=hyperparams['objective'], booster=hyperparams['booster'], eval_metric=hyperparams['eval_metric'],
                                       reg_alpha=hyperparams['reg_alpha'], reg_lambda=hyperparams['reg_lambda'], max_depth=hyperparams['max_depth'],
                                       learning_rate=hyperparams['learning_rate'], subsample=hyperparams['subsample'], colsample_bytree=hyperparams['colsample_bytree'], 
                                       n_estimators=hyperparams['num_trees'], use_label_encoder=False)

        PUmodel.fit(X_train, y_train)

        X_train['bdt_output'] = PUmodel.predict_proba(X_train)[:,1]
        FPRtrain, TPRtrain, THRtrain = metrics.roc_curve(y_train, X_train['bdt_output'])

        X_test['bdt_output'] = PUmodel.predict_proba(X_test)[:,1]
        FPRtest, TPRtest, THRtest = metrics.roc_curve(y_test, X_test['bdt_output'])

        AUCtrain = metrics.roc_auc_score(y_test,X_test['bdt_output'])
        AUCtest = metrics.roc_auc_score(y_train,X_train['bdt_output'])

        save_obj(PUmodel, indir+'/TauBDTIdentifier/PUmodel.pkl')
    else:
        PUmodel = load_obj(indir+'/TauBDTIdentifier/PUmodel.pkl')
        
        X_train, X_test, y_train, y_test = train_test_split(dfTr[featuresN], dfTr['targetId'], stratify=dfTr['targetId'], test_size=0.2)

        X_train['bdt_output'] = PUmodel.predict_proba(X_train)[:,1]
        FPRtrain, TPRtrain, THRtrain = metrics.roc_curve(y_train, X_train['bdt_output'])

        X_test['bdt_output'] = PUmodel.predict_proba(X_test)[:,1]
        FPRtest, TPRtest, THRtest = metrics.roc_curve(y_test, X_test['bdt_output'])

        AUCtrain = metrics.roc_auc_score(y_test,X_test['bdt_output'])
        AUCtest = metrics.roc_auc_score(y_train,X_train['bdt_output'])

    ############################## Model validation ##############################

    dfVal = pd.read_pickle(indir+'/X_Ident_BDT_forEvaluator.pkl')
    dfVal['cl3d_abseta'] = abs(dfVal['cl3d_eta']).copy(deep=True)
    for i in range(len(features)): dfVal[featuresN[i]] = dfVal[features[i]].copy(deep=True)

    # rescale the features if required
    if options.doRescale:
        print('\n** INFO: rescaling features to bound their values')

        # shift features to be shifted
        for feat in features2shift:
            dfVal[feat] = dfVal[feat] - 25

        # saturate features
        for feat in features2saturate:
            dfVal[feat].clip(saturation_dict[feat][0],saturation_dict[feat][1], inplace=True)
            
            # fill the bounds DF
            bounds4features[feat] = np.linspace(saturation_dict[feat][0],saturation_dict[feat][1],100)

        scale_range = [-32,32]
        MMS = MinMaxScaler(scale_range)

        for feat in features2saturate:
            MMS.fit( np.array(bounds4features[feat]).reshape(-1,1) ) # we fit every feature separately otherwise MMS compleins for dimensionality
            dfVal[feat] = MMS.transform( np.array(dfVal[feat]).reshape(-1,1) )

    dfVal['bdt_output'] = PUmodel.predict_proba(dfVal[featuresN])[:,1]
    FPRvalid, TPRvalid, THRvalid = metrics.roc_curve(dfVal['targetId'], dfVal['bdt_output'])
    AUCvalid = metrics.roc_auc_score(dfVal['targetId'], dfVal['bdt_output'])


    ######################### PLOT FEATURES #########################        
    print('\n** INFO: plotting features')

    dfPU = dfTr.query('targetId==0')
    dfTau = dfTr.query('targetId==1')

    os.system('mkdir -p '+indir+'/TauBDTIdentifier_plots/features/')

    for var in features:
        plt.figure(figsize=(10,10))
        plt.hist(dfPU[var], bins=np.arange(features_dict[var][1][0],features_dict[var][1][1],(features_dict[var][1][1]-features_dict[var][1][0])/features_dict[var][1][2]), label='PU events',      color='red',    histtype='step', lw=2, density=True)
        plt.hist(dfTau[var], bins=np.arange(features_dict[var][1][0],features_dict[var][1][1],(features_dict[var][1][1]-features_dict[var][1][0])/features_dict[var][1][2]), label='Tau signal',   color='limegreen',    histtype='step', lw=2, density=True)
        plt.legend(loc = 'upper right')
        plt.grid(linestyle=':')
        plt.xlabel(features_dict[var][0])
        plt.ylabel(r'a.u.')
        mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
        plt.savefig(indir+'/TauBDTIdentifier_plots/features/'+var+'.pdf')
        plt.close()

    del dfPU, dfTau

    ######################### PLOT SINGLE FE OPTION ROCS #########################

    print('\n** INFO: plotting train-test-validation ROC curves')
    plt.figure(figsize=(10,10))
    plt.plot(TPRtrain, FPRtrain, label='Trainining ROC, AUC = %.3f' % (AUCtrain), color='blue',lw=2)
    plt.plot(TPRtest, FPRtest,   label='Testing ROC, AUC = %.3f' % (AUCtest), color='orange',lw=2)
    plt.plot(TPRvalid, FPRvalid, label='Validation ROC, AUC = %.3f' % (AUCvalid), color='green',lw=2)
    plt.grid(linestyle=':')
    plt.legend(loc = 'upper left')
    plt.xlim(0.8,1.01)
    #plt.yscale('log')
    #plt.ylim(0.01,1)
    plt.xlabel('Signal efficiency')
    plt.ylabel('Background efficiency')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(indir+'/TauBDTIdentifier_plots/PUbdt_train_test_validation_rocs.pdf')
    plt.close()


    ######################### PLOT FEATURE IMPORTANCES #########################

    print('\n** INFO: plotting features importance and score')
    # print importance of the features used for training
    feature_importance = PUmodel.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + 0.5
    fig = plt.figure(figsize=(15, 10))
    plt.gcf().subplots_adjust(left=0.25)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, np.array(features)[sorted_idx])
    plt.xlabel(r'Importance score')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(indir+'/TauBDTIdentifier_plots/PUbdt_importances.pdf')


    df = pd.concat([dfTr.query('targetId==1').sample(1500), dfTr.query('targetId==0').sample(1500)], sort=False)[features]
    explainer = shap.Explainer(PUmodel)
    shap_values = explainer(df)

    plt.figure(figsize=(32,16))
    shap.plots.beeswarm(shap_values, max_display=99, show=False)
    plt.gcf().subplots_adjust(left=0.25)
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU', fontsize=16)
    plt.savefig(indir+'/TauBDTIdentifier_plots/PUbdt_SHAPimportance.pdf')
    plt.close()

    most_importants = list(global_shap_importance(shap_values)['features'])[:3]
    for feat in most_importants:
        plt.figure(figsize=(20,20))
        shap.plots.scatter(shap_values[:,feat], color=shap_values, show=False)
        mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU', fontsize=16)
        plt.savefig(indir+'/TauBDTIdentifier_plots/PUbdt_SHAPdependence_'+feat+'.pdf')
        plt.close()
    

    ######################### PLOT BDT SCORE #########################

    dfPU = dfVal.query('targetId==0')
    dfTau = dfVal.query('targetId==1')

    plt.figure(figsize=(10,10))
    plt.hist(dfTau['bdt_output'],  bins=np.arange(-0.0, 1.0, 0.02), color='green', histtype='step', lw=2, label='Tau', density=True)
    plt.hist(dfPU['bdt_output'], bins=np.arange(-0.0, 1.0, 0.02), color='red', histtype='step', lw=2, label='PU', density=True)
    plt.legend(loc = 'upper center')
    plt.xlabel(r'BDT score')
    plt.ylabel(r'a.u.')
    plt.grid(linestyle=':')
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(indir+'/TauBDTIdentifier_plots/PUbdt_score_.pdf')
    plt.close()

    del dfPU, dfTau
