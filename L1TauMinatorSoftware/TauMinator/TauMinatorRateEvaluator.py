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

def deltar2cluster ( df ):
    delta_eta = np.abs(df['L1_eta']-df['L1_eta_ass'])
    delta_phi = np.abs(df['L1_phi']-df['L1_phi_ass'])
    sel = delta_phi > np.pi
    delta_phi = sel*(2*np.pi) - delta_phi
    return np.sqrt( delta_eta**2 + delta_phi**2 )

def applyDRcut ( df, dR ):
    df = df.join(df, on='event', how='left', rsuffix='_ass', sort=False)
    df['deltar2cluster'] = deltar2cluster(df)
    df.query('deltar2cluster>{0}'.format(dR), inplace=True)

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

    os.system('mkdir -p '+perfdir+'/rates')

    X1CNN = np.load(indir+'/TauMinatorRateInputs_'+options.caloClNxM+options.inTag+'/X_CNN_'+options.caloClNxM+'.npz')['arr_0']
    X2CNN = np.load(indir+'/TauMinatorRateInputs_'+options.caloClNxM+options.inTag+'/X_Dense_'+options.caloClNxM+'.npz')['arr_0']
    YCNN  = np.load(indir+'/TauMinatorRateInputs_'+options.caloClNxM+options.inTag+'/Y_'+options.caloClNxM+'.npz')['arr_0']

    XBDT = pd.read_pickle(indir+'/TauMinatorRateInputs_'+options.caloClNxM+options.inTag+'/X_BDT.pkl')
    XBDT['cl3d_abseta'] = abs(XBDT['cl3d_eta']).copy(deep=True)

    ############################## Apply CNN models to inputs ##############################

    dfCNN = pd.DataFrame()
    dfCNN['event']        = YCNN[:,2].ravel().astype(int)
    dfCNN['L1_pt_CLNxM']  = TauCalibratorModel.predict([X1CNN, X2CNN]).ravel()
    dfCNN['L1_eta_CLNxM'] = X2CNN[:,0].ravel()
    dfCNN['L1_phi_CLNxM'] = X2CNN[:,1].ravel()
    dfCNN['CNNscore']     = TauIdentifierModel.predict([X1CNN, X2CNN]).ravel()
    dfCNN['CNNpass99']    = dfCNN['CNNscore'] > CNN_WP_dict['wp99']
    dfCNN['CNNpass95']    = dfCNN['CNNscore'] > CNN_WP_dict['wp95']
    dfCNN['CNNpass90']    = dfCNN['CNNscore'] > CNN_WP_dict['wp90']
    dfCNN.sort_values('event', inplace=True)

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

    dfBDT = XBDT[['event', 'cl3d_pt_c3', 'cl3d_eta', 'cl3d_phi', 'bdt_output']].copy(deep=True)
    dfBDT.rename(columns={'cl3d_pt_c3':'L1_pt_CL3D', 'cl3d_eta':'L1_eta_CL3D', 'cl3d_phi':'L1_phi_CL3D', 'bdt_output':'BDTscore'}, inplace=True)
    dfBDT['BDTpass99']  = dfBDT['BDTscore'] > BDT_WP_dict['wp99']
    dfBDT['BDTpass95']  = dfBDT['BDTscore'] > BDT_WP_dict['wp95']
    dfBDT['BDTpass90']  = dfBDT['BDTscore'] > BDT_WP_dict['wp90']
    dfBDT.sort_values('event', inplace=True)

    dfTOT = pd.concat([dfCNN, dfBDT], axis=0, sort=False)
    events_total = np.unique(dfTOT['event']).shape[0]
    del dfTOT, X1CNN, X2CNN, YCNN, XBDT

    ############################## Match CL3D to CLNxM in the endcap ##############################

    # define the endcap part of the towers >1.3 so that a DeltaEta(CLTW, CL3D)<0.25 can be applied, 0.25 ~ HGCAL_TT_size * 3
    dfCNN_CE = dfCNN[abs(dfCNN['L1_eta_CLNxM'])>1.3]     

    # CL3Ds should have a better efficiency that CLTW in HGCAL -> we give CL3D precedence in the joining
    dfCNN_CE.set_index('event', inplace=True)
    dfBDT.set_index('event', inplace=True)
    dfTauMinated_CE = dfBDT.join(dfCNN_CE, on='event', how='left', rsuffix='_joined', sort=False)

    # drop all cases in which there is no good match between CL3D and CLTW -> corresponds to DeltaR<0.47 in the furthest corner of the CLTW
    dfTauMinated_CE.dropna(axis=0, how='any', inplace=True)
    dfTauMinated_CE['L1_deltaEta'] = abs(dfTauMinated_CE['L1_eta_CL3D'] - dfTauMinated_CE['L1_eta_CLNxM'])
    dfTauMinated_CE['L1_deltaPhi'] = abs(dfTauMinated_CE['L1_phi_CL3D'] - dfTauMinated_CE['L1_phi_CLNxM'])
    dfTauMinated_CE = dfTauMinated_CE[((dfTauMinated_CE['L1_deltaEta'] < 0.25) & (dfTauMinated_CE['L1_deltaPhi'] < 0.4))] # | (dfTauMinated_CE['L1_pt_CL3D']>75) | (dfTauMinated_CE['L1_pt_CLNxM']>75)]
    dfTauMinated_CE.drop(['L1_deltaEta', 'L1_deltaPhi'], axis=1, inplace=True)

    ############################## Put togetehr the everything   again ##############################

    dfCNN.set_index(['event'], inplace=True)
    dfTauMinated = pd.concat([dfTauMinated_CE, dfCNN], axis=0, sort=False)
    
    # if L1_pt_CL3D keep that as the L1 candidate for the tau
    dfTauMinated.sort_values('L1_pt_CL3D', inplace=True)
    dfTauMinated.reset_index(inplace=True)
    dfTauMinated.drop_duplicates(['event', 'L1_pt_CLNxM', 'L1_eta_CLNxM', 'L1_phi_CLNxM'], keep='first', inplace=True)

    # saparate the rightful HGCAL candidates from the barrel candidates
    dfTauMinated.fillna(-99.9, inplace=True) # this fills the NaN of all the CLTW wothout a CL3D match
    dfTauMinated_CB = dfTauMinated[dfTauMinated['L1_pt_CL3D']==-99.9][['event',
                                                                       'L1_pt_CLNxM', 'L1_eta_CLNxM', 'L1_phi_CLNxM',
                                                                       'CNNpass99', 'CNNpass95', 'CNNpass90']]
    
    dfTauMinated_CE = dfTauMinated[dfTauMinated['L1_pt_CL3D']!=-99.9][['event',
                                                                       'L1_pt_CL3D', 'L1_eta_CL3D', 'L1_phi_CL3D',
                                                                       'BDTpass99', 'BDTpass95', 'BDTpass90', 'CNNpass99', 'CNNpass95', 'CNNpass90']]

    # do some renaiming
    dfTauMinated_CE['TauMinated99'] = (dfTauMinated_CE['BDTpass99'] == True) & (dfTauMinated_CE['CNNpass99'] == True)
    dfTauMinated_CE['TauMinated95'] = (dfTauMinated_CE['BDTpass95'] == True) & (dfTauMinated_CE['CNNpass95'] == True)
    dfTauMinated_CE['TauMinated90'] = (dfTauMinated_CE['BDTpass90'] == True) & (dfTauMinated_CE['CNNpass90'] == True)
    dfTauMinated_CE.drop(['BDTpass99', 'BDTpass95', 'BDTpass90', 'CNNpass99', 'CNNpass95', 'CNNpass90'], axis=1, inplace=True)
    dfTauMinated_CE.rename(columns={'L1_pt_CL3D':'L1_pt' , 'L1_eta_CL3D':'L1_eta' , 'L1_phi_CL3D':'L1_phi'}, inplace=True)
    dfTauMinated_CB.rename(columns={'L1_pt_CLNxM':'L1_pt' , 'L1_eta_CLNxM':'L1_eta' , 'L1_phi_CLNxM':'L1_phi' , 'CNNpass99':'TauMinated99' , 'CNNpass95':'TauMinated95' , 'CNNpass90':'TauMinated90'}, inplace=True)

    # put back everything together
    dfTauMinated = pd.concat([dfTauMinated_CB, dfTauMinated_CE], axis=0, sort=False)
    dfTauMinated.sort_values('L1_pt', inplace=True)
    dfTauMinated.drop_duplicates(['event', 'L1_pt', 'L1_eta', 'L1_phi'], keep='first', inplace=True)

    del dfTauMinated_CB, dfTauMinated_CE, dfCNN, dfBDT, dfCNN_CE


    ############################## Calculate rate ##############################

    events_frequency=2808*11.2  # N_bunches * frequency [kHz] --> from: https://cds.cern.ch/record/2130736/files/Introduction%20to%20the%20HL-LHC%20Project.pdf
    mapping_dict = load_obj(perfdir+'/online2offline_mapping.pkl')
    online_thresholds = range(20, 175, 1)
    rates_online = {'singleTau' : {}, 'doubleTau' : {}}
    rates_offline = {'singleTau' : {}, 'doubleTau' : {}}

    for WP in ['99', '95', '90']:
        tmp = dfTauMinated[(dfTauMinated['L1_pt']>32) & (dfTauMinated['TauMinated'+WP])]

        tmp['L1_wp'+WP+'_pt95'] = tmp['L1_pt'].apply(lambda x : np.interp(x, online_thresholds, mapping_dict['wp'+WP+'_pt95']))
        tmp['L1_wp'+WP+'_pt90'] = tmp['L1_pt'].apply(lambda x : np.interp(x, online_thresholds, mapping_dict['wp'+WP+'_pt90']))
        tmp['L1_wp'+WP+'_pt50'] = tmp['L1_pt'].apply(lambda x : np.interp(x, online_thresholds, mapping_dict['wp'+WP+'_pt50']))

        #********** single tau rate **********#
        sel_events = np.unique(tmp['event']).shape[0]
        print('             singleTau: selected_clusters / total_clusters = {0} / {1} = {2} '.format(sel_events, events_total, float(sel_events)/float(events_total)))

        tmp.sort_values('L1_pt', inplace=True)
        tmp['COTcount_singleTau'] = np.linspace(tmp.shape[0]-1, 0, tmp.shape[0])

        rates_online['singleTau']['wp'+WP] = ( np.sort(tmp['L1_pt']), np.array(tmp['COTcount_singleTau'])/events_total*events_frequency, np.array(tmp['COTcount_singleTau']) )
        rates_offline['singleTau']['wp'+WP+'_pt95'] = ( np.sort(tmp['L1_wp'+WP+'_pt95']), np.array(tmp['COTcount_singleTau'])/events_total*events_frequency, np.array(tmp['COTcount_singleTau']) )
        rates_offline['singleTau']['wp'+WP+'_pt90'] = ( np.sort(tmp['L1_wp'+WP+'_pt90']), np.array(tmp['COTcount_singleTau'])/events_total*events_frequency, np.array(tmp['COTcount_singleTau']) )
        rates_offline['singleTau']['wp'+WP+'_pt50'] = ( np.sort(tmp['L1_wp'+WP+'_pt50']), np.array(tmp['COTcount_singleTau'])/events_total*events_frequency, np.array(tmp['COTcount_singleTau']) )

        #********** double tau rate **********#
        tmp['doubleTau'] = tmp.duplicated('event', keep=False)
        tmp.query('doubleTau==True', inplace=True)
        applyDRcut(tmp, 0.5)
        sel_events = np.unique(tmp['event']).shape[0]
        print('             doubleTau: selected_clusters / total_clusters = {0} / {1} = {2} '.format(sel_events, events_total, float(sel_events)/float(events_total)))

        tmp.sort_values('L1_pt', inplace=True)
        tmp['COTcount_doubleTau'] = np.linspace(tmp.shape[0]-1, 0, tmp.shape[0])

        rates_online['doubleTau']['wp'+WP] = ( np.sort(tmp['L1_pt']), np.array(tmp['COTcount_doubleTau'])/events_total*events_frequency, np.array(tmp['COTcount_doubleTau']) )
        rates_offline['doubleTau']['wp'+WP+'_pt95'] = ( np.sort(tmp['L1_wp'+WP+'_pt95']), np.array(tmp['COTcount_doubleTau'])/events_total*events_frequency, np.array(tmp['COTcount_doubleTau']) )
        rates_offline['doubleTau']['wp'+WP+'_pt90'] = ( np.sort(tmp['L1_wp'+WP+'_pt90']), np.array(tmp['COTcount_doubleTau'])/events_total*events_frequency, np.array(tmp['COTcount_doubleTau']) )
        rates_offline['doubleTau']['wp'+WP+'_pt50'] = ( np.sort(tmp['L1_wp'+WP+'_pt50']), np.array(tmp['COTcount_doubleTau'])/events_total*events_frequency, np.array(tmp['COTcount_doubleTau']) )

        del tmp

    for WP in ['99', '95', '90']:
        print('    - WP = '+WP)

        plt.figure(figsize=(10,10))
        plt.plot(rates_offline['singleTau']['wp'+WP+'_pt95'][0], rates_offline['singleTau']['wp'+WP+'_pt95'][1], linewidth=2, color='blue', label='Offline threshold @ 95%')
        plt.plot(rates_offline['singleTau']['wp'+WP+'_pt90'][0], rates_offline['singleTau']['wp'+WP+'_pt90'][1], linewidth=2, color='red', label='Offline threshold @ 90%')
        plt.plot(rates_offline['singleTau']['wp'+WP+'_pt50'][0], rates_offline['singleTau']['wp'+WP+'_pt50'][1], linewidth=2, color='green', label='Offline threshold @ 50%')
        plt.yscale("log")
        plt.ylim(bottom=1)
        legend = plt.legend(loc = 'upper right', fontsize=16)
        plt.grid(linestyle=':')
        plt.xlabel('Offline threshold [GeV]')
        plt.ylabel('Rate [kHz]')
        mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
        plt.savefig(perfdir+'/rates/rate_singleTau_offline_wp'+WP+'.pdf')
        plt.close()

        print('        - single tau offline threshold for 25kHz rate @ 95% efficiency = {0}'.format( round(np.interp(25, np.flip(rates_offline['singleTau']['wp'+WP+'_pt95'][1]), np.flip(rates_offline['singleTau']['wp'+WP+'_pt95'][0]), left=0.0, right=0.0),0) ))
        print('        - single tau offline threshold for 25kHz rate @ 90% efficiency = {0}'.format( round(np.interp(25, np.flip(rates_offline['singleTau']['wp'+WP+'_pt90'][1]), np.flip(rates_offline['singleTau']['wp'+WP+'_pt90'][0]), left=0.0, right=0.0),0) ))
        print('        - single tau offline threshold for 25kHz rate @ 50% efficiency = {0}'.format( round(np.interp(25, np.flip(rates_offline['singleTau']['wp'+WP+'_pt50'][1]), np.flip(rates_offline['singleTau']['wp'+WP+'_pt50'][0]), left=0.0, right=0.0),0) ))

        plt.figure(figsize=(10,10))
        plt.plot(rates_offline['doubleTau']['wp'+WP+'_pt95'][0], rates_offline['doubleTau']['wp'+WP+'_pt95'][1], linewidth=2, color='blue', label='Offline threshold @ 95%')
        plt.plot(rates_offline['doubleTau']['wp'+WP+'_pt90'][0], rates_offline['doubleTau']['wp'+WP+'_pt90'][1], linewidth=2, color='red', label='Offline threshold @ 90%')
        plt.plot(rates_offline['doubleTau']['wp'+WP+'_pt50'][0], rates_offline['doubleTau']['wp'+WP+'_pt50'][1], linewidth=2, color='green', label='Offline threshold @ 50%')
        plt.yscale("log")
        plt.ylim(bottom=1)
        legend = plt.legend(loc = 'upper right', fontsize=16)
        plt.grid(linestyle=':')
        plt.xlabel('Offline threshold [GeV]')
        plt.ylabel('Rate [kHz]')
        mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
        plt.savefig(perfdir+'/rates/rate_doubleTau_offline_wp'+WP+'.pdf')
        plt.close()

        print('        - double tau offline threshold for 25kHz rate @ 95% efficiency = {0},{0}'.format( round(np.interp(25, np.flip(rates_offline['doubleTau']['wp'+WP+'_pt95'][1]), np.flip(rates_offline['doubleTau']['wp'+WP+'_pt95'][0]), left=0.0, right=0.0),0) ))
        print('        - double tau offline threshold for 25kHz rate @ 90% efficiency = {0},{0}'.format( round(np.interp(25, np.flip(rates_offline['doubleTau']['wp'+WP+'_pt90'][1]), np.flip(rates_offline['doubleTau']['wp'+WP+'_pt90'][0]), left=0.0, right=0.0),0) ))
        print('        - double tau offline threshold for 25kHz rate @ 50% efficiency = {0},{0}'.format( round(np.interp(25, np.flip(rates_offline['doubleTau']['wp'+WP+'_pt50'][1]), np.flip(rates_offline['doubleTau']['wp'+WP+'_pt50'][0]), left=0.0, right=0.0),0) ))





