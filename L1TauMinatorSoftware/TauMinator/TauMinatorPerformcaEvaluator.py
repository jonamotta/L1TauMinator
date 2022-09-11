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
    sel = g[(g['TauMinated'+WP]==1) & (g['L1_pt']>thr)].shape[0]
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

    os.system('mkdir -p '+perfdir+'/plots')

    X1CNN = np.load(indir+'/TauMinatorInputs_'+options.caloClNxM+options.inTag+'/X_CNN_'+options.caloClNxM+'.npz')['arr_0']
    X2CNN = np.load(indir+'/TauMinatorInputs_'+options.caloClNxM+options.inTag+'/X_Dense_'+options.caloClNxM+'.npz')['arr_0']
    YCNN  = np.load(indir+'/TauMinatorInputs_'+options.caloClNxM+options.inTag+'/Y_'+options.caloClNxM+'.npz')['arr_0']

    XBDT = pd.read_pickle(indir+'/TauMinatorInputs_'+options.caloClNxM+options.inTag+'/X_BDT.pkl')
    XBDT['cl3d_abseta'] = abs(XBDT['cl3d_eta']).copy(deep=True)

    ############################## Apply CNN models to inputs ##############################

    dfCNN = pd.DataFrame()
    dfCNN['event']        = YCNN[:,4].ravel().astype(int)
    dfCNN['tau_visPt' ]   = YCNN[:,0].ravel()
    dfCNN['tau_visEta']   = YCNN[:,1].ravel()
    dfCNN['tau_visPhi']   = YCNN[:,2].ravel()
    dfCNN['tau_DM']       = YCNN[:,3].ravel().astype(int)
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

    dfBDT = XBDT[['event', 'tau_visPt', 'tau_visEta', 'tau_visPhi', 'tau_DM', 'cl3d_pt_c3', 'cl3d_eta', 'cl3d_phi', 'bdt_output']].copy(deep=True)
    dfBDT.rename(columns={'cl3d_pt_c3':'L1_pt_CL3D', 'cl3d_eta':'L1_eta_CL3D', 'cl3d_phi':'L1_phi_CL3D', 'bdt_output':'BDTscore'}, inplace=True)
    dfBDT['BDTpass99']  = dfBDT['BDTscore'] > BDT_WP_dict['wp99']
    dfBDT['BDTpass95']  = dfBDT['BDTscore'] > BDT_WP_dict['wp95']
    dfBDT['BDTpass90']  = dfBDT['BDTscore'] > BDT_WP_dict['wp90']
    dfBDT.sort_values('event', inplace=True)

    dfTOT = pd.concat([dfCNN, dfBDT], axis=0, sort=False)[['event', 'tau_visPt', 'tau_visEta', 'tau_visPhi', 'tau_DM']]
    dfTOT.drop_duplicates(['event', 'tau_visPt', 'tau_visEta', 'tau_visPhi', 'tau_DM'], keep='first', inplace=True)

    ############################## Match CL3D to CLNxM in the endcap ##############################

    # define the endcap part of the towers >1.3 so that a DeltaEta(CLTW, CL3D)<0.25 can be applied, 0.25 ~ HGCAL_TT_size * 3
    dfCNN_CE = dfCNN[abs(dfCNN['tau_visEta'])>1.3]     

    # CL3Ds should have a better efficiency that CLTW in HGCAL -> we give CL3D precedence in the joining
    dfCNN_CE.set_index(['event', 'tau_visPt', 'tau_visEta', 'tau_visPhi', 'tau_DM'], inplace=True)
    dfBDT.set_index(['event', 'tau_visPt', 'tau_visEta', 'tau_visPhi', 'tau_DM'], inplace=True)
    dfTauMinated_CE = dfBDT.join(dfCNN_CE, on=['event', 'tau_visPt', 'tau_visEta', 'tau_visPhi', 'tau_DM'], how='left', rsuffix='_joined', sort=False)

    # drop all cases in which there is no good match between CL3D and CLTW -> corresponds to DeltaR<0.47 in the furthest corner of the CLTW
    dfTauMinated_CE.dropna(axis=0, how='any', inplace=True)
    dfTauMinated_CE['L1_deltaEta'] = abs(dfTauMinated_CE['L1_eta_CL3D'] - dfTauMinated_CE['L1_eta_CLNxM'])
    dfTauMinated_CE['L1_deltaPhi'] = abs(dfTauMinated_CE['L1_phi_CL3D'] - dfTauMinated_CE['L1_phi_CLNxM'])
    dfTauMinated_CE = dfTauMinated_CE[((dfTauMinated_CE['L1_deltaEta'] < 0.25) & (dfTauMinated_CE['L1_deltaPhi'] < 0.4))] # | (dfTauMinated_CE['L1_pt_CL3D']>75) | (dfTauMinated_CE['L1_pt_CLNxM']>75)]
    dfTauMinated_CE.drop(['L1_deltaEta', 'L1_deltaPhi'], axis=1, inplace=True)

    ############################## Put togetehr the everything again ##############################

    dfCNN.set_index(['event', 'tau_visPt', 'tau_visEta', 'tau_visPhi', 'tau_DM'], inplace=True)
    dfTauMinated = pd.concat([dfTauMinated_CE, dfCNN], axis=0, sort=False)
    
    # if L1_pt_CL3D keep that as the L1 candidate for the tau
    dfTauMinated.sort_values('L1_pt_CL3D', inplace=True)
    dfTauMinated.reset_index(inplace=True)
    dfTauMinated.drop_duplicates(['event', 'tau_visPt', 'tau_visEta', 'tau_visPhi', 'tau_DM'], keep='first', inplace=True)

    # saparate the rightful HGCAL candidates from the barrel candidates
    dfTauMinated.fillna(-99.9, inplace=True) # this fills the NaN of all the CLTW wothout a CL3D match
    dfTauMinated_CB = dfTauMinated[dfTauMinated['L1_pt_CL3D']==-99.9][['event', 'tau_visPt', 'tau_visEta', 'tau_visPhi', 'tau_DM',
                                                                       'L1_pt_CLNxM', 'L1_eta_CLNxM', 'L1_phi_CLNxM',
                                                                       'CNNpass99', 'CNNpass95', 'CNNpass90']]
    
    dfTauMinated_CE = dfTauMinated[dfTauMinated['L1_pt_CL3D']!=-99.9][['event', 'tau_visPt', 'tau_visEta', 'tau_visPhi', 'tau_DM',
                                                                       'L1_pt_CL3D', 'L1_eta_CL3D', 'L1_phi_CL3D',
                                                                       'BDTpass99', 'BDTpass95', 'BDTpass90', 'CNNpass99', 'CNNpass95', 'CNNpass90']]

    # do some renaiming
    dfTauMinated_CE['TauMinated99'] = (dfTauMinated_CE['BDTpass99'] == True) & (dfTauMinated_CE['CNNpass99'] == True)
    dfTauMinated_CE['TauMinated95'] = (dfTauMinated_CE['BDTpass95'] == True) & (dfTauMinated_CE['CNNpass95'] == True)
    dfTauMinated_CE['TauMinated90'] = (dfTauMinated_CE['BDTpass90'] == True) & (dfTauMinated_CE['CNNpass90'] == True)
    dfTauMinated_CE.drop(['BDTpass99', 'BDTpass95', 'BDTpass90', 'CNNpass99', 'CNNpass95', 'CNNpass90'], axis=1, inplace=True)
    dfTauMinated_CE.rename(columns={'L1_pt_CL3D':'L1_pt' , 'L1_eta_CL3D':'L1_eta' , 'L1_phi_CL3D':'L1_phi'}, inplace=True)
    dfTauMinated_CB.rename(columns={'L1_pt_CLNxM':'L1_pt' , 'L1_eta_CLNxM':'L1_eta' , 'L1_phi_CLNxM':'L1_phi' , 'CNNpass99':'TauMinated99' , 'CNNpass95':'TauMinated95' , 'CNNpass90':'TauMinated90'}, inplace=True)

    # put back everything toegtehr, also the taus that are not match to any L1 object 
    dfTauMinated = pd.concat([dfTauMinated_CB, dfTauMinated_CE, dfTOT], axis=0, sort=False)
    dfTauMinated.sort_values('L1_pt', inplace=True)
    dfTauMinated.drop_duplicates(['event', 'tau_visPt', 'tau_visEta', 'tau_visPhi', 'tau_DM'], keep='first', inplace=True)

    # geometrically missed tau to be investigated
    dfTauMissed = dfTauMinated[dfTauMinated['L1_pt'].isna()]
    # dfTauMinated.dropna(axis=0, how='any', inplace=True)

    del dfTOT, dfTauMinated_CB, dfTauMinated_CE, dfCNN, dfBDT, dfCNN_CE, X1CNN, X2CNN, YCNN, XBDT

    ############################## Calculate efficiencies ##############################

    dfTauMinated['tau_visPt_bin'] = pd.cut(dfTauMinated['tau_visPt'],
                              bins=[15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 200, 500],
                              labels=False,
                              include_lowest=True)

    dfTauMinated['tau_visEta_bin'] = pd.cut(dfTauMinated['tau_visEta'],
                              bins=[-3.5, -3.0, -2.7, -2.4, -2.1, -1.8, -1.5, -1.305, -1.0, -0.66, -0.33, 0.0, 0.33, 0.66, 1.0, 1.305, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0, 3.5],
                              labels=False,
                              include_lowest=True)

    dfTauMinated_dm0  = dfTauMinated[dfTauMinated['tau_DM'] == 0].copy(deep=True)
    dfTauMinated_dm1  = dfTauMinated[(dfTauMinated['tau_DM'] == 1) | (dfTauMinated['tau_DM'] == 2)].copy(deep=True)
    dfTauMinated_dm10 = dfTauMinated[dfTauMinated['tau_DM'] == 10].copy(deep=True)
    dfTauMinated_dm11 = dfTauMinated[(dfTauMinated['tau_DM'] == 11) | (dfTauMinated['tau_DM'] == 12)].copy(deep=True)

    cmap = matplotlib.cm.get_cmap('tab20c'); i=0
    pt_bins_centers = [17.5, 22.5, 27.5, 32.5, 37.5, 42.5, 47.5, 52.5, 57.5, 62.5, 67.5, 72.5, 77.5, 82.5, 87.5, 92.5, 97.5, 102.5, 107.5, 112.5, 117.5, 122.5, 127.5, 132.5, 137.5, 142.5, 147.5, 175, 350]
    eta_bins_centers = [-3.25, -2.85, -2.55, -2.25, -1.95, -1.65, -1.4025, -1.1525, -0.825, -0.495, -0.165, 0.165, 0.495, 0.825, 1.1525, 1.4025, 1.65, 1.95, 2.25, 2.55, 2.85, 3.25]
    online_thresholds = range(20, 175, 1)
    plotting_thresholds = range(20, 110, 10)
    turnons_dm_dict = {}
    turnons_dict = {}
    etaeffs_dict = {}
    mapping_dict = {'threshold':[], 'wp99_pt95':[], 'wp99_pt90':[], 'wp99_pt50':[],
                                    'wp95_pt95':[], 'wp95_pt90':[], 'wp95_pt50':[],
                                    'wp90_pt95':[], 'wp90_pt90':[], 'wp90_pt50':[]}
    offline_pts = dfTauMinated.groupby('tau_visPt_bin').mean()['tau_visPt']

    for thr in online_thresholds:
        print(' ** INFO : calculating turnons and mappings for threshold '+str(thr)+' GeV')

        # TURNONS
        grouped = dfTauMinated.groupby('tau_visPt_bin').apply(lambda g: efficiency(g, '99', thr))
        turnon = []
        for ton in grouped: turnon.append([ton[0], ton[1], ton[2]])
        turnons_dict['turnonAt99wpAt'+str(thr)+'GeV'] = np.array(turnon)

        grouped = dfTauMinated.groupby('tau_visPt_bin').apply(lambda g: efficiency(g, '95', thr))
        turnon = []
        for ton in grouped: turnon.append([ton[0], ton[1], ton[2]])
        turnons_dict['turnonAt95wpAt'+str(thr)+'GeV'] = np.array(turnon)

        grouped = dfTauMinated.groupby('tau_visPt_bin').apply(lambda g: efficiency(g, '90', thr))
        turnon = []
        for ton in grouped: turnon.append([ton[0], ton[1], ton[2]])
        turnons_dict['turnonAt90wpAt'+str(thr)+'GeV'] = np.array(turnon)

        # ONLINE TO OFFILNE MAPPING
        mapping_dict['wp99_pt95'].append(np.interp(0.95, turnons_dict['turnonAt99wpAt'+str(thr)+'GeV'][:,0], offline_pts)) #,right=-99,left=-98)
        mapping_dict['wp99_pt90'].append(np.interp(0.90, turnons_dict['turnonAt99wpAt'+str(thr)+'GeV'][:,0], offline_pts)) #,right=-99,left=-98)
        mapping_dict['wp99_pt50'].append(np.interp(0.50, turnons_dict['turnonAt99wpAt'+str(thr)+'GeV'][:,0], offline_pts)) #,right=-99,left=-98)

        mapping_dict['wp95_pt95'].append(np.interp(0.95, turnons_dict['turnonAt95wpAt'+str(thr)+'GeV'][:,0], offline_pts)) #,right=-99,left=-98)
        mapping_dict['wp95_pt90'].append(np.interp(0.90, turnons_dict['turnonAt95wpAt'+str(thr)+'GeV'][:,0], offline_pts)) #,right=-99,left=-98)
        mapping_dict['wp95_pt50'].append(np.interp(0.50, turnons_dict['turnonAt95wpAt'+str(thr)+'GeV'][:,0], offline_pts)) #,right=-99,left=-98)

        mapping_dict['wp90_pt95'].append(np.interp(0.95, turnons_dict['turnonAt90wpAt'+str(thr)+'GeV'][:,0], offline_pts)) #,right=-99,left=-98)
        mapping_dict['wp90_pt90'].append(np.interp(0.90, turnons_dict['turnonAt90wpAt'+str(thr)+'GeV'][:,0], offline_pts)) #,right=-99,left=-98)
        mapping_dict['wp90_pt50'].append(np.interp(0.50, turnons_dict['turnonAt90wpAt'+str(thr)+'GeV'][:,0], offline_pts)) #,right=-99,left=-98)

        # TURNONS DM 0
        grouped = dfTauMinated_dm0.groupby('tau_visPt_bin').apply(lambda g: efficiency(g, '99', thr))
        turnon = []
        for ton in grouped: turnon.append([ton[0], ton[1], ton[2]])
        turnons_dm_dict['dm0TurnonAt99wpAt'+str(thr)+'GeV'] = np.array(turnon)

        grouped = dfTauMinated_dm0.groupby('tau_visPt_bin').apply(lambda g: efficiency(g, '95', thr))
        turnon = []
        for ton in grouped: turnon.append([ton[0], ton[1], ton[2]])
        turnons_dm_dict['dm0TurnonAt95wpAt'+str(thr)+'GeV'] = np.array(turnon)

        grouped = dfTauMinated_dm0.groupby('tau_visPt_bin').apply(lambda g: efficiency(g, '90', thr))
        turnon = []
        for ton in grouped: turnon.append([ton[0], ton[1], ton[2]])
        turnons_dm_dict['dm0TurnonAt90wpAt'+str(thr)+'GeV'] = np.array(turnon)

        # TURNONS DM 1
        grouped = dfTauMinated_dm1.groupby('tau_visPt_bin').apply(lambda g: efficiency(g, '99', thr))
        turnon = []
        for ton in grouped: turnon.append([ton[0], ton[1], ton[2]])
        turnons_dm_dict['dm1TurnonAt99wpAt'+str(thr)+'GeV'] = np.array(turnon)

        grouped = dfTauMinated_dm1.groupby('tau_visPt_bin').apply(lambda g: efficiency(g, '95', thr))
        turnon = []
        for ton in grouped: turnon.append([ton[0], ton[1], ton[2]])
        turnons_dm_dict['dm1TurnonAt95wpAt'+str(thr)+'GeV'] = np.array(turnon)

        grouped = dfTauMinated_dm1.groupby('tau_visPt_bin').apply(lambda g: efficiency(g, '90', thr))
        turnon = []
        for ton in grouped: turnon.append([ton[0], ton[1], ton[2]])
        turnons_dm_dict['dm1TurnonAt90wpAt'+str(thr)+'GeV'] = np.array(turnon)

        # TURNONS DM 10
        grouped = dfTauMinated_dm10.groupby('tau_visPt_bin').apply(lambda g: efficiency(g, '99', thr))
        turnon = []
        for ton in grouped: turnon.append([ton[0], ton[1], ton[2]])
        turnons_dm_dict['dm10TurnonAt99wpAt'+str(thr)+'GeV'] = np.array(turnon)

        grouped = dfTauMinated_dm10.groupby('tau_visPt_bin').apply(lambda g: efficiency(g, '95', thr))
        turnon = []
        for ton in grouped: turnon.append([ton[0], ton[1], ton[2]])
        turnons_dm_dict['dm10TurnonAt95wpAt'+str(thr)+'GeV'] = np.array(turnon)

        grouped = dfTauMinated_dm10.groupby('tau_visPt_bin').apply(lambda g: efficiency(g, '90', thr))
        turnon = []
        for ton in grouped: turnon.append([ton[0], ton[1], ton[2]])
        turnons_dm_dict['dm10TurnonAt90wpAt'+str(thr)+'GeV'] = np.array(turnon)

        # TURNONS DM 11
        grouped = dfTauMinated_dm11.groupby('tau_visPt_bin').apply(lambda g: efficiency(g, '99', thr))
        turnon = []
        for ton in grouped: turnon.append([ton[0], ton[1], ton[2]])
        turnons_dm_dict['dm11TurnonAt99wpAt'+str(thr)+'GeV'] = np.array(turnon)

        grouped = dfTauMinated_dm11.groupby('tau_visPt_bin').apply(lambda g: efficiency(g, '95', thr))
        turnon = []
        for ton in grouped: turnon.append([ton[0], ton[1], ton[2]])
        turnons_dm_dict['dm11TurnonAt95wpAt'+str(thr)+'GeV'] = np.array(turnon)

        grouped = dfTauMinated_dm11.groupby('tau_visPt_bin').apply(lambda g: efficiency(g, '90', thr))
        turnon = []
        for ton in grouped: turnon.append([ton[0], ton[1], ton[2]])
        turnons_dm_dict['dm11TurnonAt90wpAt'+str(thr)+'GeV'] = np.array(turnon)

        # ETA EFFICIENCIES
        # grouped = dfTauMinated.groupby('tau_visEta_bin').apply(lambda g: efficiency(g, '99', thr))
        # efficiency = []
        # for eff in grouped: efficiency.append([eff[0], eff[1], eff[2]])
        # etaeffs_dict['efficiencyVsEtaAt99wpAt'+str(thr)+'GeV'] = np.array(efficiency)

        # grouped = dfTauMinated.groupby('tau_visEta_bin').apply(lambda g: efficiency(g, '95', thr))
        # efficiency = []
        # for eff in grouped: efficiency.append([eff[0], eff[1], eff[2]])
        # etaeffs_dict['efficiencyVsEtaAt95wpAt'+str(thr)+'GeV'] = np.array(efficiency)

        # grouped = dfTauMinated.groupby('tau_visEta_bin').apply(lambda g: efficiency(g, '90', thr))
        # efficiency = []
        # for eff in grouped: efficiency.append([eff[0], eff[1], eff[2]])
        # etaeffs_dict['efficiencyVsEtaAt90wpAt'+str(thr)+'GeV'] = np.array(efficiency)

    save_obj(mapping_dict, perfdir+'/online2offline_mapping.pkl')


    ##################################################################################
    # PLOT TURNONS
    i = 0
    plt.figure(figsize=(10,10))
    for thr in plotting_thresholds:
        if not thr%10:
            plt.errorbar(pt_bins_centers,turnons_dict['turnonAt99wpAt'+str(thr)+'GeV'][:,0],xerr=1,yerr=[turnons_dict['turnonAt99wpAt'+str(thr)+'GeV'][:,1], turnons_dict['turnonAt99wpAt'+str(thr)+'GeV'][:,2]], ls='None', label=r'$p_{T}^{L1 \tau} > %i$ GeV' % (thr), lw=2, marker='o', color=cmap(i))

            p0 = [1, thr, 1] 
            popt, pcov = curve_fit(sigmoid, pt_bins_centers, turnons_dict['turnonAt99wpAt'+str(thr)+'GeV'][:,0], p0, maxfev=5000)
            plt.plot(pt_bins_centers, sigmoid(pt_bins_centers, *popt), '-', label='_', lw=1.5, color=cmap(i))

            i+=1 
    plt.hlines(0.90, 0, 2000, lw=2, color='dimgray', label='0.90 Eff.')
    plt.hlines(0.95, 0, 2000, lw=2, color='black', label='0.95 Eff.')
    plt.legend(loc = 'lower right', fontsize=14)
    plt.ylim(0., 1.05)
    plt.xlim(0., 150.)
    # plt.xscale('log')
    plt.xlabel(r'$p_{T}^{gen,\tau}\ [GeV]$')
    plt.ylabel(r'Efficiency')
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(perfdir+'/plots/turnons_WP99.pdf')
    plt.close()

    i = 0
    plt.figure(figsize=(10,10))
    for thr in plotting_thresholds:
        if not thr%10:
            plt.errorbar(pt_bins_centers,turnons_dict['turnonAt95wpAt'+str(thr)+'GeV'][:,0],xerr=1,yerr=[turnons_dict['turnonAt95wpAt'+str(thr)+'GeV'][:,1], turnons_dict['turnonAt95wpAt'+str(thr)+'GeV'][:,2]], ls='None', label=r'$p_{T}^{L1 \tau} > %i$ GeV' % (thr), lw=2, marker='o', color=cmap(i))

            p0 = [1, thr, 1] 
            popt, pcov = curve_fit(sigmoid, pt_bins_centers, turnons_dict['turnonAt95wpAt'+str(thr)+'GeV'][:,0], p0, maxfev=5000)
            plt.plot(pt_bins_centers, sigmoid(pt_bins_centers, *popt), '-', label='_', lw=1.5, color=cmap(i))

            i+=1 
    plt.hlines(0.90, 0, 2000, lw=2, color='dimgray', label='0.90 Eff.')
    plt.hlines(0.95, 0, 2000, lw=2, color='black', label='0.95 Eff.')
    plt.legend(loc = 'lower right', fontsize=14)
    plt.ylim(0., 1.05)
    plt.xlim(0., 150.)
    # plt.xscale('log')
    plt.xlabel(r'$p_{T}^{gen,\tau}\ [GeV]$')
    plt.ylabel(r'Efficiency')
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(perfdir+'/plots/turnons_WP95.pdf')
    plt.close()

    i = 0
    plt.figure(figsize=(10,10))
    for thr in plotting_thresholds:
        if not thr%10:
            plt.errorbar(pt_bins_centers,turnons_dict['turnonAt90wpAt'+str(thr)+'GeV'][:,0],xerr=1,yerr=[turnons_dict['turnonAt90wpAt'+str(thr)+'GeV'][:,1], turnons_dict['turnonAt90wpAt'+str(thr)+'GeV'][:,2]], ls='None', label=r'$p_{T}^{L1 \tau} > %i$ GeV' % (thr), lw=2, marker='o', color=cmap(i))

            p0 = [1, thr, 1] 
            popt, pcov = curve_fit(sigmoid, pt_bins_centers, turnons_dict['turnonAt90wpAt'+str(thr)+'GeV'][:,0], p0, maxfev=5000)
            plt.plot(pt_bins_centers, sigmoid(pt_bins_centers, *popt), '-', label='_', lw=1.5, color=cmap(i))

            i+=1 
    plt.hlines(0.90, 0, 2000, lw=2, color='dimgray', label='0.90 Eff.')
    plt.hlines(0.95, 0, 2000, lw=2, color='black', label='0.95 Eff.')
    plt.legend(loc = 'lower right', fontsize=14)
    plt.ylim(0., 1.05)
    plt.xlim(0., 150.)
    # plt.xscale('log')
    plt.xlabel(r'$p_{T}^{gen,\tau}\ [GeV]$')
    plt.ylabel(r'Efficiency')
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(perfdir+'/plots/turnons_WP90.pdf')
    plt.close()


    ##################################################################################
    # PLOT ONLINE TO OFFLINE MAPPING
    plt.figure(figsize=(10,10))
    plt.plot(online_thresholds, mapping_dict['wp99_pt95'], label='@ 95% efficiency', linewidth=2, color='blue')
    plt.plot(online_thresholds, mapping_dict['wp99_pt90'], label='@ 90% efficiency', linewidth=2, color='red')
    plt.plot(online_thresholds, mapping_dict['wp99_pt50'], label='@ 50% efficiency', linewidth=2, color='green')
    plt.legend(loc = 'lower right', fontsize=14)
    plt.xlabel('L1 Threshold [GeV]')
    plt.ylabel('Offline threshold [GeV]')
    plt.xlim(0, 110)
    # plt.ylim(0, 200)
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(perfdir+'/plots/online2offline_WP99.pdf')
    plt.close()

    plt.figure(figsize=(10,10))
    plt.plot(online_thresholds, mapping_dict['wp95_pt95'], label='@ 95% efficiency', linewidth=2, color='blue')
    plt.plot(online_thresholds, mapping_dict['wp95_pt90'], label='@ 90% efficiency', linewidth=2, color='red')
    plt.plot(online_thresholds, mapping_dict['wp95_pt50'], label='@ 50% efficiency', linewidth=2, color='green')
    plt.legend(loc = 'lower right', fontsize=14)
    plt.xlabel('L1 Threshold [GeV]')
    plt.ylabel('Offline threshold [GeV]')
    plt.xlim(0, 110)
    # plt.ylim(0, 200)
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(perfdir+'/plots/online2offline_WP95.pdf')
    plt.close()

    plt.figure(figsize=(10,10))
    plt.plot(online_thresholds, mapping_dict['wp90_pt95'], label='@ 95% efficiency', linewidth=2, color='blue')
    plt.plot(online_thresholds, mapping_dict['wp90_pt90'], label='@ 90% efficiency', linewidth=2, color='red')
    plt.plot(online_thresholds, mapping_dict['wp90_pt50'], label='@ 50% efficiency', linewidth=2, color='green')
    plt.legend(loc = 'lower right', fontsize=14)
    plt.xlabel('L1 Threshold [GeV]')
    plt.ylabel('Offline threshold [GeV]')
    plt.xlim(0, 110)
    # plt.ylim(0, 200)
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(perfdir+'/plots/online2offline_WP90.pdf')
    plt.close()


    ##################################################################################
    # PLOT TURNONS PER DM
    i = 0
    plt.figure(figsize=(10,10))
    for thr in plotting_thresholds:
        if not thr%10:
            plt.errorbar(pt_bins_centers,turnons_dm_dict['dm0TurnonAt99wpAt'+str(thr)+'GeV'][:,0],xerr=1,yerr=[turnons_dm_dict['dm0TurnonAt99wpAt'+str(thr)+'GeV'][:,1], turnons_dm_dict['dm0TurnonAt99wpAt'+str(thr)+'GeV'][:,2]], ls='None', label=r'$p_{T}^{L1 \tau} > %i$ GeV' % (thr), lw=2, marker='o', color=cmap(i))

            p0 = [1, thr, 1] 
            popt, pcov = curve_fit(sigmoid, pt_bins_centers, turnons_dm_dict['dm0TurnonAt99wpAt'+str(thr)+'GeV'][:,0], p0, maxfev=5000)
            plt.plot(pt_bins_centers, sigmoid(pt_bins_centers, *popt), '-', label='_', lw=1.5, color=cmap(i))

            i+=1 
    plt.hlines(0.90, 0, 2000, lw=2, color='dimgray', label='0.90 Eff.')
    plt.hlines(0.95, 0, 2000, lw=2, color='black', label='0.95 Eff.')
    plt.legend(loc = 'lower right', fontsize=14)
    plt.ylim(0., 1.05)
    plt.xlim(0., 150.)
    plt.xlabel(r'$p_{T}^{gen,\tau}\ [GeV]$')
    plt.ylabel(r'Efficiency')
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(perfdir+'/plots/dm0_turnons_WP99.pdf')
    plt.close()

    i = 0
    plt.figure(figsize=(10,10))
    for thr in plotting_thresholds:
        if not thr%10:
            plt.errorbar(pt_bins_centers,turnons_dm_dict['dm1TurnonAt99wpAt'+str(thr)+'GeV'][:,0],xerr=1,yerr=[turnons_dm_dict['dm1TurnonAt99wpAt'+str(thr)+'GeV'][:,1], turnons_dm_dict['dm1TurnonAt99wpAt'+str(thr)+'GeV'][:,2]], ls='None', label=r'$p_{T}^{L1 \tau} > %i$ GeV' % (thr), lw=2, marker='o', color=cmap(i))

            p0 = [1, thr, 1] 
            popt, pcov = curve_fit(sigmoid, pt_bins_centers, turnons_dm_dict['dm1TurnonAt99wpAt'+str(thr)+'GeV'][:,0], p0, maxfev=5000)
            plt.plot(pt_bins_centers, sigmoid(pt_bins_centers, *popt), '-', label='_', lw=1.5, color=cmap(i))

            i+=1 
    plt.hlines(0.90, 0, 2000, lw=2, color='dimgray', label='0.90 Eff.')
    plt.hlines(0.95, 0, 2000, lw=2, color='black', label='0.95 Eff.')
    plt.legend(loc = 'lower right', fontsize=14)
    plt.ylim(0., 1.05)
    plt.xlim(0., 150.)
    plt.xlabel(r'$p_{T}^{gen,\tau}\ [GeV]$')
    plt.ylabel(r'Efficiency')
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(perfdir+'/plots/dm1_turnons_WP99.pdf')
    plt.close()

    i = 0
    plt.figure(figsize=(10,10))
    for thr in plotting_thresholds:
        if not thr%10:
            plt.errorbar(pt_bins_centers,turnons_dm_dict['dm10TurnonAt99wpAt'+str(thr)+'GeV'][:,0],xerr=1,yerr=[turnons_dm_dict['dm10TurnonAt99wpAt'+str(thr)+'GeV'][:,1], turnons_dm_dict['dm10TurnonAt99wpAt'+str(thr)+'GeV'][:,2]], ls='None', label=r'$p_{T}^{L1 \tau} > %i$ GeV' % (thr), lw=2, marker='o', color=cmap(i))

            p0 = [1, thr, 1] 
            popt, pcov = curve_fit(sigmoid, pt_bins_centers, turnons_dm_dict['dm10TurnonAt99wpAt'+str(thr)+'GeV'][:,0], p0, maxfev=5000)
            plt.plot(pt_bins_centers, sigmoid(pt_bins_centers, *popt), '-', label='_', lw=1.5, color=cmap(i))

            i+=1 
    plt.hlines(0.90, 0, 2000, lw=2, color='dimgray', label='0.90 Eff.')
    plt.hlines(0.95, 0, 2000, lw=2, color='black', label='0.95 Eff.')
    plt.legend(loc = 'lower right', fontsize=14)
    plt.ylim(0., 1.05)
    plt.xlim(0., 150.)
    plt.xlabel(r'$p_{T}^{gen,\tau}\ [GeV]$')
    plt.ylabel(r'Efficiency')
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(perfdir+'/plots/dm10_turnons_WP99.pdf')
    plt.close()

    i = 0
    plt.figure(figsize=(10,10))
    for thr in plotting_thresholds:
        if not thr%10:
            plt.errorbar(pt_bins_centers,turnons_dm_dict['dm11TurnonAt99wpAt'+str(thr)+'GeV'][:,0],xerr=1,yerr=[turnons_dm_dict['dm11TurnonAt99wpAt'+str(thr)+'GeV'][:,1], turnons_dm_dict['dm11TurnonAt99wpAt'+str(thr)+'GeV'][:,2]], ls='None', label=r'$p_{T}^{L1 \tau} > %i$ GeV' % (thr), lw=2, marker='o', color=cmap(i))

            p0 = [1, thr, 1] 
            popt, pcov = curve_fit(sigmoid, pt_bins_centers, turnons_dm_dict['dm11TurnonAt99wpAt'+str(thr)+'GeV'][:,0], p0, maxfev=5000)
            plt.plot(pt_bins_centers, sigmoid(pt_bins_centers, *popt), '-', label='_', lw=1.5, color=cmap(i))

            i+=1 
    plt.hlines(0.90, 0, 2000, lw=2, color='dimgray', label='0.90 Eff.')
    plt.hlines(0.95, 0, 2000, lw=2, color='black', label='0.95 Eff.')
    plt.legend(loc = 'lower right', fontsize=14)
    plt.ylim(0., 1.05)
    plt.xlim(0., 150.)
    plt.xlabel(r'$p_{T}^{gen,\tau}\ [GeV]$')
    plt.ylabel(r'Efficiency')
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(perfdir+'/plots/dm11_turnons_WP99.pdf')
    plt.close()


    ##################################################################################
    # PLOT EFFICIENCIES VS ETA
    # i = 0
    # plt.figure(figsize=(10,10))
    # for thr in plotting_thresholds:
    #     if not thr%10:
    #         plt.errorbar(eta_bins_centers,etaeffs_dict['efficiencyVsEtaAt99wpAt'+str(thr)+'GeV'][:,0],xerr=1,yerr=[etaeffs_dict['efficiencyVsEtaAt99wpAt'+str(thr)+'GeV'][:,1], etaeffs_dict['efficiencyVsEtaAt99wpAt'+str(thr)+'GeV'][:,2]], ls='None', label=r'$p_{T}^{L1 \tau} > %i$ GeV' % (thr), lw=2, marker='o', color=cmap(i))
    #         i+=1 
    # plt.hlines(0.90, 0, eta_bins_centers.max(), lw=2, color='dimgray', label='0.90 Eff.')
    # plt.hlines(0.95, 0, eta_bins_centers.max(), lw=2, color='black', label='0.95 Eff.')
    # plt.legend(loc = 'lower right', fontsize=14)
    # plt.ylim(0., 1.05)
    # plt.xlim(0., 150.)
    # plt.xlabel(r'$p_{T}^{gen,\tau}\ [GeV]$')
    # plt.ylabel(r'Efficiency')
    # plt.grid()
    # mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    # plt.savefig(perfdir+'/plots/turnons_WP99.pdf')
    # plt.close()

    # i = 0
    # plt.figure(figsize=(10,10))
    # for thr in plotting_thresholds:
    #     if not thr%10:
    #         plt.errorbar(eta_bins_centers,etaeffs_dict['efficiencyVsEtaAt95wpAt'+str(thr)+'GeV'][:,0],xerr=1,yerr=[etaeffs_dict['efficiencyVsEtaAt95wpAt'+str(thr)+'GeV'][:,1], etaeffs_dict['efficiencyVsEtaAt95wpAt'+str(thr)+'GeV'][:,2]], ls='None', label=r'$p_{T}^{L1 \tau} > %i$ GeV' % (thr), lw=2, marker='o', color=cmap(i))
    #         i+=1 
    # plt.hlines(0.90, 0, eta_bins_centers.max(), lw=2, color='dimgray', label='0.90 Eff.')
    # plt.hlines(0.95, 0, eta_bins_centers.max(), lw=2, color='black', label='0.95 Eff.')
    # plt.legend(loc = 'lower right', fontsize=14)
    # plt.ylim(0., 1.05)
    # plt.xlim(0., 150.)
    # plt.xlabel(r'$p_{T}^{gen,\tau}\ [GeV]$')
    # plt.ylabel(r'Efficiency')
    # plt.grid()
    # mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    # plt.savefig(perfdir+'/plots/turnons_WP95.pdf')
    # plt.close()

    # i = 0
    # plt.figure(figsize=(10,10))
    # for thr in plotting_thresholds:
    #     if not thr%10:
    #         plt.errorbar(eta_bins_centers,etaeffs_dict['efficiencyVsEtaAt90wpAt'+str(thr)+'GeV'][:,0],xerr=1,yerr=[etaeffs_dict['efficiencyVsEtaAt90wpAt'+str(thr)+'GeV'][:,1], etaeffs_dict['efficiencyVsEtaAt90wpAt'+str(thr)+'GeV'][:,2]], ls='None', label=r'$p_{T}^{L1 \tau} > %i$ GeV' % (thr), lw=2, marker='o', color=cmap(i))
    #         i+=1 
    # plt.hlines(0.90, 0, eta_bins_centers.max(), lw=2, color='dimgray', label='0.90 Eff.')
    # plt.hlines(0.95, 0, eta_bins_centers.max(), lw=2, color='black', label='0.95 Eff.')
    # plt.legend(loc = 'lower right', fontsize=14)
    # plt.ylim(0., 1.05)
    # plt.xlim(0., 150.)
    # plt.xlabel(r'$p_{T}^{gen,\tau}\ [GeV]$')
    # plt.ylabel(r'Efficiency')
    # plt.grid()
    # mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    # plt.savefig(perfdir+'/plots/turnons_WP90.pdf')
    # plt.close()


    ##################################################################################
    # PLOT MISSED TAUs
    plt.figure(figsize=(10,10))
    plt.hist(dfTauMissed['tau_visPt'], bins=np.arange(18,203,5), label=r'Geometrically missed $\tau$', linewidth=2, color='blue', density=True, histtype='step')
    plt.legend(loc = 'upper right')
    plt.xlabel(r'$p_T^{Gen. \tau}$ [GeV]')
    plt.ylabel('a.u.')
    plt.yscale('log')
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(perfdir+'/plots/missed_pt.pdf')
    plt.close()

    plt.figure(figsize=(10,10))
    plt.hist(dfTauMissed['tau_visEta'], bins=np.arange(-3.1,3.1,0.1), label=r'Geometrically missed $\tau$', linewidth=2, color='blue', density=True, histtype='step')
    plt.legend(loc = 'upper right')
    plt.xlabel(r'$\eta^{Gen. \tau}$')
    plt.ylabel('a.u.')
    # plt.xlim(0, 110)
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(perfdir+'/plots/missed_eta.pdf')
    plt.close()

    plt.figure(figsize=(10,10))
    plt.hist(dfTauMissed['tau_visPhi'], bins=np.arange(-3.2,3.2,0.1), label=r'Geometrically missed $\tau$', linewidth=2, color='blue', density=True, histtype='step')
    plt.legend(loc = 'upper right')
    plt.xlabel(r'$\phi^{Gen. \tau}$')
    plt.ylabel('a.u.')
    # plt.xlim(0, 110)
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(perfdir+'/plots/missed_phi.pdf')
    plt.close()

    plt.figure(figsize=(10,10))
    plt.hist(dfTauMissed['tau_DM'], label=r'Geometrically missed $\tau$', linewidth=2, color='blue', density=True, histtype='step')
    plt.legend(loc = 'upper right')
    plt.xlabel(r'$\tau$ decay mode')
    plt.ylabel('a.u.')
    # plt.xlim(0, 110)
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(perfdir+'/plots/missed_dm.pdf')
    plt.close()











