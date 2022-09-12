from scipy.optimize import curve_fit
from optparse import OptionParser
from scipy.special import btdtri # beta quantile function
from tensorflow import keras
from sklearn import metrics
import tensorflow as tf
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
    delta_eta = np.abs(df['cl_eta']-df['cl_eta_ass'])
    delta_phi = np.abs(df['cl_phi']-df['cl_phi_ass'])
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
    parser.add_option("--v",            dest="v",                                 default=None)
    parser.add_option("--date",         dest="date",                              default=None)
    parser.add_option("--inTag",        dest="inTag",                             default="")
    parser.add_option("--modelsTag",    dest="modelsTag",                         default="")
    parser.add_option('--caloClNxM',    dest='caloClNxM',                         default="9x9")
    (options, args) = parser.parse_args()
    print(options)

    # get clusters' shape dimensions
    N = int(options.caloClNxM.split('x')[0])
    M = int(options.caloClNxM.split('x')[1])

    indir = '/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v
    outdir = indir+'/TauBDTEvaluator'+options.inTag

    TauCalibratorModel = keras.models.load_model(indir+'/TauBDTCalibratorTraining'+options.modelsTag+'/TauBDTCalibrator', compile=False)
    TauIdentifierModel = keras.models.load_model(indir+'/TauBDTIdentifierTraining'+options.modelsTag+'/TauBDTIdentifier', compile=False)

    X1 = np.load(outdir+'/X_Rate_BDT_forEvaluator.npz')['arr_0']
    X2 = np.load(outdir+'/X_Rate_Dense_forEvaluator.npz')['arr_0']
    Y = np.load(outdir+'/Y_Rate_forEvaluator.npz')['arr_0']

    Xidentified = TauIdentifierModel.predict([X1, X2])
    Xcalibrated = TauCalibratorModel.predict([X1, X2])

    wp_dict = load_obj(indir+'/TauBDTEvaluator'+options.modelsTag+'/TauMinatorBDT_WPs.pkl')
    WP99 = wp_dict['wp99']
    WP95 = wp_dict['wp95']
    WP90 = wp_dict['wp90']

    df = pd.DataFrame()
    df['event']   = Y[:,0].ravel()
    df['cl_eta']  = Y[:,1].ravel()
    df['cl_phi']  = Y[:,2].ravel()
    df['cl_pt']   = Xcalibrated.ravel()
    df['score']   = Xidentified.ravel()
    df['pass99']  = df['score'] > WP99
    df['pass95']  = df['score'] > WP95
    df['pass90']  = df['score'] > WP90

    # df = df.head(500).copy(deep=True)

    events_frequency=2808*11.2  # N_bunches * frequency [kHz] --> from: https://cds.cern.ch/record/2130736/files/Introduction%20to%20the%20HL-LHC%20Project.pdf
    mapping_dict = load_obj(indir+'/TauBDTEvaluator'+options.modelsTag+'/online2offline_mapping.pkl')
    online_thresholds = range(20, 175, 1)
    events_total = np.unique(df['event']).shape[0]
    rates_online = {'singleTau' : {}, 'doubleTau' : {}}
    rates_offline = {'singleTau' : {}, 'doubleTau' : {}}

    for WP in ['99', '95', '90']:
        tmp = df[(df['cl_pt']>32) & (df['pass'+WP])]

        tmp['L1_wp'+WP+'_pt95'] = tmp['cl_pt'].apply(lambda x : np.interp(x, online_thresholds, mapping_dict['wp'+WP+'_pt95']))
        tmp['L1_wp'+WP+'_pt90'] = tmp['cl_pt'].apply(lambda x : np.interp(x, online_thresholds, mapping_dict['wp'+WP+'_pt90']))
        tmp['L1_wp'+WP+'_pt50'] = tmp['cl_pt'].apply(lambda x : np.interp(x, online_thresholds, mapping_dict['wp'+WP+'_pt50']))

        #********** single tau rate **********#
        sel_events = np.unique(tmp['event']).shape[0]
        print('             singleTau: selected_clusters / total_clusters = {0} / {1} = {2} '.format(sel_events, events_total, float(sel_events)/float(events_total)))

        tmp.sort_values('cl_pt', inplace=True)
        tmp['COTcount_singleTau'] = np.linspace(tmp.shape[0]-1, 0, tmp.shape[0])

        rates_online['singleTau']['wp'+WP] = ( np.sort(tmp['cl_pt']), np.array(tmp['COTcount_singleTau'])/events_total*events_frequency, np.array(tmp['COTcount_singleTau']) )
        rates_offline['singleTau']['wp'+WP+'_pt95'] = ( np.sort(tmp['L1_wp'+WP+'_pt95']), np.array(tmp['COTcount_singleTau'])/events_total*events_frequency, np.array(tmp['COTcount_singleTau']) )
        rates_offline['singleTau']['wp'+WP+'_pt90'] = ( np.sort(tmp['L1_wp'+WP+'_pt90']), np.array(tmp['COTcount_singleTau'])/events_total*events_frequency, np.array(tmp['COTcount_singleTau']) )
        rates_offline['singleTau']['wp'+WP+'_pt50'] = ( np.sort(tmp['L1_wp'+WP+'_pt50']), np.array(tmp['COTcount_singleTau'])/events_total*events_frequency, np.array(tmp['COTcount_singleTau']) )

        #********** double tau rate **********#
        tmp['doubleTau'] = tmp.duplicated('event', keep=False)
        tmp.query('doubleTau==True', inplace=True)
        applyDRcut(tmp, 0.5)
        sel_events = np.unique(tmp['event']).shape[0]
        print('             doubleTau: selected_clusters / total_clusters = {0} / {1} = {2} '.format(sel_events, events_total, float(sel_events)/float(events_total)))

        tmp.sort_values('cl_pt', inplace=True)
        tmp['COTcount_doubleTau'] = np.linspace(tmp.shape[0]-1, 0, tmp.shape[0])

        rates_online['doubleTau']['wp'+WP] = ( np.sort(tmp['cl_pt']), np.array(tmp['COTcount_doubleTau'])/events_total*events_frequency, np.array(tmp['COTcount_doubleTau']) )
        rates_offline['doubleTau']['wp'+WP+'_pt95'] = ( np.sort(tmp['L1_wp'+WP+'_pt95']), np.array(tmp['COTcount_doubleTau'])/events_total*events_frequency, np.array(tmp['COTcount_doubleTau']) )
        rates_offline['doubleTau']['wp'+WP+'_pt90'] = ( np.sort(tmp['L1_wp'+WP+'_pt90']), np.array(tmp['COTcount_doubleTau'])/events_total*events_frequency, np.array(tmp['COTcount_doubleTau']) )
        rates_offline['doubleTau']['wp'+WP+'_pt50'] = ( np.sort(tmp['L1_wp'+WP+'_pt50']), np.array(tmp['COTcount_doubleTau'])/events_total*events_frequency, np.array(tmp['COTcount_doubleTau']) )

        del tmp

    for WP in ['99', '95', '90']:
        print('    - WP = '+WP)

        plt.figure(figsize=(8,8))
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
        plt.savefig(outdir+'/rate_singleTau_offline_wp'+WP+'.pdf')
        plt.close()

        print('        - single tau offline threshold for 25kHz rate @ 95% efficiency = {0}'.format( round(np.interp(25, np.flip(rates_offline['singleTau']['wp'+WP+'_pt95'][1]), np.flip(rates_offline['singleTau']['wp'+WP+'_pt95'][0]), left=0.0, right=0.0),0) ))
        print('        - single tau offline threshold for 25kHz rate @ 90% efficiency = {0}'.format( round(np.interp(25, np.flip(rates_offline['singleTau']['wp'+WP+'_pt90'][1]), np.flip(rates_offline['singleTau']['wp'+WP+'_pt90'][0]), left=0.0, right=0.0),0) ))
        print('        - single tau offline threshold for 25kHz rate @ 50% efficiency = {0}'.format( round(np.interp(25, np.flip(rates_offline['singleTau']['wp'+WP+'_pt50'][1]), np.flip(rates_offline['singleTau']['wp'+WP+'_pt50'][0]), left=0.0, right=0.0),0) ))

        plt.figure(figsize=(8,8))
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
        plt.savefig(outdir+'/rate_doubleTau_offline_wp'+WP+'.pdf')
        plt.close()

        print('        - double tau offline threshold for 25kHz rate @ 95% efficiency = {0},{0}'.format( round(np.interp(25, np.flip(rates_offline['doubleTau']['wp'+WP+'_pt95'][1]), np.flip(rates_offline['doubleTau']['wp'+WP+'_pt95'][0]), left=0.0, right=0.0),0) ))
        print('        - double tau offline threshold for 25kHz rate @ 90% efficiency = {0},{0}'.format( round(np.interp(25, np.flip(rates_offline['doubleTau']['wp'+WP+'_pt90'][1]), np.flip(rates_offline['doubleTau']['wp'+WP+'_pt90'][0]), left=0.0, right=0.0),0) ))
        print('        - double tau offline threshold for 25kHz rate @ 50% efficiency = {0},{0}'.format( round(np.interp(25, np.flip(rates_offline['doubleTau']['wp'+WP+'_pt50'][1]), np.flip(rates_offline['doubleTau']['wp'+WP+'_pt50'][0]), left=0.0, right=0.0),0) ))














