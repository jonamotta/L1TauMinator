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

def efficiency(g, WP, thr, upper=False):
    sel = g[(g['pass'+WP]==1) & (g['L1_pt']>thr)].shape[0]
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

    X1_calib = np.load(indir+'/TauCNNEvaluator'+options.caloClNxM+options.inTag+'/X_Calib_CNN_'+options.caloClNxM+'_forEvaluator.npz')['arr_0']
    X2_calib = np.load(indir+'/TauCNNEvaluator'+options.caloClNxM+options.inTag+'/X_Calib_Dense_'+options.caloClNxM+'_forEvaluator.npz')['arr_0']
    Y_calib  = np.load(indir+'/TauCNNEvaluator'+options.caloClNxM+options.inTag+'/Y_Calib_'+options.caloClNxM+'_forEvaluator.npz')['arr_0']

    X1_ident = np.load(indir+'/TauCNNEvaluator'+options.caloClNxM+options.inTag+'/X_Ident_CNN_'+options.caloClNxM+'_forEvaluator.npz')['arr_0']
    X2_ident = np.load(indir+'/TauCNNEvaluator'+options.caloClNxM+options.inTag+'/X_Ident_Dense_'+options.caloClNxM+'_forEvaluator.npz')['arr_0']
    Y_ident  = np.load(indir+'/TauCNNEvaluator'+options.caloClNxM+options.inTag+'/Y_Ident_'+options.caloClNxM+'_forEvaluator.npz')['arr_0']

    Xidentified = TauIdentifierModel.predict([X1_ident, X2_ident])
    FPR, TPR, THR = metrics.roc_curve(Y_ident, Xidentified)
    WP99 = np.interp(0.99, TPR, THR)
    WP95 = np.interp(0.95, TPR, THR)
    WP90 = np.interp(0.90, TPR, THR)

    df = pd.DataFrame()
    df['gen_pt' ] = Y_calib[:,0].ravel()
    df['gen_eta'] = Y_calib[:,1].ravel()
    df['gen_phi'] = Y_calib[:,2].ravel()
    df['gen_dm']  = Y_calib[:,3].ravel()
    df['L1_pt']   = TauCalibratorModel.predict([X1_calib, X2_calib]).ravel()
    df['score']   = TauIdentifierModel.predict([X1_calib, X2_calib]).ravel()
    df['pass99']  = df['score'] > WP99
    df['pass95']  = df['score'] > WP95
    df['pass90']  = df['score'] > WP90

    df['gen_pt_bin'] = pd.cut(df['gen_pt'],
                              bins=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 2000],
                              # bins=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 200, 500, 1000, 2000],
                              labels=False,
                              include_lowest=True)

    df['gen_eta_bin'] = pd.cut(df['gen_eta'],
                              bins=[-3.5, -3.0, -2.7, -2.4, -2.1, -1.8, -1.5, -1.305, -1.0, -0.66, -0.33, 0.0, 0.33, 0.66, 1.0, 1.305, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0, 3.5],
                              labels=False,
                              include_lowest=True)

    df_dm0  = df[df['gen_dm'] == 0].copy(deep=True)
    df_dm1  = df[(df['gen_dm'] == 1) | (df['gen_dm'] == 2)].copy(deep=True)
    df_dm10 = df[df['gen_dm'] == 10].copy(deep=True)
    df_dm11 = df[(df['gen_dm'] == 11) | (df['gen_dm'] == 12)].copy(deep=True)

    cmap = matplotlib.cm.get_cmap('tab20c'); i=0
    # pt_bins_centers = np.arange(2.5,157.5,5)
    pt_bins_centers = np.arange(22.5,157.5,5)
    # pt_bins_centers = [22.5, 27.5, 32.5, 37.5, 42.5, 47.5, 52.5, 57.5, 62.5, 67.5, 72.5, 77.5, 82.5, 87.5, 92.5, 97.5, 102.5, 107.5, 112.5, 117.5, 122.5, 127.5, 132.5, 137.5, 142.5, 147.5, 175, 350, 750, 1500]
    eta_bins_centers = [-3.25, -2.85, -2.55, -2.25, -1.95, -1.65, -1.4025, -1.1525, -0.825, -0.495, -0.165, 0.165, 0.495, 0.825, 1.1525, 1.4025, 1.65, 1.95, 2.25, 2.55, 2.85, 3.25]
    # online_thresholds = range(1, 175, 1)
    online_thresholds = range(20, 175, 1)
    # plotting_thresholds = range(10, 110, 10)
    plotting_thresholds = range(20, 110, 10)
    turnons_dm_dict = {}
    turnons_dict = {}
    etaeffs_dict = {}
    mapping_dict = {'threshold':[], 'wp99_pt95':[], 'wp99_pt90':[], 'wp99_pt50':[],
                                    'wp95_pt95':[], 'wp95_pt90':[], 'wp95_pt50':[],
                                    'wp90_pt95':[], 'wp90_pt90':[], 'wp90_pt50':[]}
    offline_pts = df.groupby('gen_pt_bin').mean()['gen_pt']

    for thr in online_thresholds:
        print(' ** INFO : calculating turnons and mappings for threshold '+str(thr)+' GeV')

        # TURNONS
        grouped = df.groupby('gen_pt_bin').apply(lambda g: efficiency(g, '99', thr))
        turnon = []
        for ton in grouped: turnon.append([ton[0], ton[1], ton[2]])
        turnons_dict['turnonAt99wpAt'+str(thr)+'GeV'] = np.array(turnon)

        grouped = df.groupby('gen_pt_bin').apply(lambda g: efficiency(g, '95', thr))
        turnon = []
        for ton in grouped: turnon.append([ton[0], ton[1], ton[2]])
        turnons_dict['turnonAt95wpAt'+str(thr)+'GeV'] = np.array(turnon)

        grouped = df.groupby('gen_pt_bin').apply(lambda g: efficiency(g, '90', thr))
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
        grouped = df_dm0.groupby('gen_pt_bin').apply(lambda g: efficiency(g, '99', thr))
        turnon = []
        for ton in grouped: turnon.append([ton[0], ton[1], ton[2]])
        turnons_dm_dict['dm0TurnonAt99wpAt'+str(thr)+'GeV'] = np.array(turnon)

        grouped = df_dm0.groupby('gen_pt_bin').apply(lambda g: efficiency(g, '95', thr))
        turnon = []
        for ton in grouped: turnon.append([ton[0], ton[1], ton[2]])
        turnons_dm_dict['dm0TurnonAt95wpAt'+str(thr)+'GeV'] = np.array(turnon)

        grouped = df_dm0.groupby('gen_pt_bin').apply(lambda g: efficiency(g, '90', thr))
        turnon = []
        for ton in grouped: turnon.append([ton[0], ton[1], ton[2]])
        turnons_dm_dict['dm0TurnonAt90wpAt'+str(thr)+'GeV'] = np.array(turnon)

        # TURNONS DM 1
        grouped = df_dm1.groupby('gen_pt_bin').apply(lambda g: efficiency(g, '99', thr))
        turnon = []
        for ton in grouped: turnon.append([ton[0], ton[1], ton[2]])
        turnons_dm_dict['dm1TurnonAt99wpAt'+str(thr)+'GeV'] = np.array(turnon)

        grouped = df_dm1.groupby('gen_pt_bin').apply(lambda g: efficiency(g, '95', thr))
        turnon = []
        for ton in grouped: turnon.append([ton[0], ton[1], ton[2]])
        turnons_dm_dict['dm1TurnonAt95wpAt'+str(thr)+'GeV'] = np.array(turnon)

        grouped = df_dm1.groupby('gen_pt_bin').apply(lambda g: efficiency(g, '90', thr))
        turnon = []
        for ton in grouped: turnon.append([ton[0], ton[1], ton[2]])
        turnons_dm_dict['dm1TurnonAt90wpAt'+str(thr)+'GeV'] = np.array(turnon)

        # TURNONS DM 10
        grouped = df_dm10.groupby('gen_pt_bin').apply(lambda g: efficiency(g, '99', thr))
        turnon = []
        for ton in grouped: turnon.append([ton[0], ton[1], ton[2]])
        turnons_dm_dict['dm10TurnonAt99wpAt'+str(thr)+'GeV'] = np.array(turnon)

        grouped = df_dm10.groupby('gen_pt_bin').apply(lambda g: efficiency(g, '95', thr))
        turnon = []
        for ton in grouped: turnon.append([ton[0], ton[1], ton[2]])
        turnons_dm_dict['dm10TurnonAt95wpAt'+str(thr)+'GeV'] = np.array(turnon)

        grouped = df_dm10.groupby('gen_pt_bin').apply(lambda g: efficiency(g, '90', thr))
        turnon = []
        for ton in grouped: turnon.append([ton[0], ton[1], ton[2]])
        turnons_dm_dict['dm10TurnonAt90wpAt'+str(thr)+'GeV'] = np.array(turnon)

        # TURNONS DM 11
        grouped = df_dm11.groupby('gen_pt_bin').apply(lambda g: efficiency(g, '99', thr))
        turnon = []
        for ton in grouped: turnon.append([ton[0], ton[1], ton[2]])
        turnons_dm_dict['dm11TurnonAt99wpAt'+str(thr)+'GeV'] = np.array(turnon)

        grouped = df_dm11.groupby('gen_pt_bin').apply(lambda g: efficiency(g, '95', thr))
        turnon = []
        for ton in grouped: turnon.append([ton[0], ton[1], ton[2]])
        turnons_dm_dict['dm11TurnonAt95wpAt'+str(thr)+'GeV'] = np.array(turnon)

        grouped = df_dm11.groupby('gen_pt_bin').apply(lambda g: efficiency(g, '90', thr))
        turnon = []
        for ton in grouped: turnon.append([ton[0], ton[1], ton[2]])
        turnons_dm_dict['dm11TurnonAt90wpAt'+str(thr)+'GeV'] = np.array(turnon)

        # ETA EFFICIENCIES
        # grouped = df.groupby('gen_eta_bin').apply(lambda g: efficiency(g, '99', thr))
        # efficiency = []
        # for eff in grouped: efficiency.append([eff[0], eff[1], eff[2]])
        # etaeffs_dict['efficiencyVsEtaAt99wpAt'+str(thr)+'GeV'] = np.array(efficiency)

        # grouped = df.groupby('gen_eta_bin').apply(lambda g: efficiency(g, '95', thr))
        # efficiency = []
        # for eff in grouped: efficiency.append([eff[0], eff[1], eff[2]])
        # etaeffs_dict['efficiencyVsEtaAt95wpAt'+str(thr)+'GeV'] = np.array(efficiency)

        # grouped = df.groupby('gen_eta_bin').apply(lambda g: efficiency(g, '90', thr))
        # efficiency = []
        # for eff in grouped: efficiency.append([eff[0], eff[1], eff[2]])
        # etaeffs_dict['efficiencyVsEtaAt90wpAt'+str(thr)+'GeV'] = np.array(efficiency)

    save_obj(mapping_dict, indir+'/TauCNNEvaluator'+options.caloClNxM+options.inTag+'/online2offline_mapping.pkl')


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
    plt.savefig(indir+'/TauCNNEvaluator'+options.caloClNxM+options.inTag+'/turnons_WP99.pdf')
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
    plt.savefig(indir+'/TauCNNEvaluator'+options.caloClNxM+options.inTag+'/turnons_WP95.pdf')
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
    plt.savefig(indir+'/TauCNNEvaluator'+options.caloClNxM+options.inTag+'/turnons_WP90.pdf')
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
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(indir+'/TauCNNEvaluator'+options.caloClNxM+options.inTag+'/online2offline_WP99.pdf')
    plt.close()

    plt.figure(figsize=(10,10))
    plt.plot(online_thresholds, mapping_dict['wp95_pt95'], label='@ 95% efficiency', linewidth=2, color='blue')
    plt.plot(online_thresholds, mapping_dict['wp95_pt90'], label='@ 90% efficiency', linewidth=2, color='red')
    plt.plot(online_thresholds, mapping_dict['wp95_pt50'], label='@ 50% efficiency', linewidth=2, color='green')
    plt.legend(loc = 'lower right', fontsize=14)
    plt.xlabel('L1 Threshold [GeV]')
    plt.ylabel('Offline threshold [GeV]')
    plt.xlim(0, 110)
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(indir+'/TauCNNEvaluator'+options.caloClNxM+options.inTag+'/online2offline_WP95.pdf')
    plt.close()

    plt.figure(figsize=(10,10))
    plt.plot(online_thresholds, mapping_dict['wp90_pt95'], label='@ 95% efficiency', linewidth=2, color='blue')
    plt.plot(online_thresholds, mapping_dict['wp90_pt90'], label='@ 90% efficiency', linewidth=2, color='red')
    plt.plot(online_thresholds, mapping_dict['wp90_pt50'], label='@ 50% efficiency', linewidth=2, color='green')
    plt.legend(loc = 'lower right', fontsize=14)
    plt.xlabel('L1 Threshold [GeV]')
    plt.ylabel('Offline threshold [GeV]')
    plt.xlim(0, 110)
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(indir+'/TauCNNEvaluator'+options.caloClNxM+options.inTag+'/online2offline_WP90.pdf')
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
    plt.savefig(indir+'/TauCNNEvaluator'+options.caloClNxM+options.inTag+'/dm0_turnons_WP99.pdf')
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
    plt.savefig(indir+'/TauCNNEvaluator'+options.caloClNxM+options.inTag+'/dm1_turnons_WP99.pdf')
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
    plt.savefig(indir+'/TauCNNEvaluator'+options.caloClNxM+options.inTag+'/dm10_turnons_WP99.pdf')
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
    plt.savefig(indir+'/TauCNNEvaluator'+options.caloClNxM+options.inTag+'/dm11_turnons_WP99.pdf')
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
    # plt.savefig(indir+'/TauCNNEvaluator'+options.caloClNxM+options.inTag+'/turnons_WP99.pdf')
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
    # plt.savefig(indir+'/TauCNNEvaluator'+options.caloClNxM+options.inTag+'/turnons_WP95.pdf')
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
    # plt.savefig(indir+'/TauCNNEvaluator'+options.caloClNxM+options.inTag+'/turnons_WP90.pdf')
    # plt.close()