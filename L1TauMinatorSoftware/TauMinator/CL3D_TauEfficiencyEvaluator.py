from scipy.optimize import curve_fit
from optparse import OptionParser
from scipy.special import btdtri # beta quantile function
from sklearn import metrics
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
    parser.add_option("--v",            dest="v",                                 default=None)
    parser.add_option("--date",         dest="date",                              default=None)
    parser.add_option("--inTagCalib",   dest="inTagCalib",                        default="")
    parser.add_option("--inTagIdent",   dest="inTagIdent",                        default="")
    (options, args) = parser.parse_args()
    print(options)

    indir = '/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v
    outdir = indir+'/TauBDTEvaluator_Calib'+options.inTagCalib+'_Ident'+options.inTagIdent
    os.system('mkdir -p '+outdir+'/plots')

    PUmodel = load_obj(indir+'/TauBDTIdentifierTraining'+options.inTagIdent+'/TauBDTIdentifier/PUmodel.pkl')
    C1model = load_obj(indir+'/TauBDTCalibratorTraining'+options.inTagCalib+'/TauBDTCalibrator/C1model.pkl')
    C2model = load_obj(indir+'/TauBDTCalibratorTraining'+options.inTagCalib+'/TauBDTCalibrator/C2model.pkl')
    C3model = load_obj(indir+'/TauBDTCalibratorTraining'+options.inTagCalib+'/TauBDTCalibrator/C3model.pkl')

    dfCalib1 = pd.read_pickle(indir+'/TauBDTCalibratorTraining'+options.inTagCalib+'/X_Calib_BDT_forEvaluator.pkl')
    dfCalib2 = pd.read_pickle(indir+'/TauBDTCalibratorTraining'+options.inTagCalib+'/X_Calib_BDT_forCalibrator.pkl')
    dfCalib  = pd.concat([dfCalib1, dfCalib2], axis=0, sort=False)
    dfCalib['cl3d_abseta'] = abs(dfCalib['cl3d_eta']).copy(deep=True)

    featuresCalib = ['cl3d_showerlength', 'cl3d_coreshowerlength', 'cl3d_abseta', 'cl3d_spptot', 'cl3d_srrmean', 'cl3d_meanz']
    featuresCalibN = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5']

    dfIdent = pd.read_pickle(indir+'/TauBDTIdentifierTraining'+options.inTagIdent+'/X_Ident_BDT_forEvaluator.pkl')
    dfIdent['cl3d_abseta'] = abs(dfIdent['cl3d_eta']).copy(deep=True)

    featuresIdent = ['cl3d_pt', 'cl3d_coreshowerlength', 'cl3d_srrtot', 'cl3d_srrmean', 'cl3d_hoe', 'cl3d_meanz']
    featuresIdentN = ['f0', 'f1', 'f2', 'f3', 'f4', 'f5']

    # rename features with numbers (needed by PUmodel)
    for i in range(len(featuresIdent)): dfIdent[featuresIdentN[i]] = dfIdent[featuresIdent[i]].copy(deep=True)
    for i in range(len(featuresIdent)): dfCalib[featuresIdentN[i]] = dfCalib[featuresIdent[i]].copy(deep=True)

    dfCalib['bdt_output'] = PUmodel.predict_proba(dfCalib[featuresIdentN])[:,1]
    dfIdent['bdt_output'] = PUmodel.predict_proba(dfIdent[featuresIdentN])[:,1]
    FPR, TPR, THR = metrics.roc_curve(dfIdent['targetId'], dfIdent['bdt_output'])
    WP99 = np.interp(0.99, TPR, THR)
    WP95 = np.interp(0.95, TPR, THR)
    WP90 = np.interp(0.90, TPR, THR)

    wp_dict = {
        'wp99' : WP99,
        'wp95' : WP95,
        'wp90' : WP90
    }

    save_obj(wp_dict, outdir+'/TauMinatorBDT_WPs.pkl')

    # rename features with numbers (needed by C2model)
    dfCalib.drop(featuresIdentN, axis=1, inplace=True)
    for i in range(len(featuresCalib)): dfCalib[featuresCalibN[i]] = dfCalib[featuresCalib[i]].copy(deep=True)

    dfCalib['cl3d_c1'] = C1model.predict(dfCalib[['cl3d_abseta']])
    dfCalib['cl3d_pt_c1'] = dfCalib['cl3d_c1'] + dfCalib['cl3d_pt']
    dfCalib['cl3d_response_c1'] = dfCalib['cl3d_pt_c1'] / dfCalib['tau_visPt']

    dfCalib['cl3d_c2'] = C2model.predict(dfCalib[featuresCalibN])
    dfCalib['cl3d_pt_c2'] = dfCalib['cl3d_c2'] * dfCalib['cl3d_pt_c1']
    dfCalib['cl3d_response_c2'] = dfCalib['cl3d_pt_c2'] / dfCalib['tau_visPt']

    logpt1 = np.log(abs(dfCalib['cl3d_pt_c2']))
    logpt2 = logpt1**2
    logpt3 = logpt1**3
    logpt4 = logpt1**4
    dfCalib['cl3d_c3'] = C3model.predict(np.vstack([logpt1, logpt2, logpt3, logpt4]).T)
    dfCalib['cl3d_pt_c3'] = dfCalib['cl3d_pt_c2'] / dfCalib['cl3d_c3']
    dfCalib['cl3d_response_c3'] = dfCalib['cl3d_pt_c3'] / dfCalib['tau_visPt']

    df = dfCalib[['tau_visPt', 'tau_visEta', 'tau_visPhi', 'tau_DM', 'cl3d_pt_c1', 'cl3d_pt_c2', 'cl3d_pt_c3', 'bdt_output']].copy(deep=True)
    df.rename(columns={'tau_visPt':'gen_pt', 'tau_visEta':'gen_eta', 'tau_visPhi':'gen_phi', 'tau_DM':'gen_dm', 'cl3d_pt_c1':'L1_pt_c1', 'cl3d_pt_c2':'L1_pt_c2', 'cl3d_pt_c3':'L1_pt_c3', 'bdt_output':'score'}, inplace=True)
    df['pass99']  = df['score'] > WP99
    df['pass95']  = df['score'] > WP95
    df['pass90']  = df['score'] > WP90

    df['gen_pt_bin'] = pd.cut(df['gen_pt'],
                              bins=[15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 200, 500],
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

    save_obj(mapping_dict, outdir+'/online2offline_mapping.pkl')


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
    plt.legend(loc = 'lower right', fontsize=16)
    plt.ylim(0., 1.05)
    plt.xlim(0., 150.)
    # plt.xscale('log')
    plt.xlabel(r'$p_{T}^{gen,\tau}\ [GeV]$')
    plt.ylabel(r'Efficiency')
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/plots/turnons_WP99.pdf')
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
    plt.legend(loc = 'lower right', fontsize=16)
    plt.ylim(0., 1.05)
    plt.xlim(0., 150.)
    # plt.xscale('log')
    plt.xlabel(r'$p_{T}^{gen,\tau}\ [GeV]$')
    plt.ylabel(r'Efficiency')
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/plots/turnons_WP95.pdf')
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
    plt.legend(loc = 'lower right', fontsize=16)
    plt.ylim(0., 1.05)
    plt.xlim(0., 150.)
    # plt.xscale('log')
    plt.xlabel(r'$p_{T}^{gen,\tau}\ [GeV]$')
    plt.ylabel(r'Efficiency')
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/plots/turnons_WP90.pdf')
    plt.close()


    ##################################################################################
    # PLOT ONLINE TO OFFLINE MAPPING
    plt.figure(figsize=(10,10))
    plt.plot(online_thresholds, mapping_dict['wp99_pt95'], label='@ 95% efficiency', linewidth=2, color='blue')
    plt.plot(online_thresholds, mapping_dict['wp99_pt90'], label='@ 90% efficiency', linewidth=2, color='red')
    plt.plot(online_thresholds, mapping_dict['wp99_pt50'], label='@ 50% efficiency', linewidth=2, color='green')
    plt.legend(loc = 'lower right', fontsize=16)
    plt.xlabel('L1 Threshold [GeV]')
    plt.ylabel('Offline threshold [GeV]')
    plt.xlim(0, 110)
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/plots/online2offline_WP99.pdf')
    plt.close()

    plt.figure(figsize=(10,10))
    plt.plot(online_thresholds, mapping_dict['wp95_pt95'], label='@ 95% efficiency', linewidth=2, color='blue')
    plt.plot(online_thresholds, mapping_dict['wp95_pt90'], label='@ 90% efficiency', linewidth=2, color='red')
    plt.plot(online_thresholds, mapping_dict['wp95_pt50'], label='@ 50% efficiency', linewidth=2, color='green')
    plt.legend(loc = 'lower right', fontsize=16)
    plt.xlabel('L1 Threshold [GeV]')
    plt.ylabel('Offline threshold [GeV]')
    plt.xlim(0, 110)
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/plots/online2offline_WP95.pdf')
    plt.close()

    plt.figure(figsize=(10,10))
    plt.plot(online_thresholds, mapping_dict['wp90_pt95'], label='@ 95% efficiency', linewidth=2, color='blue')
    plt.plot(online_thresholds, mapping_dict['wp90_pt90'], label='@ 90% efficiency', linewidth=2, color='red')
    plt.plot(online_thresholds, mapping_dict['wp90_pt50'], label='@ 50% efficiency', linewidth=2, color='green')
    plt.legend(loc = 'lower right', fontsize=16)
    plt.xlabel('L1 Threshold [GeV]')
    plt.ylabel('Offline threshold [GeV]')
    plt.xlim(0, 110)
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/plots/online2offline_WP90.pdf')
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
    plt.legend(loc = 'lower right', fontsize=16)
    plt.ylim(0., 1.05)
    plt.xlim(0., 150.)
    plt.xlabel(r'$p_{T}^{gen,\tau}\ [GeV]$')
    plt.ylabel(r'Efficiency')
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/plots/dm0_turnons_WP99.pdf')
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
    plt.legend(loc = 'lower right', fontsize=16)
    plt.ylim(0., 1.05)
    plt.xlim(0., 150.)
    plt.xlabel(r'$p_{T}^{gen,\tau}\ [GeV]$')
    plt.ylabel(r'Efficiency')
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/plots/dm1_turnons_WP99.pdf')
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
    plt.legend(loc = 'lower right', fontsize=16)
    plt.ylim(0., 1.05)
    plt.xlim(0., 150.)
    plt.xlabel(r'$p_{T}^{gen,\tau}\ [GeV]$')
    plt.ylabel(r'Efficiency')
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/plots/dm10_turnons_WP99.pdf')
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
    plt.legend(loc = 'lower right', fontsize=16)
    plt.ylim(0., 1.05)
    plt.xlim(0., 150.)
    plt.xlabel(r'$p_{T}^{gen,\tau}\ [GeV]$')
    plt.ylabel(r'Efficiency')
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(outdir+'/plots/dm11_turnons_WP99.pdf')
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
    # plt.legend(loc = 'lower right', fontsize=16)
    # plt.ylim(0., 1.05)
    # plt.xlim(0., 150.)
    # plt.xlabel(r'$p_{T}^{gen,\tau}\ [GeV]$')
    # plt.ylabel(r'Efficiency')
    # plt.grid()
    # mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    # plt.savefig(outdir+'/plots/turnons_WP99.pdf')
    # plt.close()

    # i = 0
    # plt.figure(figsize=(10,10))
    # for thr in plotting_thresholds:
    #     if not thr%10:
    #         plt.errorbar(eta_bins_centers,etaeffs_dict['efficiencyVsEtaAt95wpAt'+str(thr)+'GeV'][:,0],xerr=1,yerr=[etaeffs_dict['efficiencyVsEtaAt95wpAt'+str(thr)+'GeV'][:,1], etaeffs_dict['efficiencyVsEtaAt95wpAt'+str(thr)+'GeV'][:,2]], ls='None', label=r'$p_{T}^{L1 \tau} > %i$ GeV' % (thr), lw=2, marker='o', color=cmap(i))
    #         i+=1 
    # plt.hlines(0.90, 0, eta_bins_centers.max(), lw=2, color='dimgray', label='0.90 Eff.')
    # plt.hlines(0.95, 0, eta_bins_centers.max(), lw=2, color='black', label='0.95 Eff.')
    # plt.legend(loc = 'lower right', fontsize=16)
    # plt.ylim(0., 1.05)
    # plt.xlim(0., 150.)
    # plt.xlabel(r'$p_{T}^{gen,\tau}\ [GeV]$')
    # plt.ylabel(r'Efficiency')
    # plt.grid()
    # mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    # plt.savefig(outdir+'/plots/turnons_WP95.pdf')
    # plt.close()

    # i = 0
    # plt.figure(figsize=(10,10))
    # for thr in plotting_thresholds:
    #     if not thr%10:
    #         plt.errorbar(eta_bins_centers,etaeffs_dict['efficiencyVsEtaAt90wpAt'+str(thr)+'GeV'][:,0],xerr=1,yerr=[etaeffs_dict['efficiencyVsEtaAt90wpAt'+str(thr)+'GeV'][:,1], etaeffs_dict['efficiencyVsEtaAt90wpAt'+str(thr)+'GeV'][:,2]], ls='None', label=r'$p_{T}^{L1 \tau} > %i$ GeV' % (thr), lw=2, marker='o', color=cmap(i))
    #         i+=1 
    # plt.hlines(0.90, 0, eta_bins_centers.max(), lw=2, color='dimgray', label='0.90 Eff.')
    # plt.hlines(0.95, 0, eta_bins_centers.max(), lw=2, color='black', label='0.95 Eff.')
    # plt.legend(loc = 'lower right', fontsize=16)
    # plt.ylim(0., 1.05)
    # plt.xlim(0., 150.)
    # plt.xlabel(r'$p_{T}^{gen,\tau}\ [GeV]$')
    # plt.ylabel(r'Efficiency')
    # plt.grid()
    # mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    # plt.savefig(outdir+'/plots/turnons_WP90.pdf')
    # plt.close()
