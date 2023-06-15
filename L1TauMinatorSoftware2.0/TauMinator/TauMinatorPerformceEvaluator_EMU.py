from scipy.optimize import curve_fit
from optparse import OptionParser
from array import array
import numpy as np
import pickle
import ROOT
import sys
import os

import matplotlib.pyplot as plt
import matplotlib
import mplhep
plt.style.use(mplhep.style.CMS)

class Logger(object):
    def __init__(self,file):
        self.terminal = sys.stdout
        self.log = open(file, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        #this flush method is needed for python 3 compatibility.
        #this handles the flush command by doing nothing.
        #you might want to specify some extra behavior here.
        pass

def save_obj(obj,dest):
    with open(dest,'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(source):
    with open(source,'rb') as f:
        return pickle.load(f)

def sigmoid(x , a, x0, k):
    return a / ( 1 + np.exp(-k*(x-x0)) )


#######################################################################
######################### SCRIPT BODY #################################
#######################################################################

if __name__ == "__main__" :
    parser = OptionParser()
    parser.add_option("--NtupleV",          dest="NtupleV",        default=None)
    parser.add_option("--v",                dest="v",              default=None)
    parser.add_option("--date",             dest="date",           default=None)
    parser.add_option('--loop',             dest='loop',           default=False, action="store_true")
    parser.add_option('--etaEr',            dest='etaEr',          default=3.0,   type=float)
    parser.add_option('--WP',               dest='WP',             default='90')
    parser.add_option('--caloClNxM',        dest='caloClNxM',      default="5x9")
    parser.add_option("--seedEtCut",        dest="seedEtCut",      default="2p5")
    parser.add_option("--inTag",            dest="inTag",          default="")
    (options, args) = parser.parse_args()
    print(options)

    ptBins=[0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 175, 200, 500]
    offline_pts = [2.5, 7.5, 12.5, 17.5,  22.5, 27.5, 32.5, 37.5, 42.5, 47.5, 52.5, 57.5, 62.5, 67.5, 72.5, 77.5, 82.5, 87.5, 92.5, 97.5, 102.5, 107.5, 112.5, 117.5, 122.5, 127.5, 132.5, 137.5, 142.5, 147.5, 152.5, 157.5, 170, 187.5, 350]

    if options.etaEr==3.0:
        etaBins=[-3.0, -2.7, -2.4, -2.1, -1.8, -1.5, -1.305, -1.0, -0.66, -0.33, 0.0, 0.33, 0.66, 1.0, 1.305, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0]
        eta_bins_centers = [-2.85, -2.55, -2.25, -1.95, -1.65, -1.4025, -1.1525, -0.825, -0.495, -0.165, 0.165, 0.495, 0.825, 1.1525, 1.4025, 1.65, 1.95, 2.25, 2.55, 2.85]
    elif options.etaEr==2.4:
        etaBins=[-2.4, -2.1, -1.8, -1.5, -1.305, -1.0, -0.66, -0.33, 0.0, 0.33, 0.66, 1.0, 1.305, 1.5, 1.8, 2.1, 2.4]
        eta_bins_centers = [-2.25, -1.95, -1.65, -1.4025, -1.1525, -0.825, -0.495, -0.165, 0.165, 0.495, 0.825, 1.1525, 1.4025, 1.65, 1.95, 2.25]
    else:
        exit()

    online_thresholds = range(0, 175, 1)
    plotting_thresholds = range(0, 110, 10)

    perfdir = '/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauMinatorPerformanceEvaluator'+options.inTag
    tag = '_Er'+str(options.etaEr).split('.')[0]+'p'+str(options.etaEr).split('.')[1]
    os.system('mkdir -p '+perfdir+'/turnons'+tag)

    WP = options.WP
    qualityCut = 0
    if WP == "99": qualityCut = 1
    if WP == "95": qualityCut = 2
    if WP == "90": qualityCut = 3

    if options.loop:
        #passing histograms (numerators)
        minated00_passing_ptBins = []
        minated00_passing_ptBins_CB = []
        minated00_passing_ptBins_CE = []

        minated99_passing_etaBins = []

        for threshold in online_thresholds:
            minated00_passing_ptBins.append(ROOT.TH1F("minated00_passing_thr"+str(int(threshold))+"_ptBins","minated00_passing_thr"+str(int(threshold))+"_ptBins",len(ptBins)-1, array('f',ptBins)))
            minated00_passing_ptBins_CB.append(ROOT.TH1F("minated00_passing_thr"+str(int(threshold))+"_ptBins_CB","minated00_passing_thr"+str(int(threshold))+"_ptBins_CB",len(ptBins)-1, array('f',ptBins)))
            minated00_passing_ptBins_CE.append(ROOT.TH1F("minated00_passing_thr"+str(int(threshold))+"_ptBins_CE","minated00_passing_thr"+str(int(threshold))+"_ptBins_CE",len(ptBins)-1, array('f',ptBins)))
            
            minated99_passing_etaBins.append(ROOT.TH1F("minated99_passing_thr"+str(int(threshold))+"_etaBins","minated99_passing_thr"+str(int(threshold))+"_etaBins",len(etaBins)-1, array('f',etaBins)))

        #denominator
        denominator_ptBins = ROOT.TH1F("denominator_ptBins","denominator_ptBins",len(ptBins)-1, array('f',ptBins))
        denominator_ptBins_CB = ROOT.TH1F("denominator_ptBins_CB","denominator_ptBins_CB",len(ptBins)-1, array('f',ptBins))
        denominator_ptBins_CE = ROOT.TH1F("denominator_ptBins_CE","denominator_ptBins_CE",len(ptBins)-1, array('f',ptBins))
        denominator_etaBins = ROOT.TH1F("denominator_etaBins","denominator_etaBins",len(etaBins)-1, array('f',etaBins))

        tot = 0

        # loop over the events to fill all the histograms
        directory = '/data_CMS/cms/motta/Phase2L1T/L1TauMinatorNtuples/v'+options.NtupleV+'/GluGluToHHTo2B2Tau_node_SM_TuneCP5_14TeV-madgraph-pythia8__Phase2Fall22DRMiniAOD-PU200_125X_mcRun4_realistic_v2-v1__GEN-SIM-DIGI-RAW-MINIAOD_seedEtCut'+options.seedEtCut+options.inTag+'/'
        inChain = ROOT.TChain("L1CaloTauNtuplizerRealEmuTest/L1TauMinatorTree");
        inChain.Add(directory+'/Ntuple_*.root');
        nEntries = inChain.GetEntries()
        for evt in range(0, nEntries):
            if evt%1000==0: print('--> ',evt)
            # if evt == 20000: break

            entry = inChain.GetEntry(evt)

            _gentau_visEta = inChain.tau_visEta
            if len(_gentau_visEta)==0: continue
            _gentau_visPhi = inChain.tau_visPhi
            _gentau_visPt = inChain.tau_visPt

            _l1tau_pt = list(inChain.minatedl1tau_pt)
            _l1tau_eta = list(inChain.minatedl1tau_eta)
            _l1tau_phi = list(inChain.minatedl1tau_phi)
            _l1tau_quality = list(inChain.minatedl1tau_quality)

            for tauPt, tauEta, tauPhi in zip(_gentau_visPt, _gentau_visEta, _gentau_visPhi):

                gentau = ROOT.TLorentzVector()
                gentau.SetPtEtaPhiM(tauPt, tauEta, tauPhi, 0)

                if abs(gentau.Eta()) > options.etaEr: continue # skip taus out of acceptance

                denominator_ptBins.Fill(gentau.Pt())
                if abs(gentau.Eta())<1.5:
                    denominator_ptBins_CB.Fill(gentau.Pt())
                else:
                    denominator_ptBins_CE.Fill(gentau.Pt())
                if gentau.Pt() > 40: denominator_etaBins.Fill(gentau.Eta())

                minatedMatched = False
                highestMinatedL1Pt = -99.9
                highestMinatedL1Id = 0.0
                highestMinatedL1isBarrel = False

                # loop over TauMinator taus
                for l1tauPt, l1tauEta, l1tauPhi, l1tauId in zip(_l1tau_pt, _l1tau_eta, _l1tau_phi, _l1tau_quality):

                    l1tau = ROOT.TLorentzVector()
                    l1tau.SetPtEtaPhiM(l1tauPt, l1tauEta, l1tauPhi, 0)

                    if abs(l1tau.Eta()) > options.etaEr: continue # skip taus out of acceptance
                    if l1tauId < qualityCut: continue # skip non identified objects

                    # check matching
                    if gentau.DeltaR(l1tau)<0.3:
                        # keep only L1 match with highest pT
                        if l1tau.Pt()>highestMinatedL1Pt:
                            minatedMatched = True
                            highestMinatedL1Pt = l1tau.Pt()
                            highestMinatedL1Eta = l1tau.Eta()

                # fill numerator histograms for every thresholds
                for i, thr in enumerate(online_thresholds): 
                    if minatedMatched and highestMinatedL1Pt>float(thr):

                        minated00_passing_ptBins[i].Fill(gentau.Pt())
                        if gentau.Pt() > 40: minated99_passing_etaBins[i].Fill(gentau.Eta())

                        if abs(highestMinatedL1Eta)<1.5 and abs(gentau.Eta())<1.5:
                            minated00_passing_ptBins_CB[i].Fill(gentau.Pt())

                        elif abs(highestMinatedL1Eta)>1.5 and abs(gentau.Eta())>1.5:
                            minated00_passing_ptBins_CE[i].Fill(gentau.Pt())

                        # if abs(gentau.Eta())<1.5:
                        #     minated00_passing_ptBins_CB[i].Fill(gentau.Pt())

                        # elif abs(gentau.Eta())>1.5:
                        #     minated00_passing_ptBins_CE[i].Fill(gentau.Pt())


        # end of the loop over the events
        #################################

        # TGraphAsymmErrors for efficiency turn-ons
        turnonsMinated00 = []
        turnonsMinated00_CB = []
        turnonsMinated00_CE = []
        etaEffMinated99 = []

        for i, thr in enumerate(online_thresholds):
            turnonsMinated00.append(ROOT.TGraphAsymmErrors(minated00_passing_ptBins[i], denominator_ptBins, "cp"))
            turnonsMinated00_CB.append(ROOT.TGraphAsymmErrors(minated00_passing_ptBins_CB[i], denominator_ptBins_CB, "cp"))
            turnonsMinated00_CE.append(ROOT.TGraphAsymmErrors(minated00_passing_ptBins_CE[i], denominator_ptBins_CE, "cp"))

            etaEffMinated99.append(ROOT.TGraphAsymmErrors(minated99_passing_etaBins[i], denominator_etaBins, "cp"))

        # save to file 

        fileout = ROOT.TFile(perfdir+"/turnons"+tag+"/efficiency_graphs"+tag+"_er"+str(options.etaEr)+".root","RECREATE")
        denominator_ptBins.Write()
        denominator_ptBins_CB.Write()
        denominator_ptBins_CE.Write()
        denominator_etaBins.Write()
        for i, thr in enumerate(online_thresholds): 
            minated00_passing_ptBins[i].Write()
            minated00_passing_ptBins_CB[i].Write()
            minated00_passing_ptBins_CE[i].Write()

            minated99_passing_etaBins[i].Write()

            turnonsMinated00[i].Write()
            turnonsMinated00_CB[i].Write()
            turnonsMinated00_CE[i].Write()

            etaEffMinated99[i].Write()


        fileout.Close()

    else:
        filein = ROOT.TFile(perfdir+"/turnons"+tag+"/efficiency_graphs"+tag+"_er"+str(options.etaEr)+".root","READ")

        # TGraphAsymmErrors for efficiency turn-ons
        turnonsMinated00 = []
        turnonsMinated00_CB = []
        turnonsMinated00_CE = []

        etaEffMinated99 = []

        for i, thr in enumerate(online_thresholds):
            turnonsMinated00.append(filein.Get("divide_minated00_passing_thr"+str(int(thr))+"_ptBins_by_denominator_ptBins"))
            turnonsMinated00_CB.append(filein.Get("divide_minated00_passing_thr"+str(int(thr))+"_ptBins_CB_by_denominator_ptBins_CB"))
            turnonsMinated00_CE.append(filein.Get("divide_minated00_passing_thr"+str(int(thr))+"_ptBins_CE_by_denominator_ptBins_CE"))

            etaEffMinated99.append(filein.Get("divide_minated99_passing_thr"+str(int(thr))+"_etaBins_by_denominator_etaBins"))


    turnons_dict = {}
    turnons_dict_CB = {}
    turnons_dict_CE = {}
    etaeffs_dict = {}
    mapping_dict = {'threshold':[],
                    'wp99_pt95':[], 'wp99_pt90':[], 'wp99_pt50':[],
                    'wp95_pt95':[], 'wp95_pt90':[], 'wp95_pt50':[],
                    'wp90_pt95':[], 'wp90_pt90':[], 'wp90_pt50':[]}
    for i, thr in enumerate(online_thresholds):
        turnons_dict['turnonAt00wpAt'+str(thr)+'GeV'] = [[],[],[]]
        turnons_dict_CB['turnonAt00wpAt'+str(thr)+'GeV'] = [[],[],[]]
        turnons_dict_CE['turnonAt00wpAt'+str(thr)+'GeV'] = [[],[],[]]

        etaeffs_dict['efficiencyVsEtaAt99wpAt'+str(thr)+'GeV'] = [[],[],[]]

        for ibin in range(0,turnonsMinated00[i].GetN()):
            turnons_dict['turnonAt00wpAt'+str(thr)+'GeV'][0].append(turnonsMinated00[i].GetPointY(ibin))
            turnons_dict['turnonAt00wpAt'+str(thr)+'GeV'][1].append(turnonsMinated00[i].GetErrorYlow(ibin))
            turnons_dict['turnonAt00wpAt'+str(thr)+'GeV'][2].append(turnonsMinated00[i].GetErrorYhigh(ibin))
            turnons_dict_CB['turnonAt00wpAt'+str(thr)+'GeV'][0].append(turnonsMinated00_CB[i].GetPointY(ibin))
            turnons_dict_CB['turnonAt00wpAt'+str(thr)+'GeV'][1].append(turnonsMinated00_CB[i].GetErrorYlow(ibin))
            turnons_dict_CB['turnonAt00wpAt'+str(thr)+'GeV'][2].append(turnonsMinated00_CB[i].GetErrorYhigh(ibin))
            turnons_dict_CE['turnonAt00wpAt'+str(thr)+'GeV'][0].append(turnonsMinated00_CE[i].GetPointY(ibin))
            turnons_dict_CE['turnonAt00wpAt'+str(thr)+'GeV'][1].append(turnonsMinated00_CE[i].GetErrorYlow(ibin))
            turnons_dict_CE['turnonAt00wpAt'+str(thr)+'GeV'][2].append(turnonsMinated00_CE[i].GetErrorYhigh(ibin))

        for ibin in range(0,etaEffMinated99[i].GetN()):
            etaeffs_dict['efficiencyVsEtaAt99wpAt'+str(thr)+'GeV'][0].append(etaEffMinated99[i].GetPointY(ibin))
            etaeffs_dict['efficiencyVsEtaAt99wpAt'+str(thr)+'GeV'][1].append(etaEffMinated99[i].GetErrorYlow(ibin))
            etaeffs_dict['efficiencyVsEtaAt99wpAt'+str(thr)+'GeV'][2].append(etaEffMinated99[i].GetErrorYhigh(ibin))
        
        # ONLINE TO OFFILNE MAPPING
        mapping_dict['wp'+WP+'_pt95'].append(np.interp(0.95, turnons_dict['turnonAt00wpAt'+str(thr)+'GeV'][0], offline_pts)) #,right=-99,left=-98)
        mapping_dict['wp'+WP+'_pt90'].append(np.interp(0.90, turnons_dict['turnonAt00wpAt'+str(thr)+'GeV'][0], offline_pts)) #,right=-99,left=-98)
        mapping_dict['wp'+WP+'_pt50'].append(np.interp(0.50, turnons_dict['turnonAt00wpAt'+str(thr)+'GeV'][0], offline_pts)) #,right=-99,left=-98)

    save_obj(mapping_dict, perfdir+'/turnons'+tag+'/online2offline_mapping'+WP+'.pkl')


    i = 0
    plt.figure(figsize=(10,10))
    thr=0
    plt.errorbar(offline_pts,turnons_dict_CB['turnonAt00wpAt'+str(thr)+'GeV'][0],xerr=1,yerr=[turnons_dict_CB['turnonAt00wpAt'+str(thr)+'GeV'][1], turnons_dict_CB['turnonAt00wpAt'+str(thr)+'GeV'][2]], ls='-', label=r'CB - $p_{T}^{L1 \tau} > %i$ GeV' % (thr), lw=2, marker='o', color='green')
    plt.errorbar(offline_pts,turnons_dict_CE['turnonAt00wpAt'+str(thr)+'GeV'][0],xerr=1,yerr=[turnons_dict_CE['turnonAt00wpAt'+str(thr)+'GeV'][1], turnons_dict_CE['turnonAt00wpAt'+str(thr)+'GeV'][2]], ls='-', label=r'CE - $p_{T}^{L1 \tau} > %i$ GeV' % (thr), lw=2, marker='o', color='red')
    plt.hlines(0.90, 0, 2000, lw=2, color='dimgray', label='0.90 Eff.')
    plt.hlines(0.95, 0, 2000, lw=2, color='black', label='0.95 Eff.')
    plt.legend(loc = 'lower right', fontsize=14)
    plt.ylim(0., 1.05)
    plt.xlim(0., 160.)
    # plt.xscale('log')
    plt.xlabel(r'$p_{T}^{gen,\tau}\ [GeV]$')
    plt.ylabel(r'Efficiency')
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(perfdir+'/turnons'+tag+'/turnons_Emyr_'+WP+'.pdf')
    plt.close()


    cmap = matplotlib.cm.get_cmap('tab20c'); i=0
    ##################################################################################
    # PLOT TURNONS
    i = 0
    plt.figure(figsize=(10,10))
    for thr in plotting_thresholds:
        if not thr%10:
            plt.errorbar(offline_pts,turnons_dict['turnonAt00wpAt'+str(thr)+'GeV'][0],xerr=1,yerr=[turnons_dict['turnonAt00wpAt'+str(thr)+'GeV'][1], turnons_dict['turnonAt00wpAt'+str(thr)+'GeV'][2]], ls='None', label=r'$p_{T}^{L1 \tau} > %i$ GeV' % (thr), lw=2, marker='o', color=cmap(i))

            p0 = [1, thr, 1] 
            popt, pcov = curve_fit(sigmoid, offline_pts, turnons_dict['turnonAt00wpAt'+str(thr)+'GeV'][0], p0, maxfev=5000)
            plt.plot(offline_pts, sigmoid(offline_pts, *popt), '-', label='_', lw=1.5, color=cmap(i))

            i+=1 
    plt.hlines(0.90, 0, 2000, lw=2, color='dimgray', label='0.90 Eff.')
    plt.hlines(0.95, 0, 2000, lw=2, color='black', label='0.95 Eff.')
    plt.legend(loc = 'lower right', fontsize=14)
    plt.ylim(0., 1.05)
    plt.xlim(0., 160.)
    # plt.xscale('log')
    plt.xlabel(r'$p_{T}^{gen,\tau}\ [GeV]$')
    plt.ylabel(r'Efficiency')
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(perfdir+'/turnons'+tag+'/turnons_WP'+WP+'.pdf')
    plt.close()

    ##################################################################################
    # PLOT TURNONS CB
    i = 0
    plt.figure(figsize=(10,10))
    for thr in plotting_thresholds:
        if not thr%10:
            plt.errorbar(offline_pts,turnons_dict_CB['turnonAt00wpAt'+str(thr)+'GeV'][0],xerr=1,yerr=[turnons_dict_CB['turnonAt00wpAt'+str(thr)+'GeV'][1], turnons_dict_CB['turnonAt00wpAt'+str(thr)+'GeV'][2]], ls='None', label=r'$p_{T}^{L1 \tau} > %i$ GeV' % (thr), lw=2, marker='o', color=cmap(i))

            p0 = [1, thr, 1] 
            popt, pcov = curve_fit(sigmoid, offline_pts, turnons_dict_CB['turnonAt00wpAt'+str(thr)+'GeV'][0], p0, maxfev=5000)
            plt.plot(offline_pts, sigmoid(offline_pts, *popt), '-', label='_', lw=1.5, color=cmap(i))

            i+=1 
    plt.hlines(0.90, 0, 2000, lw=2, color='dimgray', label='0.90 Eff.')
    plt.hlines(0.95, 0, 2000, lw=2, color='black', label='0.95 Eff.')
    plt.legend(loc = 'lower right', fontsize=14)
    plt.ylim(0., 1.05)
    plt.xlim(0., 160.)
    # plt.xscale('log')
    plt.xlabel(r'$p_{T}^{gen,\tau}\ [GeV]$')
    plt.ylabel(r'Efficiency')
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(perfdir+'/turnons'+tag+'/turnons_WP'+WP+'_CB.pdf')
    plt.close()

    ##################################################################################
    # PLOT TURNONS CE
    i = 0
    plt.figure(figsize=(10,10))
    for thr in plotting_thresholds:
        if not thr%10:
            plt.errorbar(offline_pts,turnons_dict_CE['turnonAt00wpAt'+str(thr)+'GeV'][0],xerr=1,yerr=[turnons_dict_CE['turnonAt00wpAt'+str(thr)+'GeV'][1], turnons_dict_CE['turnonAt00wpAt'+str(thr)+'GeV'][2]], ls='None', label=r'$p_{T}^{L1 \tau} > %i$ GeV' % (thr), lw=2, marker='o', color=cmap(i))

            p0 = [1, thr, 1] 
            popt, pcov = curve_fit(sigmoid, offline_pts, turnons_dict_CE['turnonAt00wpAt'+str(thr)+'GeV'][0], p0, maxfev=5000)
            plt.plot(offline_pts, sigmoid(offline_pts, *popt), '-', label='_', lw=1.5, color=cmap(i))

            i+=1 
    plt.hlines(0.90, 0, 2000, lw=2, color='dimgray', label='0.90 Eff.')
    plt.hlines(0.95, 0, 2000, lw=2, color='black', label='0.95 Eff.')
    plt.legend(loc = 'lower right', fontsize=14)
    plt.ylim(0., 1.05)
    plt.xlim(0., 160.)
    # plt.xscale('log')
    plt.xlabel(r'$p_{T}^{gen,\tau}\ [GeV]$')
    plt.ylabel(r'Efficiency')
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(perfdir+'/turnons'+tag+'/turnons_WP'+WP+'_CE.pdf')
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
    plt.xlim(20, 100)
    plt.ylim(20, 200)
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(perfdir+'/turnons'+tag+'/online2offline_WP'+WP+'.pdf')
    plt.close()

    # ##################################################################################
    # # PLOT TURNONS PER DM
    # i = 0
    # plt.figure(figsize=(10,10))
    # for thr in plotting_thresholds:
    #     if not thr%10:
    #         plt.errorbar(offline_pts,turnons_dm_dict['dm0TurnonAt99wpAt'+str(thr)+'GeV'][0],xerr=1,yerr=[turnons_dm_dict['dm0TurnonAt99wpAt'+str(thr)+'GeV'][1], turnons_dm_dict['dm0TurnonAt99wpAt'+str(thr)+'GeV'][2]], ls='None', label=r'$p_{T}^{L1 \tau} > %i$ GeV' % (thr), lw=2, marker='o', color=cmap(i))

    #         p0 = [1, thr, 1] 
    #         popt, pcov = curve_fit(sigmoid, offline_pts, turnons_dm_dict['dm0TurnonAt99wpAt'+str(thr)+'GeV'][0], p0, maxfev=5000)
    #         plt.plot(offline_pts, sigmoid(offline_pts, *popt), '-', label='_', lw=1.5, color=cmap(i))

    #         i+=1 
    # plt.hlines(0.90, 0, 2000, lw=2, color='dimgray', label='0.90 Eff.')
    # plt.hlines(0.95, 0, 2000, lw=2, color='black', label='0.95 Eff.')
    # plt.legend(loc = 'lower right', fontsize=14)
    # plt.ylim(0., 1.05)
    # plt.xlim(0., 150.)
    # plt.xlabel(r'$p_{T}^{gen,\tau}\ [GeV]$')
    # plt.ylabel(r'Efficiency')
    # plt.grid()
    # mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    # plt.savefig(perfdir+'/turnons'+tag+'/dm0_turnons_WP99.pdf')
    # plt.close()

    # i = 0
    # plt.figure(figsize=(10,10))
    # for thr in plotting_thresholds:
    #     if not thr%10:
    #         plt.errorbar(offline_pts,turnons_dm_dict['dm1TurnonAt99wpAt'+str(thr)+'GeV'][0],xerr=1,yerr=[turnons_dm_dict['dm1TurnonAt99wpAt'+str(thr)+'GeV'][1], turnons_dm_dict['dm1TurnonAt99wpAt'+str(thr)+'GeV'][2]], ls='None', label=r'$p_{T}^{L1 \tau} > %i$ GeV' % (thr), lw=2, marker='o', color=cmap(i))

    #         p0 = [1, thr, 1] 
    #         popt, pcov = curve_fit(sigmoid, offline_pts, turnons_dm_dict['dm1TurnonAt99wpAt'+str(thr)+'GeV'][0], p0, maxfev=5000)
    #         plt.plot(offline_pts, sigmoid(offline_pts, *popt), '-', label='_', lw=1.5, color=cmap(i))

    #         i+=1 
    # plt.hlines(0.90, 0, 2000, lw=2, color='dimgray', label='0.90 Eff.')
    # plt.hlines(0.95, 0, 2000, lw=2, color='black', label='0.95 Eff.')
    # plt.legend(loc = 'lower right', fontsize=14)
    # plt.ylim(0., 1.05)
    # plt.xlim(0., 150.)
    # plt.xlabel(r'$p_{T}^{gen,\tau}\ [GeV]$')
    # plt.ylabel(r'Efficiency')
    # plt.grid()
    # mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    # plt.savefig(perfdir+'/turnons'+tag+'/dm1_turnons_WP99.pdf')
    # plt.close()

    # i = 0
    # plt.figure(figsize=(10,10))
    # for thr in plotting_thresholds:
    #     if not thr%10:
    #         plt.errorbar(offline_pts,turnons_dm_dict['dm10TurnonAt99wpAt'+str(thr)+'GeV'][0],xerr=1,yerr=[turnons_dm_dict['dm10TurnonAt99wpAt'+str(thr)+'GeV'][1], turnons_dm_dict['dm10TurnonAt99wpAt'+str(thr)+'GeV'][2]], ls='None', label=r'$p_{T}^{L1 \tau} > %i$ GeV' % (thr), lw=2, marker='o', color=cmap(i))

    #         p0 = [1, thr, 1] 
    #         popt, pcov = curve_fit(sigmoid, offline_pts, turnons_dm_dict['dm10TurnonAt99wpAt'+str(thr)+'GeV'][0], p0, maxfev=5000)
    #         plt.plot(offline_pts, sigmoid(offline_pts, *popt), '-', label='_', lw=1.5, color=cmap(i))

    #         i+=1 
    # plt.hlines(0.90, 0, 2000, lw=2, color='dimgray', label='0.90 Eff.')
    # plt.hlines(0.95, 0, 2000, lw=2, color='black', label='0.95 Eff.')
    # plt.legend(loc = 'lower right', fontsize=14)
    # plt.ylim(0., 1.05)
    # plt.xlim(0., 150.)
    # plt.xlabel(r'$p_{T}^{gen,\tau}\ [GeV]$')
    # plt.ylabel(r'Efficiency')
    # plt.grid()
    # mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    # plt.savefig(perfdir+'/turnons'+tag+'/dm10_turnons_WP99.pdf')
    # plt.close()

    # i = 0
    # plt.figure(figsize=(10,10))
    # for thr in plotting_thresholds:
    #     if not thr%10:
    #         plt.errorbar(offline_pts,turnons_dm_dict['dm11TurnonAt99wpAt'+str(thr)+'GeV'][0],xerr=1,yerr=[turnons_dm_dict['dm11TurnonAt99wpAt'+str(thr)+'GeV'][1], turnons_dm_dict['dm11TurnonAt99wpAt'+str(thr)+'GeV'][2]], ls='None', label=r'$p_{T}^{L1 \tau} > %i$ GeV' % (thr), lw=2, marker='o', color=cmap(i))

    #         p0 = [1, thr, 1] 
    #         popt, pcov = curve_fit(sigmoid, offline_pts, turnons_dm_dict['dm11TurnonAt99wpAt'+str(thr)+'GeV'][0], p0, maxfev=5000)
    #         plt.plot(offline_pts, sigmoid(offline_pts, *popt), '-', label='_', lw=1.5, color=cmap(i))

    #         i+=1 
    # plt.hlines(0.90, 0, 2000, lw=2, color='dimgray', label='0.90 Eff.')
    # plt.hlines(0.95, 0, 2000, lw=2, color='black', label='0.95 Eff.')
    # plt.legend(loc = 'lower right', fontsize=14)
    # plt.ylim(0., 1.05)
    # plt.xlim(0., 150.)
    # plt.xlabel(r'$p_{T}^{gen,\tau}\ [GeV]$')
    # plt.ylabel(r'Efficiency')
    # plt.grid()
    # mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    # plt.savefig(perfdir+'/turnons'+tag+'/dm11_turnons_WP99.pdf')
    # plt.close()


    ##################################################################################
    # PLOT EFFICIENCIES VS ETA
    i = 0
    plt.figure(figsize=(10,10))
    plt.errorbar(eta_bins_centers,etaeffs_dict['efficiencyVsEtaAt99wpAt32GeV'][0],yerr=[etaeffs_dict['efficiencyVsEtaAt99wpAt32GeV'][1], etaeffs_dict['efficiencyVsEtaAt99wpAt32GeV'][2]], ls='None', label=r'$p_{T}^{L1 \tau} > %i$ GeV' % (32), lw=2, marker='o', color=cmap(i))
    plt.hlines(0.90, -3.0, 3.0, lw=2, color='dimgray', label='0.90 Eff.')
    plt.hlines(0.95, -3.0, 3.0, lw=2, color='black', label='0.95 Eff.')
    plt.legend(loc = 'lower right', fontsize=14)
    plt.ylim(0., 1.05)
    plt.xlim(-3.0, 3.0)
    plt.xlabel(r'$\eta^{gen,\tau}$')
    plt.ylabel(r'Efficiency')
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(perfdir+'/turnons'+tag+'/efficiencyVSeta_WP'+WP+'.pdf')
    plt.close()

 