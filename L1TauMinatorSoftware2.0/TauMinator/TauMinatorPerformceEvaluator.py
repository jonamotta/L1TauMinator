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
    parser.add_option('--caloClNxM',        dest='caloClNxM',      default="5x9")
    parser.add_option("--seedEtCut",        dest="seedEtCut",      default="2p5")
    parser.add_option("--inTag",            dest="inTag",          default="")
    (options, args) = parser.parse_args()
    print(options)

    ptBins=[15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 175, 200, 500]
    offline_pts = [17.5,  22.5, 27.5, 32.5, 37.5, 42.5, 47.5, 52.5, 57.5, 62.5, 67.5, 72.5, 77.5, 82.5, 87.5, 92.5, 97.5, 102.5, 107.5, 112.5, 117.5, 122.5, 127.5, 132.5, 137.5, 142.5, 147.5, 152.5, 157.5, 170, 187.5, 350]

    if options.etaEr==3.0:
        etaBins=[-3.0, -2.7, -2.4, -2.1, -1.8, -1.5, -1.305, -1.0, -0.66, -0.33, 0.0, 0.33, 0.66, 1.0, 1.305, 1.5, 1.8, 2.1, 2.4, 2.7, 3.0]
        eta_bins_centers = [-2.85, -2.55, -2.25, -1.95, -1.65, -1.4025, -1.1525, -0.825, -0.495, -0.165, 0.165, 0.495, 0.825, 1.1525, 1.4025, 1.65, 1.95, 2.25, 2.55, 2.85]
    elif options.etaEr==2.4:
        etaBins=[-2.4, -2.1, -1.8, -1.5, -1.305, -1.0, -0.66, -0.33, 0.0, 0.33, 0.66, 1.0, 1.305, 1.5, 1.8, 2.1, 2.4]
        eta_bins_centers = [-2.25, -1.95, -1.65, -1.4025, -1.1525, -0.825, -0.495, -0.165, 0.165, 0.495, 0.825, 1.1525, 1.4025, 1.65, 1.95, 2.25]
    else:
        exit()

    online_thresholds = range(20, 175, 1)
    plotting_thresholds = range(20, 110, 10)

    perfdir = '/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauMinatorPerformanceEvaluator'+options.inTag
    tag = '_Er'+str(options.etaEr).split('.')[0]+'p'+str(options.etaEr).split('.')[1]
    os.system('mkdir -p '+perfdir+'/turnons'+tag)

    if options.loop:
        #passing histograms (numerators)
        minated99_passing_ptBins = []
        minated95_passing_ptBins = []
        minated90_passing_ptBins = []
        minated85_passing_ptBins = []
        minated80_passing_ptBins = []
        minated75_passing_ptBins = []
        # square_passing_ptBins = []
        minated99_passing_etaBins = []
        minated95_passing_etaBins = []
        minated90_passing_etaBins = []
        # square_passing_etaBins = []
        for threshold in online_thresholds:
            minated99_passing_ptBins.append(ROOT.TH1F("minated99_passing_thr"+str(int(threshold))+"_ptBins","minated99_passing_thr"+str(int(threshold))+"_ptBins",len(ptBins)-1, array('f',ptBins)))
            minated95_passing_ptBins.append(ROOT.TH1F("minated95_passing_thr"+str(int(threshold))+"_ptBins","minated95_passing_thr"+str(int(threshold))+"_ptBins",len(ptBins)-1, array('f',ptBins)))
            minated90_passing_ptBins.append(ROOT.TH1F("minated90_passing_thr"+str(int(threshold))+"_ptBins","minated90_passing_thr"+str(int(threshold))+"_ptBins",len(ptBins)-1, array('f',ptBins)))
            minated85_passing_ptBins.append(ROOT.TH1F("minated85_passing_thr"+str(int(threshold))+"_ptBins","minated85_passing_thr"+str(int(threshold))+"_ptBins",len(ptBins)-1, array('f',ptBins)))
            minated80_passing_ptBins.append(ROOT.TH1F("minated80_passing_thr"+str(int(threshold))+"_ptBins","minated80_passing_thr"+str(int(threshold))+"_ptBins",len(ptBins)-1, array('f',ptBins)))
            minated75_passing_ptBins.append(ROOT.TH1F("minated75_passing_thr"+str(int(threshold))+"_ptBins","minated75_passing_thr"+str(int(threshold))+"_ptBins",len(ptBins)-1, array('f',ptBins)))
            # square_passing_ptBins.append(ROOT.TH1F("square_passing_thr"+str(int(threshold))+"_ptBins","square_passing_thr"+str(int(threshold))+"_ptBins",len(ptBins)-1, array('f',ptBins)))

            minated99_passing_etaBins.append(ROOT.TH1F("minated99_passing_thr"+str(int(threshold))+"_etaBins","minated99_passing_thr"+str(int(threshold))+"_etaBins",len(etaBins)-1, array('f',etaBins)))
            minated95_passing_etaBins.append(ROOT.TH1F("minated95_passing_thr"+str(int(threshold))+"_etaBins","minated95_passing_thr"+str(int(threshold))+"_etaBins",len(etaBins)-1, array('f',etaBins)))
            minated90_passing_etaBins.append(ROOT.TH1F("minated90_passing_thr"+str(int(threshold))+"_etaBins","minated90_passing_thr"+str(int(threshold))+"_etaBins",len(etaBins)-1, array('f',etaBins)))
            # square_passing_etaBins.append(ROOT.TH1F("square_passing_thr"+str(int(threshold))+"_etaBins","square_passing_thr"+str(int(threshold))+"_etaBins",len(etaBins)-1, array('f',etaBins)))

        #denominator
        denominator_ptBins = ROOT.TH1F("denominator_ptBins","denominator_ptBins",len(ptBins)-1, array('f',ptBins))
        denominator_etaBins = ROOT.TH1F("denominator_etaBins","denominator_etaBins",len(etaBins)-1, array('f',etaBins))

        # working points
        idWp_CB = load_obj('/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauMinator_CB_cltw'+options.caloClNxM+'_Training/TauMinator_CB_ident_plots/CLTW_TauIdentifier_WPs.pkl')
        idWp_CE = load_obj('/data_CMS/cms/motta/Phase2L1T/'+options.date+'_v'+options.v+'/TauMinator_CE_cltw'+options.caloClNxM+'_Training/TauMinator_CE_ident_plots/CLTW_TauIdentifier_WPs.pkl')

        tot = 0

        # loop over the events to fill all the histograms
        directory = '/data_CMS/cms/motta/Phase2L1T/L1TauMinatorNtuples/v'+options.NtupleV+'/GluGluToHHTo2B2Tau_node_SM_TuneCP5_14TeV-madgraph-pythia8__Phase2Fall22DRMiniAOD-PU200_125X_mcRun4_realistic_v2-v1__GEN-SIM-DIGI-RAW-MINIAOD_seedEtCut'+options.seedEtCut+options.inTag+'/'
        inChain = ROOT.TChain("L1CaloTauNtuplizer/L1TauMinatorTree");
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
            _l1tau_IDscore = list(inChain.minatedl1tau_IDscore)

            # _squarel1tau_pt = list(inChain.squarel1tau_pt)
            # _squarel1tau_eta = list(inChain.squarel1tau_eta)
            # _squarel1tau_phi = list(inChain.squarel1tau_phi)
            # _squarel1tau_iso = list(inChain.squarel1tau_qual) # iso is called quality in the ntuples of square taus

            for tauPt, tauEta, tauPhi in zip(_gentau_visPt, _gentau_visEta, _gentau_visPhi):
                if abs(tauEta) > options.etaEr: continue # skip taus out of acceptance

                denominator_ptBins.Fill(tauPt)
                if tauPt > 40: denominator_etaBins.Fill(tauEta)

                gentau = ROOT.TLorentzVector()
                gentau.SetPtEtaPhiM(tauPt, tauEta, tauPhi, 0)

                minatedMatched = False
                highestMinatedL1Pt = -99.9
                highestMinatedL1Id = 0.0
                highestMinatedL1isBarrel = False

                # loop over TauMinator taus
                for l1tauPt, l1tauEta, l1tauPhi, l1tauId in zip(_l1tau_pt, _l1tau_eta, _l1tau_phi, _l1tau_IDscore):
                    # skip the absurd cases with pT->infinity
                    if l1tauPt > 5000: continue

                    l1tau = ROOT.TLorentzVector()
                    l1tau.SetPtEtaPhiM(l1tauPt, l1tauEta, l1tauPhi, 0)

                    # check matching
                    if gentau.DeltaR(l1tau)<0.5:
                        # keep only L1 match with highest pT
                        if l1tau.Pt()>highestMinatedL1Pt:
                            minatedMatched = True
                            highestMinatedL1Pt = l1tau.Pt()
                            highestMinatedL1Id = l1tauId

                # squareMatched = False
                # highestSquaredL1Pt = -99.9
                # # loop over SquareCalo taus
                # for l1tauPt, l1tauEta, l1tauPhi, l1tauIso in zip(_squarel1tau_pt, _squarel1tau_eta, _squarel1tau_phi, _squarel1tau_iso):
                #     if not l1tauIso: continue # skip all the non iso taus

                #     l1tau = ROOT.TLorentzVector()
                #     l1tau.SetPtEtaPhiM(l1tauPt, l1tauEta, l1tauPhi, 0)

                #     # check matching
                #     if gentau.DeltaR(l1tau)<0.5:
                #         # keep only L1 match with highest pT
                #         if l1tau.Pt()>highestMinatedL1Pt:
                #            squareMatched = True
                #             highestSquaredL1Pt = l1tau.Pt()

                # fill numerator histograms for every thresholds
                for i, thr in enumerate(online_thresholds): 
                    if minatedMatched and highestMinatedL1Pt>float(thr):

                        if abs(l1tauEta)<1.5:
                            if highestMinatedL1Id >= idWp_CB['wp99']: # or highestMinatedL1Pt>75.: 
                                minated99_passing_ptBins[i].Fill(gentau.Pt())
                                if gentau.Pt() > 40: minated99_passing_etaBins[i].Fill(gentau.Eta())
                            
                            if highestMinatedL1Id >= idWp_CB['wp95']: # or highestMinatedL1Pt>75.: 
                                minated95_passing_ptBins[i].Fill(gentau.Pt())
                                if gentau.Pt() > 40: minated95_passing_etaBins[i].Fill(gentau.Eta())
                            
                            if highestMinatedL1Id >= idWp_CB['wp90']: # or highestMinatedL1Pt>75.: 
                                minated90_passing_ptBins[i].Fill(gentau.Pt())
                                if gentau.Pt() > 40: minated90_passing_etaBins[i].Fill(gentau.Eta())

                            if highestMinatedL1Id >= idWp_CB['wp85']: # or highestMinatedL1Pt>75.: 
                                minated85_passing_ptBins[i].Fill(gentau.Pt())

                            if highestMinatedL1Id >= idWp_CB['wp80']: # or highestMinatedL1Pt>75.: 
                                minated80_passing_ptBins[i].Fill(gentau.Pt())

                            if highestMinatedL1Id >= idWp_CB['wp75']: # or highestMinatedL1Pt>75.: 
                                minated75_passing_ptBins[i].Fill(gentau.Pt())
                        else:
                            if highestMinatedL1Id >= idWp_CE['wp99']: # or highestMinatedL1Pt>75.: 
                                minated99_passing_ptBins[i].Fill(gentau.Pt())
                                if gentau.Pt() > 40: minated99_passing_etaBins[i].Fill(gentau.Eta())
                            
                            if highestMinatedL1Id >= idWp_CE['wp95']: # or highestMinatedL1Pt>75.: 
                                minated95_passing_ptBins[i].Fill(gentau.Pt())
                                if gentau.Pt() > 40: minated95_passing_etaBins[i].Fill(gentau.Eta())
                            
                            if highestMinatedL1Id >= idWp_CE['wp90']: # or highestMinatedL1Pt>75.: 
                                minated90_passing_ptBins[i].Fill(gentau.Pt())
                                if gentau.Pt() > 40: minated90_passing_etaBins[i].Fill(gentau.Eta())

                            if highestMinatedL1Id >= idWp_CE['wp85']: # or highestMinatedL1Pt>75.: 
                                minated85_passing_ptBins[i].Fill(gentau.Pt())

                            if highestMinatedL1Id >= idWp_CE['wp80']: # or highestMinatedL1Pt>75.: 
                                minated80_passing_ptBins[i].Fill(gentau.Pt())

                            if highestMinatedL1Id >= idWp_CE['wp75']: # or highestMinatedL1Pt>75.: 
                                minated75_passing_ptBins[i].Fill(gentau.Pt())
                            

                    # if squareMatched and highestSquaredL1Pt>float(thr):
                    #     square_passing_ptBins[i].Fill(gentau.Pt())
                    #     square_passing_etaBins[i].Fill(gentau.Eta())

        # end of the loop over the events
        #################################

        # TGraphAsymmErrors for efficiency turn-ons
        turnonsMinated99 = []
        turnonsMinated95 = []
        turnonsMinated90 = []
        turnonsMinated85 = []
        turnonsMinated80 = []
        turnonsMinated75 = []
        # turnonsSquare = []
        etaEffMinated99 = []
        etaEffMinated95 = []
        etaEffMinated90 = []
        # etaEffSquare = []
        for i, thr in enumerate(online_thresholds):
            turnonsMinated99.append(ROOT.TGraphAsymmErrors(minated99_passing_ptBins[i], denominator_ptBins, "cp"))
            turnonsMinated95.append(ROOT.TGraphAsymmErrors(minated95_passing_ptBins[i], denominator_ptBins, "cp"))
            turnonsMinated90.append(ROOT.TGraphAsymmErrors(minated90_passing_ptBins[i], denominator_ptBins, "cp"))
            turnonsMinated85.append(ROOT.TGraphAsymmErrors(minated85_passing_ptBins[i], denominator_ptBins, "cp"))
            turnonsMinated80.append(ROOT.TGraphAsymmErrors(minated80_passing_ptBins[i], denominator_ptBins, "cp"))
            turnonsMinated75.append(ROOT.TGraphAsymmErrors(minated75_passing_ptBins[i], denominator_ptBins, "cp"))
            # turnonsSquare.append(ROOT.TGraphAsymmErrors(square_passing_ptBins[i], denominator_ptBins, "cp"))

            etaEffMinated99.append(ROOT.TGraphAsymmErrors(minated99_passing_etaBins[i], denominator_etaBins, "cp"))
            etaEffMinated95.append(ROOT.TGraphAsymmErrors(minated95_passing_etaBins[i], denominator_etaBins, "cp"))
            etaEffMinated90.append(ROOT.TGraphAsymmErrors(minated90_passing_etaBins[i], denominator_etaBins, "cp"))
            # etaEffSquare.append(ROOT.TGraphAsymmErrors(square_passing_etaBins[i], denominator_etaBins, "cp"))

        # save to file 

        fileout = ROOT.TFile(perfdir+"/turnons"+tag+"/efficiency_graphs"+tag+"_er"+str(options.etaEr)+".root","RECREATE")
        denominator_ptBins.Write()
        denominator_etaBins.Write()
        for i, thr in enumerate(online_thresholds): 
            minated99_passing_ptBins[i].Write()
            minated95_passing_ptBins[i].Write()
            minated90_passing_ptBins[i].Write()
            minated85_passing_ptBins[i].Write()
            minated80_passing_ptBins[i].Write()
            minated75_passing_ptBins[i].Write()
            # square_passing_ptBins[i].Write()
            minated99_passing_etaBins[i].Write()
            minated95_passing_etaBins[i].Write()
            minated90_passing_etaBins[i].Write()
            # square_passing_etaBins[i].Write()

            turnonsMinated99[i].Write()
            turnonsMinated95[i].Write()
            turnonsMinated90[i].Write()
            turnonsMinated85[i].Write()
            turnonsMinated80[i].Write()
            turnonsMinated75[i].Write()
            # turnonsSquare[i].Write()
            etaEffMinated99[i].Write()
            etaEffMinated95[i].Write()
            etaEffMinated90[i].Write()
            # etaEffSquare[i].Write()

        fileout.Close()

else:
    filein = ROOT.TFile(perfdir+"/turnons"+tag+"/efficiency_graphs"+tag+"_er"+str(options.etaEr)+".root","READ")

    # TGraphAsymmErrors for efficiency turn-ons
    turnonsMinated99 = []
    turnonsMinated95 = []
    turnonsMinated90 = []
    turnonsMinated85 = []
    turnonsMinated80 = []
    turnonsMinated75 = []
    # turnonsSquare = []
    etaEffMinated99 = []
    etaEffMinated95 = []
    etaEffMinated90 = []
    # etaEffSquare = []
    for i, thr in enumerate(online_thresholds):
        turnonsMinated99.append(filein.Get("divide_minated99_passing_thr"+str(int(thr))+"_ptBins_by_denominator_ptBins"))
        turnonsMinated95.append(filein.Get("divide_minated95_passing_thr"+str(int(thr))+"_ptBins_by_denominator_ptBins"))
        turnonsMinated90.append(filein.Get("divide_minated90_passing_thr"+str(int(thr))+"_ptBins_by_denominator_ptBins"))
        turnonsMinated85.append(filein.Get("divide_minated85_passing_thr"+str(int(thr))+"_ptBins_by_denominator_ptBins"))
        turnonsMinated80.append(filein.Get("divide_minated80_passing_thr"+str(int(thr))+"_ptBins_by_denominator_ptBins"))
        turnonsMinated75.append(filein.Get("divide_minated75_passing_thr"+str(int(thr))+"_ptBins_by_denominator_ptBins"))
        # turnonsSquare.append(filein.Get("divide_minated99_passing_thr"+str(int(thr))+"_ptBins_by_denominator_ptBins"))

        etaEffMinated99.append(filein.Get("divide_minated99_passing_thr"+str(int(thr))+"_etaBins_by_denominator_etaBins"))
        etaEffMinated95.append(filein.Get("divide_minated95_passing_thr"+str(int(thr))+"_etaBins_by_denominator_etaBins"))
        etaEffMinated90.append(filein.Get("divide_minated90_passing_thr"+str(int(thr))+"_etaBins_by_denominator_etaBins"))
        # etaEffSquare.append(filein.Get("divide_minated99_passing_thr"+str(int(thr))+"_etaBins_by_denominator_etaBins"))



turnons_dict = {}
etaeffs_dict = {}
mapping_dict = {'threshold':[],
                'wp99_pt95':[], 'wp99_pt90':[], 'wp99_pt50':[],
                'wp95_pt95':[], 'wp95_pt90':[], 'wp95_pt50':[],
                'wp90_pt95':[], 'wp90_pt90':[], 'wp90_pt50':[],
                'wp85_pt95':[], 'wp85_pt90':[], 'wp85_pt50':[],
                'wp80_pt95':[], 'wp80_pt90':[], 'wp80_pt50':[],
                'wp75_pt95':[], 'wp75_pt90':[], 'wp75_pt50':[]}
for i, thr in enumerate(online_thresholds):
    turnons_dict['turnonAt99wpAt'+str(thr)+'GeV'] = [[],[],[]]
    turnons_dict['turnonAt95wpAt'+str(thr)+'GeV'] = [[],[],[]]
    turnons_dict['turnonAt90wpAt'+str(thr)+'GeV'] = [[],[],[]]
    turnons_dict['turnonAt85wpAt'+str(thr)+'GeV'] = [[],[],[]]
    turnons_dict['turnonAt80wpAt'+str(thr)+'GeV'] = [[],[],[]]
    turnons_dict['turnonAt75wpAt'+str(thr)+'GeV'] = [[],[],[]]

    etaeffs_dict['efficiencyVsEtaAt99wpAt'+str(thr)+'GeV'] = [[],[],[]]
    etaeffs_dict['efficiencyVsEtaAt95wpAt'+str(thr)+'GeV'] = [[],[],[]]
    etaeffs_dict['efficiencyVsEtaAt90wpAt'+str(thr)+'GeV'] = [[],[],[]]

    for ibin in range(0,turnonsMinated99[i].GetN()):
        turnons_dict['turnonAt99wpAt'+str(thr)+'GeV'][0].append(turnonsMinated99[i].GetPointY(ibin))
        turnons_dict['turnonAt95wpAt'+str(thr)+'GeV'][0].append(turnonsMinated95[i].GetPointY(ibin))
        turnons_dict['turnonAt90wpAt'+str(thr)+'GeV'][0].append(turnonsMinated90[i].GetPointY(ibin))
        turnons_dict['turnonAt85wpAt'+str(thr)+'GeV'][0].append(turnonsMinated85[i].GetPointY(ibin))
        turnons_dict['turnonAt80wpAt'+str(thr)+'GeV'][0].append(turnonsMinated80[i].GetPointY(ibin))
        turnons_dict['turnonAt75wpAt'+str(thr)+'GeV'][0].append(turnonsMinated75[i].GetPointY(ibin))

        turnons_dict['turnonAt99wpAt'+str(thr)+'GeV'][1].append(turnonsMinated99[i].GetErrorYlow(ibin))
        turnons_dict['turnonAt95wpAt'+str(thr)+'GeV'][1].append(turnonsMinated95[i].GetErrorYlow(ibin))
        turnons_dict['turnonAt90wpAt'+str(thr)+'GeV'][1].append(turnonsMinated90[i].GetErrorYlow(ibin))
        turnons_dict['turnonAt85wpAt'+str(thr)+'GeV'][1].append(turnonsMinated85[i].GetErrorYlow(ibin))
        turnons_dict['turnonAt80wpAt'+str(thr)+'GeV'][1].append(turnonsMinated80[i].GetErrorYlow(ibin))
        turnons_dict['turnonAt75wpAt'+str(thr)+'GeV'][1].append(turnonsMinated75[i].GetErrorYlow(ibin))

        turnons_dict['turnonAt99wpAt'+str(thr)+'GeV'][2].append(turnonsMinated99[i].GetErrorYhigh(ibin))
        turnons_dict['turnonAt95wpAt'+str(thr)+'GeV'][2].append(turnonsMinated95[i].GetErrorYhigh(ibin))
        turnons_dict['turnonAt90wpAt'+str(thr)+'GeV'][2].append(turnonsMinated90[i].GetErrorYhigh(ibin))
        turnons_dict['turnonAt85wpAt'+str(thr)+'GeV'][2].append(turnonsMinated85[i].GetErrorYhigh(ibin))
        turnons_dict['turnonAt80wpAt'+str(thr)+'GeV'][2].append(turnonsMinated80[i].GetErrorYhigh(ibin))
        turnons_dict['turnonAt75wpAt'+str(thr)+'GeV'][2].append(turnonsMinated75[i].GetErrorYhigh(ibin))

    for ibin in range(0,etaEffMinated99[i].GetN()):
        etaeffs_dict['efficiencyVsEtaAt99wpAt'+str(thr)+'GeV'][0].append(etaEffMinated99[i].GetPointY(ibin))
        etaeffs_dict['efficiencyVsEtaAt95wpAt'+str(thr)+'GeV'][0].append(etaEffMinated95[i].GetPointY(ibin))
        etaeffs_dict['efficiencyVsEtaAt90wpAt'+str(thr)+'GeV'][0].append(etaEffMinated90[i].GetPointY(ibin))

        etaeffs_dict['efficiencyVsEtaAt99wpAt'+str(thr)+'GeV'][1].append(etaEffMinated99[i].GetErrorYlow(ibin))
        etaeffs_dict['efficiencyVsEtaAt95wpAt'+str(thr)+'GeV'][1].append(etaEffMinated95[i].GetErrorYlow(ibin))
        etaeffs_dict['efficiencyVsEtaAt90wpAt'+str(thr)+'GeV'][1].append(etaEffMinated90[i].GetErrorYlow(ibin))

        etaeffs_dict['efficiencyVsEtaAt99wpAt'+str(thr)+'GeV'][2].append(etaEffMinated99[i].GetErrorYhigh(ibin))
        etaeffs_dict['efficiencyVsEtaAt95wpAt'+str(thr)+'GeV'][2].append(etaEffMinated95[i].GetErrorYhigh(ibin))
        etaeffs_dict['efficiencyVsEtaAt90wpAt'+str(thr)+'GeV'][2].append(etaEffMinated90[i].GetErrorYhigh(ibin))

    # ONLINE TO OFFILNE MAPPING
    mapping_dict['wp99_pt95'].append(np.interp(0.95, turnons_dict['turnonAt99wpAt'+str(thr)+'GeV'][0], offline_pts)) #,right=-99,left=-98)
    mapping_dict['wp99_pt90'].append(np.interp(0.90, turnons_dict['turnonAt99wpAt'+str(thr)+'GeV'][0], offline_pts)) #,right=-99,left=-98)
    mapping_dict['wp99_pt50'].append(np.interp(0.50, turnons_dict['turnonAt99wpAt'+str(thr)+'GeV'][0], offline_pts)) #,right=-99,left=-98)

    mapping_dict['wp95_pt95'].append(np.interp(0.95, turnons_dict['turnonAt95wpAt'+str(thr)+'GeV'][0], offline_pts)) #,right=-99,left=-98)
    mapping_dict['wp95_pt90'].append(np.interp(0.90, turnons_dict['turnonAt95wpAt'+str(thr)+'GeV'][0], offline_pts)) #,right=-99,left=-98)
    mapping_dict['wp95_pt50'].append(np.interp(0.50, turnons_dict['turnonAt95wpAt'+str(thr)+'GeV'][0], offline_pts)) #,right=-99,left=-98)

    mapping_dict['wp90_pt95'].append(np.interp(0.95, turnons_dict['turnonAt90wpAt'+str(thr)+'GeV'][0], offline_pts)) #,right=-99,left=-98)
    mapping_dict['wp90_pt90'].append(np.interp(0.90, turnons_dict['turnonAt90wpAt'+str(thr)+'GeV'][0], offline_pts)) #,right=-99,left=-98)
    mapping_dict['wp90_pt50'].append(np.interp(0.50, turnons_dict['turnonAt90wpAt'+str(thr)+'GeV'][0], offline_pts)) #,right=-99,left=-98)

    mapping_dict['wp85_pt95'].append(np.interp(0.95, turnons_dict['turnonAt85wpAt'+str(thr)+'GeV'][0], offline_pts)) #,right=-99,left=-98)
    mapping_dict['wp85_pt90'].append(np.interp(0.90, turnons_dict['turnonAt85wpAt'+str(thr)+'GeV'][0], offline_pts)) #,right=-99,left=-98)
    mapping_dict['wp85_pt50'].append(np.interp(0.50, turnons_dict['turnonAt85wpAt'+str(thr)+'GeV'][0], offline_pts)) #,right=-99,left=-98)

    mapping_dict['wp80_pt95'].append(np.interp(0.95, turnons_dict['turnonAt80wpAt'+str(thr)+'GeV'][0], offline_pts)) #,right=-99,left=-98)
    mapping_dict['wp80_pt90'].append(np.interp(0.90, turnons_dict['turnonAt80wpAt'+str(thr)+'GeV'][0], offline_pts)) #,right=-99,left=-98)
    mapping_dict['wp80_pt50'].append(np.interp(0.50, turnons_dict['turnonAt80wpAt'+str(thr)+'GeV'][0], offline_pts)) #,right=-99,left=-98)

    mapping_dict['wp75_pt95'].append(np.interp(0.95, turnons_dict['turnonAt75wpAt'+str(thr)+'GeV'][0], offline_pts)) #,right=-99,left=-98)
    mapping_dict['wp75_pt90'].append(np.interp(0.90, turnons_dict['turnonAt75wpAt'+str(thr)+'GeV'][0], offline_pts)) #,right=-99,left=-98)
    mapping_dict['wp75_pt50'].append(np.interp(0.50, turnons_dict['turnonAt75wpAt'+str(thr)+'GeV'][0], offline_pts)) #,right=-99,left=-98)

save_obj(mapping_dict, perfdir+'/turnons'+tag+'/online2offline_mapping.pkl')



# X_square = [2.8468899521531092, 8.875598086124402, 15.0, 20.933014354066984, 26.794258373205736, 32.99043062200957, 39.01913875598086, 44.88038277511962, 50.909090909090914, 57.10526315789473, 63.0, 68.99521531100478, 75.1, 81.0, 87.0, 93.0, 99.13875598086125, 105.0, 111.1, 117.0, 123.1, 129.0, 135.0, 141.17224880382778, 147.0334928229665]
# Y_square = [0.20067385444743946, 0.08301886792452828, 0.10727762803234508, 0.28557951482479793, 0.55, 0.8035040431266847, 0.9247978436657682, 0.9721024258760108, 0.9757412398921833, 0.992722371967655, 0.9987870619946092, 0.9939353099730459, 0.9987870619946092, 0.9987870619946092, 0.9987870619946092, 1.0, 1.0, 0.9987870619946092, 0.9987870619946092, 0.9975741239892184, 1.0, 1.0, 1.0, 1.0, 1.0]
# X_puppi = [8.875598086124402, 15.0, 20.933014354066984, 26.794258373205736, 32.99043062200957, 39.01913875598086, 44.88038277511962, 50.909090909090914, 57.10526315789473, 63.0, 68.99521531100478, 75.1, 81.0, 87.0, 93.0, 99.13875598086125, 105.0, 111.1, 117.0, 123.1, 129.0, 135.0, 141.17224880382778, 147.0334928229665]
# Y_puppi = [0.0, 0.005390835579514919, 0.013881401617250821, 0.10242587601078168, 0.5827493261455526, 0.8047169811320755, 0.8629380053908356, 0.9150943396226415, 0.9478436657681941, 0.9684636118598383, 0.9733153638814016, 0.9842318059299192, 0.9757412398921833, 0.9842318059299192, 0.9769541778975741, 0.9951482479784367, 0.9890835579514825, 0.9939353099730459, 0.9975741239892184, 0.9951482479784367, 0.992722371967655, 1.0, 0.9915094339622642, 1.0]

# plt.figure(figsize=(10,10))
# plt.errorbar(offline_pts,turnons_dict['turnonAt90wpAt28GeV'][0],xerr=1,yerr=[turnons_dict['turnonAt90wpAt28GeV'][1], turnons_dict['turnonAt90wpAt28GeV'][2]], ls='None', label=r'$p_{T}^{L1 \tau} > 32 GeV', lw=2, marker='o', color='green')
# plt.errorbar(X_square,Y_square,xerr=1, ls='None', label=r'$p_{T}^{L1 \tau} > 32 GeV', lw=2, marker='o', color='blue')
# plt.errorbar(X_puppi,Y_puppi,xerr=1, ls='None', label=r'$p_{T}^{L1 \tau} > 32 GeV', lw=2, marker='o', color='red')
# plt.hlines(0.90, 0, 2000, lw=2, color='dimgray', label='0.90 Eff.')
# plt.hlines(0.95, 0, 2000, lw=2, color='black', label='0.95 Eff.')
# plt.legend(loc = 'lower right', fontsize=14)
# plt.ylim(0., 1.05)
# plt.xlim(15., 160.)
# # plt.xscale('log')
# plt.xlabel(r'$p_{T}^{gen,\tau}\ [GeV]$')
# plt.ylabel(r'Efficiency')
# plt.grid()
# mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
# plt.savefig(perfdir+'/turnons'+tag+'/turnons_hacky.pdf')
# plt.close()



cmap = matplotlib.cm.get_cmap('tab20c'); i=0
##################################################################################
# PLOT TURNONS
i = 0
plt.figure(figsize=(10,10))
for thr in plotting_thresholds:
    if not thr%10:
        plt.errorbar(offline_pts,turnons_dict['turnonAt99wpAt'+str(thr)+'GeV'][0],xerr=1,yerr=[turnons_dict['turnonAt99wpAt'+str(thr)+'GeV'][1], turnons_dict['turnonAt99wpAt'+str(thr)+'GeV'][2]], ls='None', label=r'$p_{T}^{L1 \tau} > %i$ GeV' % (thr), lw=2, marker='o', color=cmap(i))

        p0 = [1, thr, 1] 
        popt, pcov = curve_fit(sigmoid, offline_pts, turnons_dict['turnonAt99wpAt'+str(thr)+'GeV'][0], p0, maxfev=5000)
        plt.plot(offline_pts, sigmoid(offline_pts, *popt), '-', label='_', lw=1.5, color=cmap(i))

        i+=1 
plt.hlines(0.90, 0, 2000, lw=2, color='dimgray', label='0.90 Eff.')
plt.hlines(0.95, 0, 2000, lw=2, color='black', label='0.95 Eff.')
plt.legend(loc = 'lower right', fontsize=14)
plt.ylim(0., 1.05)
plt.xlim(15., 160.)
# plt.xscale('log')
plt.xlabel(r'$p_{T}^{gen,\tau}\ [GeV]$')
plt.ylabel(r'Efficiency')
plt.grid()
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig(perfdir+'/turnons'+tag+'/turnons_WP99.pdf')
plt.close()

i = 0
plt.figure(figsize=(10,10))
for thr in plotting_thresholds:
    if not thr%10:
        plt.errorbar(offline_pts,turnons_dict['turnonAt95wpAt'+str(thr)+'GeV'][0],xerr=1,yerr=[turnons_dict['turnonAt95wpAt'+str(thr)+'GeV'][1], turnons_dict['turnonAt95wpAt'+str(thr)+'GeV'][2]], ls='None', label=r'$p_{T}^{L1 \tau} > %i$ GeV' % (thr), lw=2, marker='o', color=cmap(i))

        p0 = [1, thr, 1] 
        popt, pcov = curve_fit(sigmoid, offline_pts, turnons_dict['turnonAt95wpAt'+str(thr)+'GeV'][0], p0, maxfev=5000)
        plt.plot(offline_pts, sigmoid(offline_pts, *popt), '-', label='_', lw=1.5, color=cmap(i))

        i+=1 
plt.hlines(0.90, 0, 2000, lw=2, color='dimgray', label='0.90 Eff.')
plt.hlines(0.95, 0, 2000, lw=2, color='black', label='0.95 Eff.')
plt.legend(loc = 'lower right', fontsize=14)
plt.ylim(0., 1.05)
plt.xlim(15., 160.)
# plt.xscale('log')
plt.xlabel(r'$p_{T}^{gen,\tau}\ [GeV]$')
plt.ylabel(r'Efficiency')
plt.grid()
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig(perfdir+'/turnons'+tag+'/turnons_WP95.pdf')
plt.close()

i = 0
plt.figure(figsize=(10,10))
for thr in plotting_thresholds:
    if not thr%10:
        plt.errorbar(offline_pts,turnons_dict['turnonAt90wpAt'+str(thr)+'GeV'][0],xerr=1,yerr=[turnons_dict['turnonAt90wpAt'+str(thr)+'GeV'][1], turnons_dict['turnonAt90wpAt'+str(thr)+'GeV'][2]], ls='None', label=r'$p_{T}^{L1 \tau} > %i$ GeV' % (thr), lw=2, marker='o', color=cmap(i))

        p0 = [1, thr, 1] 
        popt, pcov = curve_fit(sigmoid, offline_pts, turnons_dict['turnonAt90wpAt'+str(thr)+'GeV'][0], p0, maxfev=5000)
        plt.plot(offline_pts, sigmoid(offline_pts, *popt), '-', label='_', lw=1.5, color=cmap(i))

        i+=1 
plt.hlines(0.90, 0, 2000, lw=2, color='dimgray', label='0.90 Eff.')
plt.hlines(0.95, 0, 2000, lw=2, color='black', label='0.95 Eff.')
plt.legend(loc = 'lower right', fontsize=14)
plt.ylim(0., 1.05)
plt.xlim(15., 160.)
# plt.xscale('log')
plt.xlabel(r'$p_{T}^{gen,\tau}\ [GeV]$')
plt.ylabel(r'Efficiency')
plt.grid()
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig(perfdir+'/turnons'+tag+'/turnons_WP90.pdf')
plt.close()

i = 0
plt.figure(figsize=(10,10))
for thr in plotting_thresholds:
    if not thr%10:
        plt.errorbar(offline_pts,turnons_dict['turnonAt85wpAt'+str(thr)+'GeV'][0],xerr=1,yerr=[turnons_dict['turnonAt85wpAt'+str(thr)+'GeV'][1], turnons_dict['turnonAt85wpAt'+str(thr)+'GeV'][2]], ls='None', label=r'$p_{T}^{L1 \tau} > %i$ GeV' % (thr), lw=2, marker='o', color=cmap(i))

        p0 = [1, thr, 1] 
        popt, pcov = curve_fit(sigmoid, offline_pts, turnons_dict['turnonAt85wpAt'+str(thr)+'GeV'][0], p0, maxfev=5000)
        plt.plot(offline_pts, sigmoid(offline_pts, *popt), '-', label='_', lw=1.5, color=cmap(i))

        i+=1 
plt.hlines(0.90, 0, 2000, lw=2, color='dimgray', label='0.90 Eff.')
plt.hlines(0.95, 0, 2000, lw=2, color='black', label='0.95 Eff.')
plt.legend(loc = 'lower right', fontsize=14)
plt.ylim(0., 1.05)
plt.xlim(15., 160.)
# plt.xscale('log')
plt.xlabel(r'$p_{T}^{gen,\tau}\ [GeV]$')
plt.ylabel(r'Efficiency')
plt.grid()
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig(perfdir+'/turnons'+tag+'/turnons_WP85.pdf')
plt.close()


i = 0
plt.figure(figsize=(10,10))
for thr in plotting_thresholds:
    if not thr%10:
        plt.errorbar(offline_pts,turnons_dict['turnonAt80wpAt'+str(thr)+'GeV'][0],xerr=1,yerr=[turnons_dict['turnonAt80wpAt'+str(thr)+'GeV'][1], turnons_dict['turnonAt80wpAt'+str(thr)+'GeV'][2]], ls='None', label=r'$p_{T}^{L1 \tau} > %i$ GeV' % (thr), lw=2, marker='o', color=cmap(i))

        p0 = [1, thr, 1] 
        popt, pcov = curve_fit(sigmoid, offline_pts, turnons_dict['turnonAt80wpAt'+str(thr)+'GeV'][0], p0, maxfev=5000)
        plt.plot(offline_pts, sigmoid(offline_pts, *popt), '-', label='_', lw=1.5, color=cmap(i))

        i+=1 
plt.hlines(0.90, 0, 2000, lw=2, color='dimgray', label='0.90 Eff.')
plt.hlines(0.95, 0, 2000, lw=2, color='black', label='0.95 Eff.')
plt.legend(loc = 'lower right', fontsize=14)
plt.ylim(0., 1.05)
plt.xlim(15., 160.)
# plt.xscale('log')
plt.xlabel(r'$p_{T}^{gen,\tau}\ [GeV]$')
plt.ylabel(r'Efficiency')
plt.grid()
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig(perfdir+'/turnons'+tag+'/turnons_WP80.pdf')
plt.close()


i = 0
plt.figure(figsize=(10,10))
for thr in plotting_thresholds:
    if not thr%10:
        plt.errorbar(offline_pts,turnons_dict['turnonAt75wpAt'+str(thr)+'GeV'][0],xerr=1,yerr=[turnons_dict['turnonAt75wpAt'+str(thr)+'GeV'][1], turnons_dict['turnonAt75wpAt'+str(thr)+'GeV'][2]], ls='None', label=r'$p_{T}^{L1 \tau} > %i$ GeV' % (thr), lw=2, marker='o', color=cmap(i))

        p0 = [1, thr, 1] 
        popt, pcov = curve_fit(sigmoid, offline_pts, turnons_dict['turnonAt75wpAt'+str(thr)+'GeV'][0], p0, maxfev=5000)
        plt.plot(offline_pts, sigmoid(offline_pts, *popt), '-', label='_', lw=1.5, color=cmap(i))

        i+=1 
plt.hlines(0.90, 0, 2000, lw=2, color='dimgray', label='0.90 Eff.')
plt.hlines(0.95, 0, 2000, lw=2, color='black', label='0.95 Eff.')
plt.legend(loc = 'lower right', fontsize=14)
plt.ylim(0., 1.05)
plt.xlim(15., 160.)
# plt.xscale('log')
plt.xlabel(r'$p_{T}^{gen,\tau}\ [GeV]$')
plt.ylabel(r'Efficiency')
plt.grid()
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig(perfdir+'/turnons'+tag+'/turnons_WP75.pdf')
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
plt.savefig(perfdir+'/turnons'+tag+'/online2offline_WP99.pdf')
plt.close()

plt.figure(figsize=(10,10))
plt.plot(online_thresholds, mapping_dict['wp95_pt95'], label='@ 95% efficiency', linewidth=2, color='blue')
plt.plot(online_thresholds, mapping_dict['wp95_pt90'], label='@ 90% efficiency', linewidth=2, color='red')
plt.plot(online_thresholds, mapping_dict['wp95_pt50'], label='@ 50% efficiency', linewidth=2, color='green')
plt.legend(loc = 'lower right', fontsize=14)
plt.xlabel('L1 Threshold [GeV]')
plt.ylabel('Offline threshold [GeV]')
plt.xlim(20, 100)
plt.ylim(20, 200)
plt.grid()
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig(perfdir+'/turnons'+tag+'/online2offline_WP95.pdf')
plt.close()

plt.figure(figsize=(10,10))
plt.plot(online_thresholds, mapping_dict['wp90_pt95'], label='@ 95% efficiency', linewidth=2, color='blue')
plt.plot(online_thresholds, mapping_dict['wp90_pt90'], label='@ 90% efficiency', linewidth=2, color='red')
plt.plot(online_thresholds, mapping_dict['wp90_pt50'], label='@ 50% efficiency', linewidth=2, color='green')
plt.legend(loc = 'lower right', fontsize=14)
plt.xlabel('L1 Threshold [GeV]')
plt.ylabel('Offline threshold [GeV]')
plt.xlim(20, 100)
plt.ylim(20, 200)
plt.grid()
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig(perfdir+'/turnons'+tag+'/online2offline_WP90.pdf')
plt.close()

plt.figure(figsize=(10,10))
plt.plot(online_thresholds, mapping_dict['wp85_pt95'], label='@ 95% efficiency', linewidth=2, color='blue')
plt.plot(online_thresholds, mapping_dict['wp85_pt90'], label='@ 90% efficiency', linewidth=2, color='red')
plt.plot(online_thresholds, mapping_dict['wp85_pt50'], label='@ 50% efficiency', linewidth=2, color='green')
plt.legend(loc = 'lower right', fontsize=14)
plt.xlabel('L1 Threshold [GeV]')
plt.ylabel('Offline threshold [GeV]')
plt.xlim(20, 100)
plt.ylim(20, 200)
plt.grid()
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig(perfdir+'/turnons'+tag+'/online2offline_WP85.pdf')
plt.close()

plt.figure(figsize=(10,10))
plt.plot(online_thresholds, mapping_dict['wp80_pt95'], label='@ 95% efficiency', linewidth=2, color='blue')
plt.plot(online_thresholds, mapping_dict['wp80_pt90'], label='@ 90% efficiency', linewidth=2, color='red')
plt.plot(online_thresholds, mapping_dict['wp80_pt50'], label='@ 50% efficiency', linewidth=2, color='green')
plt.legend(loc = 'lower right', fontsize=14)
plt.xlabel('L1 Threshold [GeV]')
plt.ylabel('Offline threshold [GeV]')
plt.xlim(20, 100)
plt.ylim(20, 200)
plt.grid()
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig(perfdir+'/turnons'+tag+'/online2offline_WP80.pdf')
plt.close()

plt.figure(figsize=(10,10))
plt.plot(online_thresholds, mapping_dict['wp75_pt95'], label='@ 95% efficiency', linewidth=2, color='blue')
plt.plot(online_thresholds, mapping_dict['wp75_pt90'], label='@ 90% efficiency', linewidth=2, color='red')
plt.plot(online_thresholds, mapping_dict['wp75_pt50'], label='@ 50% efficiency', linewidth=2, color='green')
plt.legend(loc = 'lower right', fontsize=14)
plt.xlabel('L1 Threshold [GeV]')
plt.ylabel('Offline threshold [GeV]')
plt.xlim(20, 100)
plt.ylim(20, 200)
plt.grid()
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig(perfdir+'/turnons'+tag+'/online2offline_WP75.pdf')
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
plt.savefig(perfdir+'/turnons'+tag+'/efficiencyVSeta_WP99.pdf')
plt.close()

i = 0
plt.figure(figsize=(10,10))
plt.errorbar(eta_bins_centers,etaeffs_dict['efficiencyVsEtaAt95wpAt32GeV'][0],yerr=[etaeffs_dict['efficiencyVsEtaAt95wpAt32GeV'][1], etaeffs_dict['efficiencyVsEtaAt95wpAt32GeV'][2]], ls='None', label=r'$p_{T}^{L1 \tau} > %i$ GeV' % (32), lw=2, marker='o', color=cmap(i))
plt.hlines(0.90, -3.0, 3.0, lw=2, color='dimgray', label='0.90 Eff.')
plt.hlines(0.95, -3.0, 3.0, lw=2, color='black', label='0.95 Eff.')
plt.legend(loc = 'lower right', fontsize=14)
plt.ylim(0., 1.05)
plt.xlim(-3.0, 3.0)
plt.xlabel(r'$\eta^{gen,\tau}$')
plt.ylabel(r'Efficiency')
plt.grid()
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig(perfdir+'/turnons'+tag+'/efficiencyVSeta_WP95.pdf')
plt.close()

i = 0
plt.figure(figsize=(10,10))
plt.errorbar(eta_bins_centers,etaeffs_dict['efficiencyVsEtaAt90wpAt32GeV'][0],yerr=[etaeffs_dict['efficiencyVsEtaAt90wpAt32GeV'][1], etaeffs_dict['efficiencyVsEtaAt90wpAt32GeV'][2]], ls='None', label=r'$p_{T}^{L1 \tau} > %i$ GeV' % (32), lw=2, marker='o', color=cmap(i))
plt.hlines(0.90, -3.0, 3.0, lw=2, color='dimgray', label='0.90 Eff.')
plt.hlines(0.95, -3.0, 3.0, lw=2, color='black', label='0.95 Eff.')
plt.legend(loc = 'lower right', fontsize=14)
plt.ylim(0., 1.05)
plt.xlim(-3.0, 3.0)
plt.xlabel(r'$\eta^{gen,\tau}$')
plt.ylabel(r'Efficiency')
plt.grid()
mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
plt.savefig(perfdir+'/turnons'+tag+'/efficiencyVSeta_WP90.pdf')
plt.close()




