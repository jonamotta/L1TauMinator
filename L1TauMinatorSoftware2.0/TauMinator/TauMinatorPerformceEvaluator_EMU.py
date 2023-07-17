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

################################################################
## MENU WAY OF DOING SCALING

def _interpolate(H, K1, K2):

        A = np.ones(len(K1)) * (-K2)
        B = [k1i + 2 * K2 for k1i in K1]
        C = np.ones(len(K1)) * (-K2)
        D = []

        A[0] = 0
        C[-1] = 0

        D = [k1i * h1i for k1i, h1i in zip(K1, H)]

        D[0] = D[0] + K2 * 0
        D[-1] = D[-1] + K2 * 1

        for i in range(1, len(K1)):
            F = A[i] / B[i - 1]

            A[i] = A[i] - B[i - 1] * F
            B[i] = B[i] - C[i - 1] * F
            C[i] = C[i]
            D[i] = D[i] - D[i - 1] * F

        Y = np.ones(len(K1))
        Y[-1] = D[-1] / B[-1]

        for i in reversed(range(len(K1) - 2)):
            Y[i] = (D[i] - C[i] * Y[i + 1]) / B[i]

        return Y

def _get_point_on_curve(x, graph_x, graph_y):

        if (x < graph_x[0]):
            return 0

        if (x >= graph_x[len(graph_x) - 1]):
            return 1

        xr = graph_x[0]
        yr = graph_y[0]
        for i in range(len(graph_x) - 1):
            xl = xr
            yl = yr
            xr = graph_x[i + 1]
            yr = graph_y[i + 1]
            if ((x < xr) & (x >= xl)):
                return yl + (yr - yl) / (xr - xl) * (x - xl)

        return -1

def _find_turnon_cut(graph_x, graph_y, Target):
        L = 0
        R = np.max(graph_x)

        while (R - L > 0.0001):
            C = (L + R) / 2
            V = _get_point_on_curve(C, graph_x, graph_y)

            if (V < Target):
                L = C
            else:
                R = C

        return (R + L) / 2.

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

    # MENU BINNING
    # ptBins = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72, 78, 84, 90, 96, 102, 108, 114, 120, 126, 132, 138, 144, 150]
    # offline_pts = [3, 9, 15, 21, 27, 33, 39, 45, 51, 57, 63, 69, 75, 81, 87, 93, 99, 105, 111, 117, 123, 129, 135, 141, 147]

    if options.etaEr==3.0:
        etaBins=[-3.0, -2.9, -2.8, -2.7, -2.6, -2.5, -2.4, -2.3, -2.2, -2.1, -2.0, -1.9, -1.8, -1.7, -1.6, -1.5, -1.305, -1.0, -0.66, -0.33, 0.0, 0.33, 0.66, 1.0, 1.305, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0]
        eta_bins_centers = [-2.95, -2.85, -2.75, -2.65, -2.55, -2.45, -2.35, -2.25, -2.15, -2.05, -1.95, -1.85, -1.75, -1.65, -1.55, -1.4025, -1.1525, -0.825, -0.495, -0.165, 0.165, 0.495, 0.825, 1.1525, 1.4025, 1.55, 1.65, 1.75, 1.85, 1.95, 2.05, 2.15, 2.25, 2.35, 2.45, 2.55, 2.65, 2.75, 2.85, 2.95]
    elif options.etaEr==2.4:
        etaBins=[-2.4, -2.3, -2.2, -2.1, -2.0, -1.9, -1.8, -1.7, -1.6, -1.5, -1.305, -1.0, -0.66, -0.33, 0.0, 0.33, 0.66, 1.0, 1.305, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4]
        eta_bins_centers = [-2.35, -2.25, -2.15, -2.05, -1.95, -1.85, -1.75, -1.65, -1.55, -1.4025, -1.1525, -0.825, -0.495, -0.165, 0.165, 0.495, 0.825, 1.1525, 1.4025, 1.55, 1.65, 1.75, 1.85, 1.95, 2.05, 2.15, 2.25, 2.35]
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
        minated_passing_ptBins = []
        minated_passing_ptBins_CB = []
        minated_passing_ptBins_CE = []

        minated_passing_etaBins = []

        for threshold in online_thresholds:
            minated_passing_ptBins.append(ROOT.TH1F("minated_passing_thr"+str(int(threshold))+"_ptBins","minated_passing_thr"+str(int(threshold))+"_ptBins",len(ptBins)-1, array('f',ptBins)))
            minated_passing_ptBins_CB.append(ROOT.TH1F("minated_passing_thr"+str(int(threshold))+"_ptBins_CB","minated_passing_thr"+str(int(threshold))+"_ptBins_CB",len(ptBins)-1, array('f',ptBins)))
            minated_passing_ptBins_CE.append(ROOT.TH1F("minated_passing_thr"+str(int(threshold))+"_ptBins_CE","minated_passing_thr"+str(int(threshold))+"_ptBins_CE",len(ptBins)-1, array('f',ptBins)))
            
            minated_passing_etaBins.append(ROOT.TH1F("minated_passing_thr"+str(int(threshold))+"_etaBins","minated_passing_thr"+str(int(threshold))+"_etaBins",len(etaBins)-1, array('f',etaBins)))

        #denominator
        denominator_ptBins = ROOT.TH1F("denominator_ptBins","denominator_ptBins",len(ptBins)-1, array('f',ptBins))
        denominator_ptBins_CB = ROOT.TH1F("denominator_ptBins_CB","denominator_ptBins_CB",len(ptBins)-1, array('f',ptBins))
        denominator_ptBins_CE = ROOT.TH1F("denominator_ptBins_CE","denominator_ptBins_CE",len(ptBins)-1, array('f',ptBins))
        denominator_etaBins = ROOT.TH1F("denominator_etaBins","denominator_etaBins",len(etaBins)-1, array('f',etaBins))

        tot = 0

        # loop over the events to fill all the histograms
        directory = '/data_CMS/cms/motta/Phase2L1T/L1TauMinatorNtuples/v'+options.NtupleV+'/GluGluToHHTo2B2Tau_node_SM_TuneCP5_14TeV-madgraph-pythia8__Phase2Fall22DRMiniAOD-PU200_125X_mcRun4_realistic_v2-v1__GEN-SIM-DIGI-RAW-MINIAOD_seedEtCut'+options.seedEtCut+options.inTag+'/'
        if "PRD" in options.inTag:   inChain = ROOT.TChain("L1CaloTauNtuplizerProducer/L1TauMinatorTree");
        elif "EMU" in options.inTag: inChain = ROOT.TChain("L1CaloTauNtuplizerEmulator/L1TauMinatorTree");
        else:                        inChain = ROOT.TChain("L1CaloTauNtuplizer/L1TauMinatorTree");

        inChain.Add(directory+'/Ntuple_*.root');
        nEntries = inChain.GetEntries()
        for evt in range(0, nEntries):
            if evt%1000==0: print('--> ',evt)
            # if evt == 10000: break

            entry = inChain.GetEntry(evt)

            ### JONA'S TAUS
            # _gentau_visEta = list(inChain.tau_visEta)
            # _gentau_visPhi = list(inChain.tau_visPhi)
            # _gentau_visPt = list(inChain.tau_visPt)

            ### MENU'S TAUS
            _gentau_visEta = list(inChain.tau_mod_visEta)
            _gentau_visPhi = list(inChain.tau_mod_visPhi)
            _gentau_visPt = list(inChain.tau_mod_visPt)

            _l1tau_pt = list(inChain.minatedl1tau_pt)
            _l1tau_eta = list(inChain.minatedl1tau_eta)
            _l1tau_phi = list(inChain.minatedl1tau_phi)
            _l1tau_quality = list(inChain.minatedl1tau_quality)

            if len(_gentau_visPt) == 0: continue

            ###########################################################################
            # BARREL TURNON
            maxPt = -99.
            for genPt, genEta in zip(_gentau_visPt, _gentau_visEta):
                if abs(genEta) > 1.5: continue
                if genPt > maxPt: maxPt = genPt

            if maxPt > 0:
                maxEta = _gentau_visEta[_gentau_visPt.index(maxPt)]
                maxPhi = _gentau_visPhi[_gentau_visPt.index(maxPt)]

                gentau = ROOT.TLorentzVector()
                gentau.SetPtEtaPhiM(maxPt, maxEta, maxPhi, 0)

                if abs(gentau.Eta()) < options.etaEr: # skip taus out of acceptance

                    denominator_ptBins_CB.Fill(gentau.Pt())

                    minatedMatched = False
                    highestMinatedL1Pt = -99.9

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

                    # fill numerator histograms for every thresholds
                    for i, thr in enumerate(online_thresholds): 
                        if minatedMatched and highestMinatedL1Pt>float(thr):
                            minated_passing_ptBins_CB[i].Fill(gentau.Pt())


            ###########################################################################
            # ENDCAP TURNON
            maxPt = -99.
            for genPt, genEta in zip(_gentau_visPt, _gentau_visEta):
                if abs(genEta) < 1.5: continue
                if genPt > maxPt: maxPt = genPt

            if maxPt > 0:
                maxEta = _gentau_visEta[_gentau_visPt.index(maxPt)]
                maxPhi = _gentau_visPhi[_gentau_visPt.index(maxPt)]

                gentau = ROOT.TLorentzVector()
                gentau.SetPtEtaPhiM(maxPt, maxEta, maxPhi, 0)

                if abs(gentau.Eta()) < options.etaEr: # skip taus out of acceptance

                    denominator_ptBins_CE.Fill(gentau.Pt())

                    minatedMatched = False
                    highestMinatedL1Pt = -99.9

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

                    # fill numerator histograms for every thresholds
                    for i, thr in enumerate(online_thresholds): 
                        if minatedMatched and highestMinatedL1Pt>float(thr):
                                minated_passing_ptBins_CE[i].Fill(gentau.Pt())


            ###########################################################################
            # OVERALL TURNON
            maxPt = -99.
            for genPt, genEta in zip(_gentau_visPt, _gentau_visEta):
                # if abs(genEta) < 1.5: continue
                if genPt > maxPt: maxPt = genPt

            # if maxPt < 0:
            #     for genPt, genEta in zip(_gentau_visPt, _gentau_visEta):
            #         if genPt > maxPt: maxPt = genPt

            if maxPt > 0:
                maxEta = _gentau_visEta[_gentau_visPt.index(maxPt)]
                maxPhi = _gentau_visPhi[_gentau_visPt.index(maxPt)]

                gentau = ROOT.TLorentzVector()
                gentau.SetPtEtaPhiM(maxPt, maxEta, maxPhi, 0)

                if abs(gentau.Eta()) < options.etaEr: # skip taus out of acceptance

                    denominator_ptBins.Fill(gentau.Pt())
                    if gentau.Pt() > 40: denominator_etaBins.Fill(gentau.Eta())

                    minatedMatched = False
                    highestMinatedL1Pt = -99.9

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

                    # fill numerator histograms for every thresholds
                    for i, thr in enumerate(online_thresholds): 
                        if minatedMatched and highestMinatedL1Pt>float(thr):
                            minated_passing_ptBins[i].Fill(gentau.Pt())
                            if gentau.Pt() > 40: minated_passing_etaBins[i].Fill(gentau.Eta())


        # end of the loop over the events
        #################################

        # TGraphAsymmErrors for efficiency turn-ons
        turnonsMinated = []
        turnonsMinated_CB = []
        turnonsMinated_CE = []
        etaEffMinated = []

        for i, thr in enumerate(online_thresholds):
            turnonsMinated.append(ROOT.TGraphAsymmErrors(minated_passing_ptBins[i], denominator_ptBins, "cp"))
            turnonsMinated_CB.append(ROOT.TGraphAsymmErrors(minated_passing_ptBins_CB[i], denominator_ptBins_CB, "cp"))
            turnonsMinated_CE.append(ROOT.TGraphAsymmErrors(minated_passing_ptBins_CE[i], denominator_ptBins_CE, "cp"))

            etaEffMinated.append(ROOT.TGraphAsymmErrors(minated_passing_etaBins[i], denominator_etaBins, "cp"))

        # save to file 

        fileout = ROOT.TFile(perfdir+"/turnons"+tag+"/efficiency_graphs"+tag+"_er"+str(options.etaEr)+".root","RECREATE")
        denominator_ptBins.Write()
        denominator_ptBins_CB.Write()
        denominator_ptBins_CE.Write()
        denominator_etaBins.Write()
        for i, thr in enumerate(online_thresholds): 
            minated_passing_ptBins[i].Write()
            minated_passing_ptBins_CB[i].Write()
            minated_passing_ptBins_CE[i].Write()

            minated_passing_etaBins[i].Write()

            turnonsMinated[i].Write()
            turnonsMinated_CB[i].Write()
            turnonsMinated_CE[i].Write()

            etaEffMinated[i].Write()


        fileout.Close()

    else:
        filein = ROOT.TFile(perfdir+"/turnons"+tag+"/efficiency_graphs"+tag+"_er"+str(options.etaEr)+".root","READ")

        # TGraphAsymmErrors for efficiency turn-ons
        turnonsMinated = []
        turnonsMinated_CB = []
        turnonsMinated_CE = []

        etaEffMinated = []

        for i, thr in enumerate(online_thresholds):
            turnonsMinated.append(filein.Get("divide_minated_passing_thr"+str(int(thr))+"_ptBins_by_denominator_ptBins"))
            turnonsMinated_CB.append(filein.Get("divide_minated_passing_thr"+str(int(thr))+"_ptBins_CB_by_denominator_ptBins_CB"))
            turnonsMinated_CE.append(filein.Get("divide_minated_passing_thr"+str(int(thr))+"_ptBins_CE_by_denominator_ptBins_CE"))

            etaEffMinated.append(filein.Get("divide_minated_passing_thr"+str(int(thr))+"_etaBins_by_denominator_etaBins"))


    turnons_dict = {}
    turnons_dict_CB = {}
    turnons_dict_CE = {}
    etaeffs_dict = {}
    mapping_dict = {'threshold':[],
                    'wp'+WP+'_pt95':[], 'wp'+WP+'_pt90':[], 'wp'+WP+'_pt50':[],
                    'wp'+WP+'_slope_pt95' : 0.0, 'wp'+WP+'_intercept_pt95': 0.0,
                    'wp'+WP+'_slope_pt90' : 0.0, 'wp'+WP+'_intercept_pt90': 0.0,
                    'wp'+WP+'_slope_pt50' : 0.0, 'wp'+WP+'_intercept_pt50': 0.0}
    mapping_dict_CB = {'threshold':[],
                       'wp'+WP+'_pt95':[], 'wp'+WP+'_pt90':[], 'wp'+WP+'_pt50':[],
                       'wp'+WP+'_slope_pt95' : 0.0, 'wp'+WP+'_intercept_pt95': 0.0,
                       'wp'+WP+'_slope_pt90' : 0.0, 'wp'+WP+'_intercept_pt90': 0.0,
                       'wp'+WP+'_slope_pt50' : 0.0, 'wp'+WP+'_intercept_pt50': 0.0}
    mapping_dict_CE = {'threshold':[],
                       'wp'+WP+'_pt95':[], 'wp'+WP+'_pt90':[], 'wp'+WP+'_pt50':[],
                       'wp'+WP+'_slope_pt95' : 0.0, 'wp'+WP+'_intercept_pt95': 0.0,
                       'wp'+WP+'_slope_pt90' : 0.0, 'wp'+WP+'_intercept_pt90': 0.0,
                       'wp'+WP+'_slope_pt50' : 0.0, 'wp'+WP+'_intercept_pt50': 0.0}

    for i, thr in enumerate(online_thresholds):
        turnons_dict['turnonAtwpAt'+str(thr)+'GeV'] = [[],[],[]]
        turnons_dict_CB['turnonAtwpAt'+str(thr)+'GeV'] = [[],[],[]]
        turnons_dict_CE['turnonAtwpAt'+str(thr)+'GeV'] = [[],[],[]]

        etaeffs_dict['efficiencyVsEtaAtwpAt'+str(thr)+'GeV'] = [[],[],[]]

        for ibin in range(0,turnonsMinated[i].GetN()):
            turnons_dict['turnonAtwpAt'+str(thr)+'GeV'][0].append(turnonsMinated[i].GetPointY(ibin))
            turnons_dict['turnonAtwpAt'+str(thr)+'GeV'][1].append(turnonsMinated[i].GetErrorYlow(ibin))
            turnons_dict['turnonAtwpAt'+str(thr)+'GeV'][2].append(turnonsMinated[i].GetErrorYhigh(ibin))
            turnons_dict_CB['turnonAtwpAt'+str(thr)+'GeV'][0].append(turnonsMinated_CB[i].GetPointY(ibin))
            turnons_dict_CB['turnonAtwpAt'+str(thr)+'GeV'][1].append(turnonsMinated_CB[i].GetErrorYlow(ibin))
            turnons_dict_CB['turnonAtwpAt'+str(thr)+'GeV'][2].append(turnonsMinated_CB[i].GetErrorYhigh(ibin))
            turnons_dict_CE['turnonAtwpAt'+str(thr)+'GeV'][0].append(turnonsMinated_CE[i].GetPointY(ibin))
            turnons_dict_CE['turnonAtwpAt'+str(thr)+'GeV'][1].append(turnonsMinated_CE[i].GetErrorYlow(ibin))
            turnons_dict_CE['turnonAtwpAt'+str(thr)+'GeV'][2].append(turnonsMinated_CE[i].GetErrorYhigh(ibin))

        for ibin in range(0,etaEffMinated[i].GetN()):
            etaeffs_dict['efficiencyVsEtaAtwpAt'+str(thr)+'GeV'][0].append(etaEffMinated[i].GetPointY(ibin))
            etaeffs_dict['efficiencyVsEtaAtwpAt'+str(thr)+'GeV'][1].append(etaEffMinated[i].GetErrorYlow(ibin))
            etaeffs_dict['efficiencyVsEtaAtwpAt'+str(thr)+'GeV'][2].append(etaEffMinated[i].GetErrorYhigh(ibin))
        
        # ONLINE TO OFFILNE MAPPING AS MENU DOES
        # efficiency = turnons_dict['turnonAtwpAt'+str(thr)+'GeV'][0]
        # er_dn = turnons_dict['turnonAtwpAt'+str(thr)+'GeV'][1]
        # er_up = turnons_dict['turnonAtwpAt'+str(thr)+'GeV'][2]
        # xbins = 0.5 * np.array(ptBins[1:] + ptBins[:-1])
        # K1 = []
        # for i in range(len(efficiency)): K1.append(1 / (er_dn[i] + er_up[i]) / (er_up[i] + er_dn[i]))
        # mapping_dict['wp'+WP+'_pt95'].append(_find_turnon_cut(xbins, _interpolate(efficiency, K1, 100), 0.95))
        # mapping_dict['wp'+WP+'_pt90'].append(_find_turnon_cut(xbins, _interpolate(efficiency, K1, 100), 0.90))
        # mapping_dict['wp'+WP+'_pt50'].append(_find_turnon_cut(xbins, _interpolate(efficiency, K1, 100), 0.50))

        # efficiency = turnons_dict_CB['turnonAtwpAt'+str(thr)+'GeV'][0]
        # er_dn = turnons_dict_CB['turnonAtwpAt'+str(thr)+'GeV'][1]
        # er_up = turnons_dict_CB['turnonAtwpAt'+str(thr)+'GeV'][2]
        # xbins = 0.5 * np.array(ptBins[1:] + ptBins[:-1])
        # K1 = []
        # for i in range(len(efficiency)): K1.append(1 / (er_dn[i] + er_up[i]) / (er_up[i] + er_dn[i]))
        # mapping_dict_CB['wp'+WP+'_pt95'].append(_find_turnon_cut(xbins, _interpolate(efficiency, K1, 100), 0.95))
        # mapping_dict_CB['wp'+WP+'_pt90'].append(_find_turnon_cut(xbins, _interpolate(efficiency, K1, 100), 0.90))
        # mapping_dict_CB['wp'+WP+'_pt50'].append(_find_turnon_cut(xbins, _interpolate(efficiency, K1, 100), 0.50))

        # efficiency = turnons_dict_CE['turnonAtwpAt'+str(thr)+'GeV'][0]
        # er_dn = turnons_dict_CE['turnonAtwpAt'+str(thr)+'GeV'][1]
        # er_up = turnons_dict_CE['turnonAtwpAt'+str(thr)+'GeV'][2]
        # xbins = 0.5 * np.array(ptBins[1:] + ptBins[:-1])
        # K1 = []
        # for i in range(len(efficiency)): K1.append(1 / (er_dn[i] + er_up[i]) / (er_up[i] + er_dn[i]))
        # mapping_dict_CE['wp'+WP+'_pt95'].append(_find_turnon_cut(xbins, _interpolate(efficiency, K1, 100), 0.95))
        # mapping_dict_CE['wp'+WP+'_pt90'].append(_find_turnon_cut(xbins, _interpolate(efficiency, K1, 100), 0.90))
        # mapping_dict_CE['wp'+WP+'_pt50'].append(_find_turnon_cut(xbins, _interpolate(efficiency, K1, 100), 0.50))


        # ONLINE TO OFFILNE MAPPING AS I DO
        mapping_dict['wp'+WP+'_pt95'].append(np.interp(0.95, turnons_dict['turnonAtwpAt'+str(thr)+'GeV'][0], offline_pts)) #,right=-99,left=-98)
        mapping_dict['wp'+WP+'_pt90'].append(np.interp(0.90, turnons_dict['turnonAtwpAt'+str(thr)+'GeV'][0], offline_pts)) #,right=-99,left=-98)
        mapping_dict['wp'+WP+'_pt50'].append(np.interp(0.50, turnons_dict['turnonAtwpAt'+str(thr)+'GeV'][0], offline_pts)) #,right=-99,left=-98)

        mapping_dict_CB['wp'+WP+'_pt95'].append(np.interp(0.95, turnons_dict_CB['turnonAtwpAt'+str(thr)+'GeV'][0], offline_pts)) #,right=-99,left=-98)
        mapping_dict_CB['wp'+WP+'_pt90'].append(np.interp(0.90, turnons_dict_CB['turnonAtwpAt'+str(thr)+'GeV'][0], offline_pts)) #,right=-99,left=-98)
        mapping_dict_CB['wp'+WP+'_pt50'].append(np.interp(0.50, turnons_dict_CB['turnonAtwpAt'+str(thr)+'GeV'][0], offline_pts)) #,right=-99,left=-98)

        mapping_dict_CE['wp'+WP+'_pt95'].append(np.interp(0.95, turnons_dict_CE['turnonAtwpAt'+str(thr)+'GeV'][0], offline_pts)) #,right=-99,left=-98)
        mapping_dict_CE['wp'+WP+'_pt90'].append(np.interp(0.90, turnons_dict_CE['turnonAtwpAt'+str(thr)+'GeV'][0], offline_pts)) #,right=-99,left=-98)
        mapping_dict_CE['wp'+WP+'_pt50'].append(np.interp(0.50, turnons_dict_CE['turnonAtwpAt'+str(thr)+'GeV'][0], offline_pts)) #,right=-99,left=-98)

    ##################################################################################
    # FIT SCALINGS

    def line(x, A, B):
        return A * x + B

    p0 = [1,1]
    popt,    pcov = curve_fit(line, online_thresholds[25:100], mapping_dict['wp'+WP+'_pt95'][25:100], p0, maxfev=5000)
    popt_CB, pcov = curve_fit(line, online_thresholds[25:100], mapping_dict_CB['wp'+WP+'_pt95'][25:100], p0, maxfev=5000)
    popt_CE, pcov = curve_fit(line, online_thresholds[25:60],  mapping_dict_CE['wp'+WP+'_pt95'][25:60], p0, maxfev=5000)
    mapping_dict['wp'+WP+'_slope_pt95'] = popt[0]
    mapping_dict['wp'+WP+'_intercept_pt95'] = popt[1]
    mapping_dict_CB['wp'+WP+'_slope_pt95'] = popt_CB[0]
    mapping_dict_CB['wp'+WP+'_intercept_pt95'] = popt_CB[1]
    mapping_dict_CE['wp'+WP+'_slope_pt95'] = popt_CE[0]
    mapping_dict_CE['wp'+WP+'_intercept_pt95'] = popt_CE[1]
    print('\n50% efficiency scaling')
    print('Overall : y =', round(popt[0], 3), '* x +', round(popt[1],3))
    print('Barrel : y =', round(popt_CB[0], 3), '* x +', round(popt_CB[1],3))
    print('Endcap : y =', round(popt_CE[0], 3), '* x +', round(popt_CE[1],3))

    popt,    pcov = curve_fit(line, online_thresholds[25:100], mapping_dict['wp'+WP+'_pt90'][25:100], p0, maxfev=5000)
    popt_CB, pcov = curve_fit(line, online_thresholds[25:100], mapping_dict_CB['wp'+WP+'_pt90'][25:100], p0, maxfev=5000)
    popt_CE, pcov = curve_fit(line, online_thresholds[25:60],  mapping_dict_CE['wp'+WP+'_pt90'][25:60], p0, maxfev=5000)
    mapping_dict['wp'+WP+'_slope_pt90'] = popt[0]
    mapping_dict['wp'+WP+'_intercept_pt90'] = popt[1]
    mapping_dict_CB['wp'+WP+'_slope_pt90'] = popt_CB[0]
    mapping_dict_CB['wp'+WP+'_intercept_pt90'] = popt_CB[1]
    mapping_dict_CE['wp'+WP+'_slope_pt90'] = popt_CE[0]
    mapping_dict_CE['wp'+WP+'_intercept_pt90'] = popt_CE[1]
    print('\n90% efficiency scaling')
    print('Overall : y =', round(popt[0], 3), '* x +', round(popt[1],3))
    print('Barrel : y =', round(popt_CB[0], 3), '* x +', round(popt_CB[1],3))
    print('Endcap : y =', round(popt_CE[0], 3), '* x +', round(popt_CE[1],3))

    popt,    pcov = curve_fit(line, online_thresholds[25:100], mapping_dict['wp'+WP+'_pt50'][25:100], p0, maxfev=5000)
    popt_CB, pcov = curve_fit(line, online_thresholds[25:100], mapping_dict_CB['wp'+WP+'_pt50'][25:100], p0, maxfev=5000)
    popt_CE, pcov = curve_fit(line, online_thresholds[25:100],  mapping_dict_CE['wp'+WP+'_pt50'][25:100], p0, maxfev=5000)
    mapping_dict['wp'+WP+'_slope_pt50'] = popt[0]
    mapping_dict['wp'+WP+'_intercept_pt50'] = popt[1]
    mapping_dict_CB['wp'+WP+'_slope_pt50'] = popt_CB[0]
    mapping_dict_CB['wp'+WP+'_intercept_pt50'] = popt_CB[1]
    mapping_dict_CE['wp'+WP+'_slope_pt50'] = popt_CE[0]
    mapping_dict_CE['wp'+WP+'_intercept_pt50'] = popt_CE[1]
    print('\n50% efficiency scaling')
    print('Overall : y =', round(popt[0], 3), '* x +', round(popt[1],3))
    print('Barrel : y =', round(popt_CB[0], 3), '* x +', round(popt_CB[1],3))
    print('Endcap : y =', round(popt_CE[0], 3), '* x +', round(popt_CE[1],3))

    save_obj(mapping_dict, perfdir+'/turnons'+tag+'/online2offline_mapping'+WP+'.pkl')
    save_obj(mapping_dict_CB, perfdir+'/turnons'+tag+'/online2offline_mapping'+WP+'_CB.pkl')
    save_obj(mapping_dict_CE, perfdir+'/turnons'+tag+'/online2offline_mapping'+WP+'_CE.pkl')


    ##################################################################################
    # PLOT TURNONS

    x_Emyr_CB = [9.032258064516128, 15.11921458625526, 21.00981767180925, 27.09677419354839, 32.987377279102375, 38.87798036465638, 45.16129032258064, 51.051893408134646, 56.94249649368864, 63.02945301542778, 68.92005610098177, 75.0070126227209, 80.8976157082749, 86.98457223001401, 93.07152875175315, 98.96213183730715, 105.24544179523141, 110.93969144460027, 116.83029453015428, 123.11360448807852, 129.00420757363253, 135.09116409537165, 140.98176718092566, 146.87237026647966]
    y_Emyr_CB = [0.04236006051437213, 0.24810892586989397, 0.6066565809379727, 0.7715582450832071, 0.8865355521936459, 0.9273827534039334, 0.9531013615733736, 0.956127080181543, 0.966717095310136, 0.966717095310136, 0.9697428139183055, 0.9621785173978818, 0.9757942511346444, 0.9803328290468986, 0.9818456883509833, 0.9788199697428138, 0.9848714069591527, 0.9818456883509833, 0.9833585476550679, 0.9818456883509833, 0.9848714069591527, 0.9773071104387291, 0.9712556732223903, 0.9863842662632374]

    x_Emyr_CE = [2.9453015427769955, 8.835904628330997, 14.922861150070126, 21.00981767180925, 27.09677419354839, 32.987377279102375, 38.87798036465638, 44.964936886395506, 51.051893408134646, 56.94249649368864, 63.02945301542778, 68.92005610098177, 75.0070126227209, 80.8976157082749, 86.78821879382889, 92.87517531556801, 98.96213183730715, 105.04908835904628, 110.93969144460027, 116.83029453015428, 122.9172510518934, 128.8078541374474, 134.89481065918653, 140.98176718092566, 146.87237026647966]
    y_Emyr_CE = [0.021180030257186067, 0.04689863842662634, 0.25113464447806344, 0.475037821482602, 0.6369137670196671, 0.7473524962178517, 0.8033282904689862, 0.8184568835098335, 0.8668683812405445, 0.8835098335854765, 0.8865355521936459, 0.8835098335854765, 0.8940998487140694, 0.9077155824508321, 0.9213313161875945, 0.9137670196671708, 0.9258698940998487, 0.9213313161875945, 0.9425113464447805, 0.8850226928895611, 0.939485627836611, 0.9198184568835097, 0.9273827534039334, 0.9697428139183055, 0.8577912254160363]

    i = 0
    plt.figure(figsize=(10,10))
    thr=0
    plt.errorbar(offline_pts,turnons_dict_CB['turnonAtwpAt'+str(thr)+'GeV'][0],xerr=1,yerr=[turnons_dict_CB['turnonAtwpAt'+str(thr)+'GeV'][1], turnons_dict_CB['turnonAtwpAt'+str(thr)+'GeV'][2]], ls='-', label=r'CB - $p_{T}^{L1 \tau} > %i$ GeV' % (thr), lw=2, marker='o', color='green', alpha=0.75)
    plt.errorbar(offline_pts,turnons_dict_CE['turnonAtwpAt'+str(thr)+'GeV'][0],xerr=1,yerr=[turnons_dict_CE['turnonAtwpAt'+str(thr)+'GeV'][1], turnons_dict_CE['turnonAtwpAt'+str(thr)+'GeV'][2]], ls='-', label=r'CE - $p_{T}^{L1 \tau} > %i$ GeV' % (thr), lw=2, marker='o', color='red', alpha=0.75)
    plt.errorbar(x_Emyr_CB,y_Emyr_CB, xerr=1, yerr=0.01, ls='--', label=r'CB Menu - $p_{T}^{L1 \tau} > %i$ GeV' % (thr), lw=2, marker='o', color='green')
    plt.errorbar(x_Emyr_CE,y_Emyr_CE, xerr=1, yerr=0.01, ls='--', label=r'CE Menu - $p_{T}^{L1 \tau} > %i$ GeV' % (thr), lw=2, marker='o', color='red')
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
    plt.savefig(perfdir+'/turnons'+tag+'/turnons_Emyr_'+WP+'.pdf')
    plt.close()


    cmap = matplotlib.cm.get_cmap('tab20c'); i=0
    
    i = 0
    plt.figure(figsize=(10,10))
    for thr in plotting_thresholds:
        if not thr%10:
            plt.errorbar(offline_pts,turnons_dict['turnonAtwpAt'+str(thr)+'GeV'][0],xerr=1,yerr=[turnons_dict['turnonAtwpAt'+str(thr)+'GeV'][1], turnons_dict['turnonAtwpAt'+str(thr)+'GeV'][2]], ls='None', label=r'$p_{T}^{L1 \tau} > %i$ GeV' % (thr), lw=2, marker='o', color=cmap(i))

            p0 = [1, thr, 1] 
            popt, pcov = curve_fit(sigmoid, offline_pts, turnons_dict['turnonAtwpAt'+str(thr)+'GeV'][0], p0, maxfev=5000)
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
            plt.errorbar(offline_pts,turnons_dict_CB['turnonAtwpAt'+str(thr)+'GeV'][0],xerr=1,yerr=[turnons_dict_CB['turnonAtwpAt'+str(thr)+'GeV'][1], turnons_dict_CB['turnonAtwpAt'+str(thr)+'GeV'][2]], ls='None', label=r'$p_{T}^{L1 \tau} > %i$ GeV' % (thr), lw=2, marker='o', color=cmap(i))

            p0 = [1, thr, 1] 
            popt, pcov = curve_fit(sigmoid, offline_pts, turnons_dict_CB['turnonAtwpAt'+str(thr)+'GeV'][0], p0, maxfev=5000)
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
            plt.errorbar(offline_pts,turnons_dict_CE['turnonAtwpAt'+str(thr)+'GeV'][0],xerr=1,yerr=[turnons_dict_CE['turnonAtwpAt'+str(thr)+'GeV'][1], turnons_dict_CE['turnonAtwpAt'+str(thr)+'GeV'][2]], ls='None', label=r'$p_{T}^{L1 \tau} > %i$ GeV' % (thr), lw=2, marker='o', color=cmap(i))

            p0 = [1, thr, 1] 
            popt, pcov = curve_fit(sigmoid, offline_pts, turnons_dict_CE['turnonAtwpAt'+str(thr)+'GeV'][0], p0, maxfev=5000)
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
    # PLOT EFFICIENCIES VS ETA
    i = 0
    plt.figure(figsize=(10,10))
    plt.errorbar(eta_bins_centers,etaeffs_dict['efficiencyVsEtaAtwpAt0GeV'][0],yerr=[etaeffs_dict['efficiencyVsEtaAtwpAt0GeV'][1], etaeffs_dict['efficiencyVsEtaAtwpAt0GeV'][2]], ls='None', label=r'$p_{T}^{L1 \tau} > %i$ GeV' % (32), lw=2, marker='o', color=cmap(i))
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

    ##################################################################################
    # PLOT ONLINE TO OFFLINE MAPPING
    plt.figure(figsize=(10,10))
    plt.plot(online_thresholds, mapping_dict['wp'+WP+'_pt95'], label='@ 95% efficiency', linewidth=2, color='blue')
    plt.plot(online_thresholds, mapping_dict['wp'+WP+'_pt90'], label='@ 90% efficiency', linewidth=2, color='red')
    plt.plot(online_thresholds, mapping_dict['wp'+WP+'_pt50'], label='@ 50% efficiency', linewidth=2, color='green')
    plt.legend(loc = 'lower right', fontsize=14)
    plt.xlabel('L1 Threshold [GeV]')
    plt.ylabel('Offline threshold [GeV]')
    plt.xlim(20, 100)
    plt.ylim(20, 200)
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(perfdir+'/turnons'+tag+'/online2offline_WP'+WP+'.pdf')
    plt.close()

    p0 = [1,1]
    popt, pcov = curve_fit(line, online_thresholds[25:100], mapping_dict['wp'+WP+'_pt90'][25:100], p0, maxfev=5000)
    # print('Overall : y =', round(popt[0], 3), '* x +', round(popt[1],3))
    popt_CB, pcov = curve_fit(line, online_thresholds[25:100], mapping_dict_CB['wp'+WP+'_pt90'][25:100], p0, maxfev=5000)
    # print('Barrel : y =', round(popt_CB[0], 3), '* x +', round(popt_CB[1],3))
    popt_CE, pcov = curve_fit(line, online_thresholds[25:60], mapping_dict_CE['wp'+WP+'_pt90'][25:60], p0, maxfev=5000)
    # print('Endcap : y =', round(popt_CE[0], 3), '* x +', round(popt_CE[1],3))

    plt.figure(figsize=(10,10))
    plt.plot(online_thresholds, mapping_dict['wp'+WP+'_pt90'], label='Overall', linewidth=2, color='black')
    plt.plot(online_thresholds, mapping_dict_CB['wp'+WP+'_pt90'], label='Barrel', linewidth=2, color='green')
    plt.plot(online_thresholds, mapping_dict_CE['wp'+WP+'_pt90'], label='Endcap', linewidth=2, color='red')
    plt.plot(online_thresholds, line(online_thresholds, *popt), label='Fit Overall', linewidth=2, ls='--', color='black')
    plt.plot(online_thresholds, line(online_thresholds, *popt_CB), label='Fit Barrel', linewidth=2, ls='--', color='green')
    plt.plot(online_thresholds, line(online_thresholds, *popt_CE), label='Fit Endcap', linewidth=2, ls='--', color='red')
    plt.legend(loc = 'lower right', fontsize=14)
    plt.xlabel('L1 Threshold [GeV]')
    plt.ylabel('Offline threshold [GeV]')
    # plt.xlim(20, 100)
    # plt.ylim(20, 200)
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(perfdir+'/turnons'+tag+'/online2offline_WP'+WP+'_pt90_split.pdf')
    plt.close()

    p0 = [1,1]
    popt, pcov = curve_fit(line, online_thresholds[25:100], mapping_dict['wp'+WP+'_pt50'][25:100], p0, maxfev=5000)
    # print('Overall : y =', round(popt[0], 3), '* x +', round(popt[1],3))
    popt_CB, pcov = curve_fit(line, online_thresholds[25:100], mapping_dict_CB['wp'+WP+'_pt50'][25:100], p0, maxfev=5000)
    # print('Barrel : y =', round(popt_CB[0], 3), '* x +', round(popt_CB[1],3))
    popt_CE, pcov = curve_fit(line, online_thresholds[25:100], mapping_dict_CE['wp'+WP+'_pt50'][25:100], p0, maxfev=5000)
    # print('Endcap : y =', round(popt_CE[0], 3), '* x +', round(popt_CE[1],3))

    plt.figure(figsize=(10,10))
    plt.plot(online_thresholds, mapping_dict['wp'+WP+'_pt50'], label='Overall', linewidth=2, color='black')
    plt.plot(online_thresholds, mapping_dict_CB['wp'+WP+'_pt50'], label='Barrel', linewidth=2, color='green')
    plt.plot(online_thresholds, mapping_dict_CE['wp'+WP+'_pt50'], label='Endcap', linewidth=2, color='red')
    plt.plot(online_thresholds, line(online_thresholds, *popt), label='Fit Overall', linewidth=2, ls='--', color='black')
    plt.plot(online_thresholds, line(online_thresholds, *popt_CB), label='Fit Barrel', linewidth=2, ls='--', color='green')
    plt.plot(online_thresholds, line(online_thresholds, *popt_CE), label='Fit Endcap', linewidth=2, ls='--', color='red')
    plt.legend(loc = 'lower right', fontsize=14)
    plt.xlabel('L1 Threshold [GeV]')
    plt.ylabel('Offline threshold [GeV]')
    # plt.xlim(20, 100)
    # plt.ylim(20, 200)
    plt.grid()
    mplhep.cms.label('Phase-2 Simulation', data=True, rlabel='14 TeV, 200 PU')
    plt.savefig(perfdir+'/turnons'+tag+'/online2offline_WP'+WP+'_pt50_split.pdf')
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


    

 